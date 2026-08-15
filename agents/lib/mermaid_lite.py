"""agents/lib/mermaid_lite.py — In-process Mermaid flowchart renderer.

Parses a *subset* of Mermaid flowchart syntax and renders it directly to
SVG using a layered (Sugiyama-style) layout — no external CLI, no Node,
no headless Chromium. Only dependency is `networkx` (pure Python).

Supported syntax:
    graph TD | graph LR | flowchart TB | ...  (TD/TB, BT, LR, RL)
    A[Rect Label]
    A(Rounded Label)
    A((Circle Label))
    A{Diamond Label}
    A --> B
    A --> B --> C            (chained edges)
    A -->|label| B           (labeled edge)
    A -.-> B                 (dashed edge)
    A ==> B                  (thick edge)
    A --- B                  (line, no arrowhead)
    %% comment                (ignored)

Not supported (silently skipped, does not error): subgraph blocks,
classDef/class/click/style directives, click handlers. Good enough for
LLM-generated flowcharts, which rarely use those.

Usage:
    from agents.lib.mermaid_lite import render_mermaid_to_svg
    svg_string = render_mermaid_to_svg(mermaid_script_text)
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

# Palette pulled directly from the app's own design tokens (chat.css :root),
# so a diagram dropped into the chat transcript reads as part of the product
# rather than a differently-branded insert. One restrained accent per shape
# role (process / decision / start-end) instead of a rainbow per shape type.
COLORS = {
    "bg": "#ffffff",
    "rect_fill": "#eaf1ff",       # --accent-wash
    "rect_stroke": "#2563eb",     # --accent
    "rounded_fill": "#f1f5f9",    # --surface-2
    "rounded_stroke": "#475569",  # --ink-2
    "diamond_fill": "#fff7ed",
    "diamond_stroke": "#d97706",  # --warn
    "circle_fill": "#eaf1ff",     # --accent-wash
    "circle_stroke": "#1d4ed8",   # --accent-ink
    "text": "#0f172a",            # --ink
    "edge": "#7c8da5",            # --ink-3
    "edge_label_bg": "#ffffff",
}

CHAR_WIDTH_PX = 7.2          # rough monospace-ish estimate at font-size 14
LINE_HEIGHT_PX = 18
NODE_PAD_X = 22
NODE_PAD_Y = 14
MIN_NODE_W = 90
MIN_NODE_H = 46
LAYER_GAP = 110               # gap between layers (direction axis)
NODE_GAP = 40                 # gap between nodes within a layer
WRAP_CHARS = 22               # wrap node label after this many chars
MARGIN = 40


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    label: str
    shape: str = "rect"       # rect | rounded | diamond | circle
    lines: List[str] = field(default_factory=list)
    w: float = MIN_NODE_W
    h: float = MIN_NODE_H
    x: float = 0.0
    y: float = 0.0
    layer: int = 0


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    style: str = "solid"      # solid | dashed | thick


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DIRECTION_RE = re.compile(r"^\s*(?:graph|flowchart)\s+(TD|TB|BT|LR|RL)\b", re.IGNORECASE)
_SKIP_PREFIXES = ("subgraph", "end", "classdef", "class ", "click ", "style ", "%%", "linkstyle")

# Other Mermaid diagram types this parser does not implement. Without this
# check, a line like "sequenceDiagram" matches the bare-node-reference pattern
# below (it's just word characters) and silently becomes a single stray node
# instead of failing — producing a "successful" one-box diagram that isn't
# what was asked for, and giving the caller no signal to fall back to a full
# Mermaid renderer. Checked once, up front, against the first meaningful line.
_OTHER_DIAGRAM_TYPES = (
    "sequencediagram", "classdiagram", "statediagram", "statediagram-v2",
    "erdiagram", "journey", "gantt", "pie", "quadrantchart", "requirementdiagram",
    "gitgraph", "mindmap", "timeline", "sankey-beta", "c4context", "c4container",
    "c4component", "c4dynamic", "block-beta", "xychart-beta", "zenuml",
)

# Order matters: check the most specific (circle) before the more general (rounded).
_NODE_PATTERNS = [
    (re.compile(r"^([A-Za-z0-9_\-]+)\(\(\s*(.*?)\s*\)\)$"), "circle"),
    (re.compile(r"^([A-Za-z0-9_\-]+)\{\s*(.*?)\s*\}$"), "diamond"),
    (re.compile(r"^([A-Za-z0-9_\-]+)\(\s*(.*?)\s*\)$"), "rounded"),
    (re.compile(r"^([A-Za-z0-9_\-]+)\[\s*(.*?)\s*\]$"), "rect"),
    (re.compile(r"^([A-Za-z0-9_\-]+)$"), None),  # bare reference: shape unspecified
]

# Longest operators first so "-.->" isn't mis-split by "-->".
_EDGE_OP_RE = re.compile(r"(-\.->|-\.-|==>|===|-->|---)(?:\|([^|]*)\|)?")

_EDGE_STYLE_BY_OP = {
    "-->": "solid", "---": "solid",
    "-.->": "dashed", "-.-": "dashed",
    "==>": "thick", "===": "thick",
}


def _parse_node_spec(text: str) -> Optional[Tuple[str, Optional[str], str]]:
    text = text.strip()
    if not text:
        return None
    for pattern, shape in _NODE_PATTERNS:
        m = pattern.match(text)
        if m:
            node_id = m.group(1)
            label = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            return node_id, label, shape
    return None


def parse_mermaid(script: str) -> Tuple[str, Dict[str, Node], List[Edge]]:
    """Returns (direction, nodes_by_id, edges)."""
    script = re.sub(r"^```(?:mermaid)?\s*", "", script.strip(), flags=re.IGNORECASE)
    script = re.sub(r"\s*```$", "", script.strip())

    first_line = next((l.strip() for l in script.splitlines() if l.strip()), "")
    first_word = re.split(r"[\s{]", first_line, maxsplit=1)[0].lower()
    if first_word in _OTHER_DIAGRAM_TYPES:
        raise ValueError(
            f"'{first_word}' is not a flowchart — this renderer only supports "
            f"Mermaid 'graph'/'flowchart' syntax."
        )

    direction = "TD"
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    def ensure_node(node_id: str, label: Optional[str] = None, shape: Optional[str] = None) -> None:
        if node_id not in nodes:
            nodes[node_id] = Node(id=node_id, label=label or node_id, shape=shape or "rect")
        else:
            if label is not None:
                nodes[node_id].label = label
            if shape is not None:
                nodes[node_id].shape = shape

    for raw_line in script.splitlines():
        line = raw_line.strip().rstrip(";")
        if not line:
            continue

        dmatch = _DIRECTION_RE.match(line)
        if dmatch:
            direction = dmatch.group(1).upper()
            continue

        lowered = line.lower()
        if lowered.startswith(_SKIP_PREFIXES) or line.startswith("%%"):
            continue

        matches = list(_EDGE_OP_RE.finditer(line))
        if not matches:
            # A bare node declaration, e.g. "A[Start]"
            spec = _parse_node_spec(line)
            if spec:
                node_id, label, shape = spec
                ensure_node(node_id, label, shape)
            continue

        # Walk the chain: node, op, node, op, node ...
        segments: List[str] = []
        ops: List[Tuple[str, str]] = []  # (style, label)
        last_end = 0
        for m in matches:
            segments.append(line[last_end:m.start()])
            op_style = _EDGE_STYLE_BY_OP.get(m.group(1), "solid")
            op_label = (m.group(2) or "").strip()
            ops.append((op_style, op_label))
            last_end = m.end()
        segments.append(line[last_end:])

        parsed_ids: List[str] = []
        for seg in segments:
            spec = _parse_node_spec(seg)
            if not spec:
                parsed_ids.append(None)  # type: ignore[arg-type]
                continue
            node_id, label, shape = spec
            ensure_node(node_id, label, shape)
            parsed_ids.append(node_id)

        for i, (style, op_label) in enumerate(ops):
            src, dst = parsed_ids[i], parsed_ids[i + 1]
            if src and dst:
                edges.append(Edge(src=src, dst=dst, label=op_label, style=style))

    return direction, nodes, edges


# ---------------------------------------------------------------------------
# Sizing + layered layout
# ---------------------------------------------------------------------------

def _wrap_label(label: str) -> List[str]:
    wrapped = textwrap.wrap(label, width=WRAP_CHARS) or [""]
    return wrapped[:4]  # hard cap so a runaway label can't blow up layout


def _size_node(node: Node) -> None:
    node.lines = _wrap_label(node.label)
    text_w = max((len(l) for l in node.lines), default=1) * CHAR_WIDTH_PX
    text_h = len(node.lines) * LINE_HEIGHT_PX
    node.w = max(MIN_NODE_W, text_w + 2 * NODE_PAD_X)
    node.h = max(MIN_NODE_H, text_h + 2 * NODE_PAD_Y)
    if node.shape == "diamond":
        # Diamonds need extra room so the label clears the sloped edges.
        node.w *= 1.5
        node.h *= 1.5
    elif node.shape == "circle":
        side = max(node.w, node.h)
        node.w = node.h = side


def _assign_layers(nodes: Dict[str, Node], edges: List[Edge]) -> List[List[Node]]:
    g = nx.DiGraph()
    g.add_nodes_from(nodes.keys())
    g.add_edges_from((e.src, e.dst) for e in edges)

    if nx.is_directed_acyclic_graph(g):
        generations = list(nx.topological_generations(g))
    else:
        # Fall back to BFS layering from in-degree-0 roots; any leftover
        # (cyclic) nodes get appended as trailing layers so we still
        # produce a usable, if imperfect, picture instead of erroring out.
        roots = [n for n, d in g.in_degree() if d == 0] or list(g.nodes)[:1]
        layer_of: Dict[str, int] = {}
        frontier = list(roots)
        depth = 0
        while frontier:
            nxt = []
            for n in frontier:
                if n not in layer_of:
                    layer_of[n] = depth
                    nxt.extend(g.successors(n))
            frontier = [n for n in nxt if n not in layer_of]
            depth += 1
        for n in g.nodes:
            layer_of.setdefault(n, depth)
        max_layer = max(layer_of.values(), default=0)
        generations = [[] for _ in range(max_layer + 1)]
        for n, l in layer_of.items():
            generations[l].append(n)

    layered: List[List[Node]] = []
    for gen in generations:
        row = [nodes[n] for n in gen if n in nodes]
        if row:
            layered.append(row)
    for i, row in enumerate(layered):
        for node in row:
            node.layer = i
    return layered


def _position_nodes(layered: List[List[Node]], direction: str) -> Tuple[float, float]:
    for row in layered:
        for node in row:
            _size_node(node)

    axis_is_vertical = direction in ("TD", "TB", "BT")
    # axis_is_vertical: layers stack top-to-bottom (main axis = Y, sized by
    # node.h), and nodes within a layer spread left-to-right (cross axis =
    # X, sized by node.w). LR/RL is the transpose of this.

    # Position along the "cross" axis (spread within a layer) first.
    max_cross = 0.0
    for row in layered:
        cross = 0.0
        for node in row:
            span = node.w if axis_is_vertical else node.h
            if axis_is_vertical:
                node.x = cross + span / 2
            else:
                node.y = cross + span / 2
            cross += span + NODE_GAP
        cross -= NODE_GAP if row else 0
        max_cross = max(max_cross, cross)

    # Center each layer's nodes within the overall cross-axis extent.
    for row in layered:
        if not row:
            continue
        span = (row[-1].x + row[-1].w / 2) if axis_is_vertical else (row[-1].y + row[-1].h / 2)
        offset = (max_cross - span) / 2
        for node in row:
            if axis_is_vertical:
                node.x += offset
            else:
                node.y += offset

    # Position along the "main" axis (layer order).
    main_pos = 0.0
    for row in layered:
        layer_extent = max((node.h if axis_is_vertical else node.w) for node in row)
        for node in row:
            if axis_is_vertical:
                node.y = main_pos + layer_extent / 2
            else:
                node.x = main_pos + layer_extent / 2
        main_pos += layer_extent + LAYER_GAP

    if direction == "BT":
        max_y = max((n.y for row in layered for n in row), default=0)
        for row in layered:
            for n in row:
                n.y = max_y - n.y
    elif direction == "RL":
        max_x = max((n.x for row in layered for n in row), default=0)
        for row in layered:
            for n in row:
                n.x = max_x - n.x

    all_nodes = [n for row in layered for n in row]
    width = max((n.x + n.w / 2 for n in all_nodes), default=0) + MARGIN
    height = max((n.y + n.h / 2 for n in all_nodes), default=0) + MARGIN
    min_x = min((n.x - n.w / 2 for n in all_nodes), default=0)
    min_y = min((n.y - n.h / 2 for n in all_nodes), default=0)
    shift_x = MARGIN - min_x
    shift_y = MARGIN - min_y
    for n in all_nodes:
        n.x += shift_x
        n.y += shift_y

    return width + shift_x, height + shift_y


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _node_svg(node: Node) -> str:
    cx, cy = node.x, node.y
    w, h = node.w, node.h
    fill_key, stroke_key = {
        "rect": ("rect_fill", "rect_stroke"),
        "rounded": ("rounded_fill", "rounded_stroke"),
        "diamond": ("diamond_fill", "diamond_stroke"),
        "circle": ("circle_fill", "circle_stroke"),
    }[node.shape]
    fill, stroke = COLORS[fill_key], COLORS[stroke_key]

    if node.shape == "rect":
        shape_svg = (
            f'<rect x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#node-shadow)"/>'
        )
    elif node.shape == "rounded":
        r = h / 2
        shape_svg = (
            f'<rect x="{cx - w/2:.1f}" y="{cy - h/2:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{r:.1f}" ry="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#node-shadow)"/>'
        )
    elif node.shape == "circle":
        r = min(w, h) / 2
        shape_svg = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#node-shadow)"/>'
    else:  # diamond
        pts = f"{cx:.1f},{cy - h/2:.1f} {cx + w/2:.1f},{cy:.1f} {cx:.1f},{cy + h/2:.1f} {cx - w/2:.1f},{cy:.1f}"
        shape_svg = f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#node-shadow)"/>'

    n_lines = len(node.lines)
    start_y = cy - (n_lines - 1) * LINE_HEIGHT_PX / 2
    text_svg = "".join(
        f'<text x="{cx:.1f}" y="{start_y + i * LINE_HEIGHT_PX:.1f}" text-anchor="middle" '
        f'dominant-baseline="middle" font-family="{FONT_FAMILY}" font-size="14" '
        f'fill="{COLORS["text"]}">{_xml_escape(line)}</text>'
        for i, line in enumerate(node.lines)
    )
    return shape_svg + text_svg


def _edge_anchor(node: Node, toward: Node) -> Tuple[float, float]:
    """Point on node's boundary closest to `toward`'s center (rect approx for all shapes)."""
    dx, dy = toward.x - node.x, toward.y - node.y
    if dx == 0 and dy == 0:
        return node.x, node.y
    hw, hh = node.w / 2, node.h / 2
    if hw == 0 or hh == 0:
        return node.x, node.y
    scale = min(hw / abs(dx) if dx else float("inf"), hh / abs(dy) if dy else float("inf"))
    return node.x + dx * scale, node.y + dy * scale


def _edge_svg(edge: Edge, nodes: Dict[str, Node], back_edge_index: Dict[Tuple[str, str], int]) -> str:
    src, dst = nodes[edge.src], nodes[edge.dst]
    x1, y1 = _edge_anchor(src, dst)
    x2, y2 = _edge_anchor(dst, src)

    dash = ' stroke-dasharray="6,4"' if edge.style == "dashed" else ""
    width = 2.5 if edge.style == "thick" else 1.6

    # Loop-back / same-layer edges get bowed out on a curve so they don't
    # run straight through the forward-flow edges and nodes between them.
    is_back = dst.layer <= src.layer
    if is_back:
        key = (edge.src, edge.dst)
        idx = back_edge_index.get(key, 0)
        back_edge_index[key] = idx + 1
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        perp_x, perp_y = -dy / length, dx / length
        bow = 55 + idx * 35 + min(abs(dst.layer - src.layer), 4) * 12
        mx, my = (x1 + x2) / 2 + perp_x * bow, (y1 + y2) / 2 + perp_y * bow
        line_svg = (
            f'<path d="M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}" fill="none" '
            f'stroke="{COLORS["edge"]}" stroke-width="{width}"{dash} marker-end="url(#arrowhead)"/>'
        )
        label_mid = (mx, my)
    else:
        line_svg = (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{COLORS["edge"]}" stroke-width="{width}"{dash} marker-end="url(#arrowhead)"/>'
        )
        label_mid = ((x1 + x2) / 2, (y1 + y2) / 2)

    label_svg = ""
    if edge.label:
        mx, my = label_mid
        label_w = len(edge.label) * 6.5 + 10
        label_svg = (
            f'<rect x="{mx - label_w/2:.1f}" y="{my - 10:.1f}" width="{label_w:.1f}" height="18" '
            f'fill="{COLORS["edge_label_bg"]}" opacity="0.9"/>'
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="{FONT_FAMILY}" '
            f'font-size="12" fill="{COLORS["text"]}">{_xml_escape(edge.label)}</text>'
        )
    return line_svg + label_svg


def render_mermaid_to_svg(script: str) -> str:
    """Parses a Mermaid flowchart subset and returns a complete standalone SVG string."""
    direction, nodes, edges = parse_mermaid(script)
    if not nodes:
        raise ValueError("No nodes could be parsed from the supplied Mermaid script.")

    layered = _assign_layers(nodes, edges)
    width, height = _position_nodes(layered, direction)

    back_edge_index: Dict[Tuple[str, str], int] = {}
    edges_svg = "".join(
        _edge_svg(e, nodes, back_edge_index) for e in edges if e.src in nodes and e.dst in nodes
    )
    nodes_svg = "".join(_node_svg(n) for row in layered for n in row)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'width="{width:.0f}" height="{height:.0f}">'
        f'<defs>'
        f'<marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        f'orient="auto"><polygon points="0 0, 10 4, 0 8" fill="{COLORS["edge"]}"/></marker>'
        # Soft, low-spread shadow — deck-style depth without looking skeuomorphic.
        f'<filter id="node-shadow" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0f172a" flood-opacity="0.14"/>'
        f'</filter>'
        f'</defs>'
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" fill="{COLORS["bg"]}"/>'
        f"{edges_svg}{nodes_svg}"
        f"</svg>"
    )