"""I1 Extract — media-type extractors.

Contract: every extractor returns `Extracted` or raises `ExtractError`.
Nothing here writes to disk, calls a model, or touches Track A stages.

Optional dependencies degrade to a reason_code, never a crash:
    docx   -> python-docx
    html   -> trafilatura (preferred) | beautifulsoup4 (fallback)
    images -> MultimodalImageExtractor via Ollama cloud endpoint (NO tesseract)
    xlsx   -> deliberately DEFERRED (see XlsxExtractor)

Image extraction is TWO-PHASE by design:
    Phase 1 (I1a): prose extractors run — docx/html/md.
                   Embedded images are stubbed as <!-- image:pending:N --> markers.
    Phase 2 (I1b): ImageExtractionJob resolves all pending markers in parallel,
                   respecting the cloud endpoint's concurrency cap.
    The chunker only runs after both phases complete for a doc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ result


@dataclass(frozen=True)
class Extracted:
    text: str
    extractor: str
    extractor_version: str
    page_count: int | None = None
    quality: float = 1.0
    warnings: list[str] = field(default_factory=list)


class ExtractError(Exception):
    def __init__(self, reason_code: str, message: str = ""):
        super().__init__(message or reason_code)
        self.reason_code = reason_code


# ------------------------------------------------------------------ quality

_WS = re.compile(r"\s+")


def score_text(text: str, expected_min_chars: int = 200) -> tuple[float, list[str]]:
    """Cheap, deterministic extraction-quality heuristic in [0,1].

    Catches the silent failure mode: a file that 'extracted fine' but yielded
    ligature soup, one line of nav text, or a wall of replacement chars.
    """
    warn: list[str] = []
    n = len(text)
    if n == 0:
        return 0.0, ["empty_extraction"]

    alnum = sum(c.isalnum() or c.isspace() for c in text) / n
    if alnum < 0.75:
        warn.append("low_alphanumeric_ratio")

    bad = text.count("\ufffd") + text.count("\x00")
    if bad / n > 0.001:
        warn.append("replacement_chars_present")

    words = _WS.split(text.strip())
    avg_word = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word > 18:
        warn.append("no_word_boundaries")          # classic PDF/OCR failure
    if n < expected_min_chars:
        warn.append("suspiciously_short")

    score = 1.0
    score -= 0.35 * (alnum < 0.75)
    score -= 0.35 * (bad / n > 0.001)
    score -= 0.25 * (avg_word > 18)
    score -= 0.20 * (n < expected_min_chars)
    return max(0.0, round(score, 3)), warn


# ------------------------------------------------------------------ base


class Extractor:
    name = "base"
    version = "0"
    suffixes: tuple[str, ...] = ()
    bound = "cpu"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    def extract(self, path: Path) -> Extracted:  # pragma: no cover
        raise NotImplementedError


# ------------------------------------------------------------------ markdown


class MarkdownExtractor(Extractor):
    name, version = "native_md", "1.0"
    suffixes = (".md", ".markdown", ".txt")

    def extract(self, path: Path) -> Extracted:
        text = path.read_text(encoding="utf-8", errors="replace")
        q, w = score_text(text)
        return Extracted(text, self.name, self.version, quality=q, warnings=w)


# ------------------------------------------------------------------ docx


class DocxExtractor(Extractor):
    """Extracts text preserving heading hierarchy and tables.

    Embedded images are stubbed as <!-- image:pending:N:inline_shape --> and
    their raw bytes are written to an ImageBlobStore for Phase 2 (I1b) resolution.
    Callers that don't need image extraction can pass blob_store=None — stubs
    are still emitted so the resolver can run later if needed.
    """
    name, version = "python-docx", "1.0"
    suffixes = (".docx",)

    def extract(self, path: Path, blob_store=None, doc_id: str = "") -> Extracted:
        try:
            import docx  # type: ignore
            from docx.oxml.ns import qn  # type: ignore
        except ImportError as e:
            raise ExtractError("extractor_unavailable:python-docx") from e

        try:
            d = docx.Document(str(path))
        except Exception as e:
            raise ExtractError("docx_unreadable", str(e)) from e

        parts: list[str] = []
        warn: list[str] = []
        img_idx = 0

        # Walk body elements in document order to preserve image placement context
        for elem in d.element.body:
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if tag == "p":
                # Check for inline images in this paragraph
                for drawing in elem.findall(".//" + qn("a:blip"), elem.nsmap
                                            if hasattr(elem, "nsmap") else {}):
                    pass  # handled below via InlineShape walk

                # Extract paragraph text
                from docx.text.paragraph import Paragraph  # type: ignore
                para = Paragraph(elem, d)
                t = para.text.strip()
                style = (para.style.name or "").lower()

                # Inline shapes in this paragraph
                shapes = elem.findall(".//" + qn("wp:inline")) if hasattr(qn, "__call__") else []
                for shape in shapes:
                    # Get image relationship id
                    blip = shape.find(".//" + qn("a:blip"))
                    if blip is not None:
                        r_embed = blip.get(qn("r:embed"))
                        if r_embed and r_embed in d.part.rels:
                            rel = d.part.rels[r_embed]
                            if "image" in rel.reltype:
                                raw = rel.target_part.blob
                                stub = f"<!-- image:pending:{img_idx}:inline_shape -->"
                                parts.append(stub)
                                if blob_store and doc_id:
                                    blob_store.put(doc_id, img_idx, raw)
                                img_idx += 1
                                warn.append("contains_embedded_image")

                if not t:
                    continue
                if style.startswith("heading"):
                    lvl = "".join(ch for ch in style if ch.isdigit()) or "1"
                    parts.append(f"{'#' * min(int(lvl), 6)} {t}")
                else:
                    parts.append(t)

            elif tag == "tbl":
                from docx.table import Table  # type: ignore
                tbl = Table(elem, d)
                rows = [[c.text.strip().replace("\n", " ") for c in r.cells] for r in tbl.rows]
                if not rows:
                    continue
                head, *body = rows
                md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                md += ["| " + " | ".join(r) + " |" for r in body]
                parts.append(f"<!-- table:{img_idx} -->\n" + "\n".join(md))
                warn.append("contains_table")

        text = "\n\n".join(parts)
        q, w = score_text(text)
        if img_idx > 0:
            w = sorted(set(w + [f"embedded_images={img_idx}"]))
        return Extracted(text, self.name, self.version,
                         page_count=None, quality=q, warnings=sorted(set(w + warn)))


# ------------------------------------------------------------------ html


class HtmlExtractor(Extractor):
    name, version = "trafilatura", "1.0"
    suffixes = (".html", ".htm", ".xhtml")

    def extract(self, path: Path) -> Extracted:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, used = "", ""

        try:
            import trafilatura  # type: ignore
            text = trafilatura.extract(
                raw, include_tables=True, include_links=False,
                favor_recall=True, output_format="markdown",
            ) or ""
            used = "trafilatura"
        except ImportError:
            pass

        if not text.strip():
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except ImportError as e:
                raise ExtractError("extractor_unavailable:trafilatura|bs4") from e
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            used = "bs4_fallback"

        q, w = score_text(text)
        # HTML is the prompt-injection surface: mark it so I2/Q6 can fence it
        w = sorted(set(w + ["untrusted_html_source"]))
        return Extracted(text, used or self.name, self.version, quality=q, warnings=w)


# ------------------------------------------------------------------ images

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass as _dc


@_dc
class ImageConfig:
    """Resolved from env at construction — never hard-coded values here."""
    endpoint: str = "http://localhost:11434"
    model: str = "gemma4:31b"
    fallback_model: str = "gpt-oss:20b"
    quality_floor: float = 0.35
    timeout_s: int = 90
    max_retries: int = 3
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "ImageConfig":
        # Sourced from the app's central Settings (config.py) rather than raw
        # os.environ reads, so .env is the single place this is configured —
        # ollama_image_processing_model is the field to change, not an
        # OLLAMA_VISION_MODEL env var that lived only here.
        from config import get_settings
        cfg = get_settings()

        def _strip(model: str) -> str:
            """Registry uses 'ollama/gemma4:31b' but Ollama API wants 'gemma4:31b'."""
            return model.removeprefix("ollama/")

        primary = _strip(cfg.ollama_image_processing_model)
        return cls(
            endpoint=cfg.ollama_inference_url,
            model=primary,
            # No separate config field for a fallback vision model yet — default
            # to the primary (retry-with-backoff still applies on 5xx/timeout;
            # a real second model can be layered in via OLLAMA_VISION_FALLBACK
            # in .env without needing a Settings field, since pydantic-settings
            # ignores env vars it doesn't declare rather than erroring).
            fallback_model=_strip(os.environ.get("OLLAMA_VISION_FALLBACK", primary)),
            quality_floor=float(os.environ.get("OLLAMA_VISION_QUALITY_FLOOR", "0.35")),
            timeout_s=int(os.environ.get("OLLAMA_VISION_TIMEOUT_S", "90")),
            max_retries=int(os.environ.get("OLLAMA_VISION_MAX_RETRIES", "3")),
            api_key=cfg.ollama_inference_api_key,
        )


_IMAGE_PROMPT = """You are an expert document analyst. Examine this image carefully.

Respond in this exact structure:

## Text content
Transcribe every word of text visible in the image, preserving logical reading order.
Include all headings, body text, captions, labels, and annotations.
If no text is present write: (no text)

## Tables
Reproduce any tables as GitHub-flavoured markdown tables.
If no tables write: (none)

## Image description
One concise paragraph describing what the image depicts (diagram type, chart type,
photograph subject, etc.) and any key data points visible.
"""


def _ollama_call(b64: str, cfg: "ImageConfig", model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": _IMAGE_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = cfg.endpoint.rstrip("/") + "/api/generate"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            return json.loads(resp.read()).get("response", "")
    except urllib.error.HTTPError as e:
        raise ExtractError(f"ollama_http_{e.code}", str(e)) from e
    except TimeoutError as e:
        raise ExtractError("ollama_timeout", str(e)) from e
    except Exception as e:
        raise ExtractError(f"ollama_error:{type(e).__name__}", str(e)) from e


class MultimodalImageExtractor(Extractor):
    """Ollama cloud vision endpoint. Replaces Tesseract entirely.

    Handles: scanned text, diagrams, charts, tables-in-images, handwriting,
    mixed-language content, photographs — anything the vision model can see.

    Retry ladder per model: up to max_retries with exponential backoff.
    Model ladder: primary model -> fallback model -> ExtractError.
    Both file paths (standalone images) and raw bytes (embedded images) accepted.
    """
    name = "ollama_vision"
    version = "1.0"
    suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
    bound = "io"    # network-bound -> asyncio slot, not process pool slot

    def __init__(self, cfg: ImageConfig | None = None):
        self.cfg = cfg or ImageConfig.from_env()

    def _call_with_retry(self, b64: str) -> tuple[str, str]:
        """Returns (response_text, model_used). Primary then fallback."""
        last: ExtractError | None = None
        for model in [self.cfg.model, self.cfg.fallback_model]:
            for attempt in range(self.cfg.max_retries):
                try:
                    return _ollama_call(b64, self.cfg, model), model
                except ExtractError as e:
                    last = e
                    if "http_4" in e.reason_code:   # 4xx: don't retry same model
                        break
                    if attempt < self.cfg.max_retries - 1:
                        time.sleep(0.4 * (2 ** attempt))
        raise last or ExtractError("ollama_all_models_failed")

    def extract(self, path: Path) -> Extracted:
        try:
            raw = path.read_bytes()
        except Exception as e:
            raise ExtractError("image_unreadable", str(e)) from e
        return self._extract_b64(base64.b64encode(raw).decode())

    def extract_bytes(self, raw: bytes) -> Extracted:
        """For embedded images in docx/html — called by ImageEmbedResolver."""
        return self._extract_b64(base64.b64encode(raw).decode())

    def _extract_b64(self, b64: str) -> Extracted:
        response, model_used = self._call_with_retry(b64)
        if not response.strip():
            raise ExtractError("ollama_empty_response")
        warn: list[str] = []
        q, hw = score_text(response, expected_min_chars=40)
        warn.extend(hw)
        no_text = "(no text)" in response
        if no_text and "(none)" in response:
            warn.append("image_no_text_content")
        quality = max(q, 0.5) if not no_text else q
        return Extracted(
            text=response.strip(),
            extractor=f"ollama_vision/{model_used}",
            extractor_version=self.version,
            page_count=1,
            quality=quality,
            warnings=sorted(set(warn)),
        )


# ------------------------------------------------------------------ xlsx


class XlsxExtractor(Extractor):
    """DEFERRED BY DESIGN — not skipped.

    Spreadsheets are record-oriented, not prose. They need their own I1/I3:
    one logical record per row, header row carried into every chunk, sheet name
    and row number as the citation locator. Running them through the prose
    chunker produces confidently-cited nonsense.

    Recorded in the manifest as `quarantined / deferred_tabular` so the corpus
    is honest about what it does not contain.
    """
    name, version = "deferred", "1.0"
    suffixes = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv")

    def extract(self, path: Path) -> Extracted:
        raise ExtractError("deferred_tabular", f"{path.name} routed to the tabular track")


# ------------------------------------------------------------------ registry

DEFAULT_EXTRACTORS: list[Extractor] = []   # populated lazily

def _build_extractors() -> list[Extractor]:
    try:
        from docstore.pdf_extractor import PdfExtractor
        pdf: list[Extractor] = [PdfExtractor()]
    except ImportError:
        pdf = []
    # Imported lazily: code_extractors imports Extracted/Extractor from this
    # module, so a top-level import here would be circular.
    from docstore.code_extractors import CodeExtractor, HtmlModeRouter

    # Order is the contract — select() returns the FIRST extractor that claims
    # the suffix. HtmlModeRouter must precede HtmlExtractor or every .html file
    # goes to trafilatura and source markup is stripped before it is indexed.
    return [MarkdownExtractor(), DocxExtractor(),
            CodeExtractor(),            # .py .js .css .ts .sql .yaml .json ...
            HtmlModeRouter(),           # article vs source, decided per file
            HtmlExtractor(),
            *pdf, MultimodalImageExtractor(), XlsxExtractor()]


def select(path: Path, extractors: list[Extractor] | None = None) -> Extractor:
    pool = extractors or _build_extractors()
    for e in pool:
        if e.supports(path):
            return e
    raise ExtractError(f"unsupported_media_type:{path.suffix.lower() or 'none'}")
