"""PDF extractor — three-strategy pipeline per page.

Strategy 1 (text layer): pymupdf extracts text from the PDF's text layer.
    Fast, zero model calls, zero cost. Works for all digitally-created PDFs.
    Quality-scored; pages below threshold escalate to Strategy 2.

Strategy 2 (multimodal): page rendered to PNG → MultimodalImageExtractor.
    For scanned PDFs, image-heavy pages, forms with no text layer.
    Each page is an independent vision call — expensive but accurate.
    Blobs written to ImageBlobStore for I1b if calling from prose context.

Strategy 3 (skip): if both strategies fail for a page, page is recorded as
    failed in warnings; extraction continues on remaining pages.
    A document where >50% of pages fail → quarantined overall.

Embedded images:
    PDF pages can contain embedded raster images (figures, charts, photos).
    These are extracted as blobs and written to ImageBlobStore under
    {doc_id}:{page_idx}_{img_idx} for I1b phase-2 resolution.
    The text output contains <!-- image:pending:N:pdf_figure --> markers.

Dependencies:
    pymupdf (pip install pymupdf)     — always required
    pillow (pip install Pillow)       — for page rendering
    MultimodalImageExtractor          — from this package
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from docstore.extractors import (
    Extracted,
    ExtractError,
    Extractor,
    ImageConfig,
    MultimodalImageExtractor,
    score_text,
)

if TYPE_CHECKING:
    pass


@dataclass
class PdfExtractionConfig:
    """Tuning knobs — all overridable from env/config."""
    text_quality_floor: float = 0.45    # below this → try multimodal for the page
    page_fail_threshold: float = 0.50   # >50% pages failed → quarantine whole doc
    render_dpi: int = 150               # page render resolution for vision calls
    max_pages_multimodal: int = 50      # cap: very long scanned docs need batching
    extract_embedded_images: bool = True

    @classmethod
    def from_env(cls) -> "PdfExtractionConfig":
        import os
        return cls(
            text_quality_floor=float(
                os.environ.get("PDF_TEXT_QUALITY_FLOOR", "0.45")),
            render_dpi=int(os.environ.get("PDF_RENDER_DPI", "150")),
            max_pages_multimodal=int(
                os.environ.get("PDF_MAX_PAGES_MULTIMODAL", "50")),
        )


class PdfExtractor(Extractor):
    """Hybrid PDF extractor.

    Text-layer pages  → fast pymupdf extraction
    Scanned pages     → multimodal vision (same Ollama cloud endpoint)
    Embedded images   → blob store stubs for I1b resolution
    """
    name = "pymupdf_hybrid"
    version = "1.0"
    suffixes = (".pdf",)
    bound = "cpu"   # pymupdf is CPU; vision pages get IO classification at I1b

    def __init__(
        self,
        cfg: PdfExtractionConfig | None = None,
        vision_cfg: ImageConfig | None = None,
        blob_store=None,
        doc_id: str = "",
    ):
        self.cfg = cfg or PdfExtractionConfig.from_env()
        self.vision_cfg = vision_cfg or ImageConfig.from_env()
        self.blob_store = blob_store
        self.doc_id = doc_id
        self._vision = MultimodalImageExtractor(self.vision_cfg)

    def extract(self, path: Path) -> Extracted:
        try:
            import pymupdf as fitz  # type: ignore  (pymupdf >= 1.24 canonical import)
        except ImportError as e:
            raise ExtractError("extractor_unavailable:pymupdf") from e

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise ExtractError("pdf_unreadable", str(e)) from e

        page_count = len(doc)
        parts: list[str] = []
        warn: list[str] = []
        failed_pages = 0
        img_idx = 0
        vision_pages = 0

        for page_num, page in enumerate(doc):
            # ---- Strategy 1: text layer ----
            text = page.get_text("text").strip()        # "markdown" not supported in all pymupdf versions
            q, _ = score_text(text, expected_min_chars=80)

            if q >= self.cfg.text_quality_floor and text:
                parts.append(f"<!-- page:{page_num + 1} -->\n{text}")

                # Extract embedded images from text-layer pages too
                if self.cfg.extract_embedded_images:
                    img_idx = self._extract_embedded_images(
                        page, page_num, img_idx, parts, warn
                    )
                continue

            # ---- Strategy 2: multimodal vision ----
            if vision_pages >= self.cfg.max_pages_multimodal:
                warn.append(f"page_{page_num + 1}:vision_cap_reached")
                failed_pages += 1
                continue

            try:
                png_bytes = self._render_page(page, doc)
                vision_result = self._vision.extract_bytes(png_bytes)
                parts.append(
                    f"<!-- page:{page_num + 1}:scanned -->\n{vision_result.text}"
                )
                vision_pages += 1
                if vision_result.warnings:
                    warn.extend(
                        f"page_{page_num + 1}:{w}" for w in vision_result.warnings
                    )
            except ExtractError as e:
                warn.append(f"page_{page_num + 1}:failed:{e.reason_code}")
                failed_pages += 1

        doc.close()

        if vision_pages > 0:
            warn.append(f"scanned_pages={vision_pages}")

        fail_rate = failed_pages / max(page_count, 1)
        if fail_rate > self.cfg.page_fail_threshold:
            raise ExtractError(
                "pdf_high_page_failure_rate",
                f"{failed_pages}/{page_count} pages failed",
            )

        full_text = "\n\n".join(parts)
        q, hw = score_text(full_text, expected_min_chars=200)
        warn.extend(hw)

        return Extracted(
            text=full_text,
            extractor=self.name,
            extractor_version=self.version,
            page_count=page_count,
            quality=q,
            warnings=sorted(set(warn)),
        )

    def _render_page(self, page, doc) -> bytes:
        """Render a page to PNG bytes for vision model."""
        try:
            from PIL import Image as _PilImage  # type: ignore
        except ImportError:
            pass  # fall through to raw fitz render without Pillow

        mat = page.get_pixmap(dpi=self.cfg.render_dpi)
        return mat.tobytes("png")

    def _extract_embedded_images(
        self,
        page,
        page_num: int,
        img_idx: int,
        parts: list[str],
        warn: list[str],
    ) -> int:
        """Extract raster images embedded in PDF page; write blobs for I1b."""
        try:
            import pymupdf as fitz  # type: ignore
            doc = page.parent
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    stub = (
                        f"<!-- image:pending:{img_idx}:"
                        f"pdf_figure_p{page_num + 1} -->"
                    )
                    parts.append(stub)
                    if self.blob_store and self.doc_id:
                        self.blob_store.put(self.doc_id, img_idx, img_bytes)
                    img_idx += 1
                    warn.append("contains_embedded_image")
                except Exception:
                    pass   # single image failure never aborts the page
        except Exception:
            pass
        return img_idx
