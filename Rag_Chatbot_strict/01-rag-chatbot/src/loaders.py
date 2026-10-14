"""File loading: turn a PDF/TXT/CSV path into LangChain Documents.

Scanned PDF pages (no text layer) are OCR'd. Three things here matter for
getting answers that match the PDF exactly:

1. OCR engine: the `rapidocr` package (v3) ships multilingual PP-OCRv6 models
   that keep the spaces between English words. The older
   `rapidocr_onnxruntime` package uses a Chinese model that glues words
   together ("DeanFacultyofCivilandPetroleumEngineering"), which breaks both
   retrieval and the bot's answers. The old package is only used as a fallback.
2. Reading order: OCR boxes are re-ordered column by column (see
   ocr_layout.py) so two-column pages are not interleaved.
3. Repeated page furniture ("NED UNIVERSITY", "PROSPECTUS 2025",
   "www.neduet.edu.pk", ...) is removed so it doesn't pollute every chunk.

Raw OCR boxes are cached per file, so the layout step can be improved and
re-run without OCR'ing the whole PDF again.
"""

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .ocr_layout import order_boxes, to_boxes

CACHE_DIR = Path("ocr_cache")
CACHE_VERSION = "boxes-v2"  # older text-only caches are ignored (wrong order, glued words)
OCR_DPI = 200

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MIN_TEXT_CHARS = 20  # a page with fewer characters is treated as "no text layer"
REPEATED_LINE_SHARE = 0.4  # a line on >= 40% of pages is a header/footer

_ocr_engine = None
_ocr_engine_name = None


def _get_ocr_engine():
    global _ocr_engine, _ocr_engine_name
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR  # v3: bundled PP-OCRv6 models, keeps spaces

            _ocr_engine = RapidOCR(params={"Global.log_level": "error"})
            _ocr_engine_name = "rapidocr-v3"
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR

            print(
                "WARNING: using rapidocr_onnxruntime; English words may be glued "
                "together. Run: pip install rapidocr opencv-python-headless",
                flush=True,
            )
            _ocr_engine = RapidOCR()
            _ocr_engine_name = "rapidocr-onnxruntime"
    return _ocr_engine


def _ocr_page_boxes(page: "fitz.Page") -> dict:
    """OCR one page. Returns {"w": width_px, "items": [[quad, text], ...]}."""
    engine = _get_ocr_engine()
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if _ocr_engine_name == "rapidocr-v3":
        result = engine(img)
        quads = [] if result.boxes is None else result.boxes.tolist()
        texts = list(result.txts or [])
    else:
        result, _ = engine(img)
        quads = [r[0] for r in result or []]
        texts = [r[1] for r in result or []]
    return {"w": pix.width, "items": [[q, t] for q, t in zip(quads, texts)]}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _norm_line(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.lower())


def _strip_repeated_lines(docs: list[Document]) -> None:
    """Remove headers/footers: lines (ignoring case, spaces, digits) that
    appear on a large share of the pages."""
    if len(docs) < 4:
        return
    counts = Counter()
    for d in docs:
        counts.update({_norm_line(l) for l in d.page_content.splitlines() if _norm_line(l)})
    repeated = {
        k for k, n in counts.items()
        if n >= REPEATED_LINE_SHARE * len(docs) and len(k) >= 6
    }
    for d in docs:
        kept = [l for l in d.page_content.splitlines() if _norm_line(l) not in repeated]
        d.page_content = "\n".join(kept).strip()


def _load_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    empty = [d for d in docs if len(d.page_content.strip()) < MIN_TEXT_CHARS]

    if empty:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{_file_hash(path)}.{CACHE_VERSION}.json"
        cache = (
            json.loads(cache_file.read_text(encoding="utf-8"))
            if cache_file.exists()
            else {}
        )
        print(f"{path.name}: {len(docs)} pages, {len(empty)} need OCR", flush=True)
        start = time.time()
        with fitz.open(str(path)) as pdf:
            for n, d in enumerate(empty, 1):
                key = str(d.metadata.get("page", 0))
                if key not in cache:
                    cache[key] = _ocr_page_boxes(pdf[int(key)])
                    cache_file.write_text(json.dumps(cache), encoding="utf-8")
                    print(f"OCR {n}/{len(empty)} ({time.time() - start:.0f}s)", flush=True)
                page = cache[key]
                lines = order_boxes(to_boxes(page["items"]), page["w"])
                d.page_content = "\n".join(lines)
                d.metadata["ocr"] = True

    _strip_repeated_lines(docs)
    for d in docs:
        # 1-based page number as shown in a PDF viewer
        d.metadata["page_number"] = int(d.metadata.get("page", 0)) + 1
    return [d for d in docs if d.page_content.strip()]


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = _load_pdf(path)
    elif ext == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
    elif ext == ".csv":
        docs = CSVLoader(str(path)).load()
    else:
        raise ValueError(
            f"Unsupported file type: {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not docs:
        raise ValueError(f"No text could be extracted from {path.name}.")

    for doc in docs:
        doc.metadata["source"] = path.name
    return docs


def load_files(paths: list[str | Path]) -> list[Document]:
    docs: list[Document] = []
    for path in paths:
        docs.extend(load_file(path))
    return docs
