"""Measure what the expensive steps cost, so the plan rests on numbers.

Two questions decide how the corpus is approached: how long OCR would take on
the scanned half, and how long embedding would take on all of it. Both are
measured on a small real sample and extrapolated, rather than guessed.

The heavy libraries are imported inside the functions: the inventory itself
must keep working on an install without the `ingest` and `ml` extras.
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time
from pathlib import Path

import pymupdf

from nsu_rag_llm.ingest.inventory import Inventory
from nsu_rag_llm.ingest.inventory_report import thousands

#: Language for the OCR engine. RapidOCR, the engine docling installs by
#: default, rejects a bare "ru" and wants the ISO-prefixed form; its Russian
#: model exists only in the onnxruntime backend, not the torch one. It also
#: takes exactly one language per pass and silently drops the rest, so asking
#: for Russian and English would measure a Russian-only run under a misleading
#: label. English-only scans need their own pass - a phase 4.5 problem.
OCR_LANGUAGES = ["iso:ru"]

#: Roughly the chunk size phase 1c will produce (~400 tokens with ~50 overlap).
CHUNK_TOKENS = 400


def extract_sample_pages(
    inventory: Inventory,
    root: Path,
    count: int,
    seed: int = 0,
) -> list[Path]:
    """Cut `count` random pages out of scanned documents into one-page PDFs.

    Pages are drawn from different documents where possible, because scan
    quality varies far more between books than within one.
    """
    scans = [
        f
        for f in inventory.primary_files
        if f.text_layer in {"none", "weak"} and (f.pages or 0) > 4
    ]
    if not scans:
        return []

    rng = random.Random(seed)
    chosen = [scans[i % len(scans)] for i in range(count)]
    workdir = Path(tempfile.mkdtemp(prefix="nsu-rag-ocr-"))

    out: list[Path] = []
    for index, report in enumerate(chosen):
        source = root / report.path
        page_no = rng.randrange(1, report.pages or 1)
        target = workdir / f"page-{index}.pdf"
        with pymupdf.open(source) as doc:
            single = pymupdf.open()
            single.insert_pdf(doc, from_page=page_no, to_page=page_no)
            single.save(target)
            single.close()
        out.append(target)
    return out


def measure_ocr(pages: list[Path]) -> tuple[float | None, list[str]]:
    """Return the median seconds per page, and notes about the measurement.

    The first page is a warm-up: docling downloads and loads its models on the
    first conversion, which has nothing to do with the per-page cost.
    """
    if len(pages) < 2:
        return None, ["Недостаточно страниц-сканов для замера."]

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    # Table structure recognition is a separate model and a separate bill;
    # phase 1b can turn it on once OCR itself is affordable.
    options.do_table_structure = False
    options.ocr_options.lang = OCR_LANGUAGES

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    started = time.perf_counter()
    warmup = converter.convert(pages[0])
    cold = time.perf_counter() - started

    timings: list[float] = []
    characters: list[int] = []
    for page in pages[1:]:
        started = time.perf_counter()
        result = converter.convert(page)
        timings.append(time.perf_counter() - started)
        characters.append(len(result.document.export_to_markdown()))

    empty = sum(1 for c in characters if c == 0)
    notes = [
        f"OCR: движок RapidOCR (onnxruntime), язык {', '.join(OCR_LANGUAGES)} "
        f"(движок берёт ровно один язык за проход); распознавание таблиц выключено.",
        f"Первый вызов стоил {cold:.0f} с — это загрузка моделей, в расчёт не берётся.",
        f"Замер по {len(timings)} страницам: "
        f"медиана {statistics.median(timings):.1f} с, "
        f"разброс {min(timings):.1f}–{max(timings):.1f} с.",
        f"Со страницы получается {statistics.median(characters):.0f} символов "
        f"(прогревочная дала {len(warmup.document.export_to_markdown())}).",
    ]
    if empty:
        notes.append(
            f"На {empty} из {len(timings)} страниц движок не нашёл текста — "
            f"скорее всего попалась вклейка или пустая страница. "
            f"Время они всё равно стоят, поэтому из замера не выброшены."
        )
    return statistics.median(timings), notes


def collect_chunk_texts(
    inventory: Inventory,
    root: Path,
    wanted: int,
    chars_per_chunk: int,
    seed: int = 0,
) -> list[str]:
    """Take real text off pages that already have a text layer."""
    rng = random.Random(seed)
    readable = [f for f in inventory.primary_files if f.text_layer == "good" and (f.pages or 0) > 4]
    rng.shuffle(readable)

    chunks: list[str] = []
    for report in readable:
        if len(chunks) >= wanted:
            break
        with pymupdf.open(root / report.path) as doc:
            text = "".join(doc.load_page(i).get_text() for i in range(min(doc.page_count, 40)))
        for start in range(0, len(text), chars_per_chunk):
            piece = text[start : start + chars_per_chunk].strip()
            if len(piece) > chars_per_chunk // 2:
                chunks.append(piece)
            if len(chunks) >= wanted:
                break
    return chunks


def measure_embedding(
    inventory: Inventory,
    root: Path,
    model_name: str,
    device: str,
    sample_chunks: int = 256,
    batch_size: int = 32,
) -> tuple[float | None, int, list[str]]:
    """Return chunks per second, the estimated corpus size in chunks, and notes."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    tokenizer = model.tokenizer

    # Convert the target chunk size from tokens to characters using this
    # corpus and this tokenizer, rather than a rule of thumb: Russian text
    # costs noticeably more tokens per character than English.
    probe = collect_chunk_texts(inventory, root, wanted=20, chars_per_chunk=2000)
    if not probe:
        return None, 0, ["Нет файлов с текстовым слоем — замер невозможен."]
    chars_per_token = statistics.median(
        len(t) / max(len(tokenizer.encode(t, add_special_tokens=False)), 1) for t in probe
    )
    chars_per_chunk = int(CHUNK_TOKENS * chars_per_token)

    chunks = collect_chunk_texts(inventory, root, sample_chunks, chars_per_chunk)
    if len(chunks) < batch_size:
        return None, 0, ["Слишком мало текста для замера эмбеддингов."]

    # e5 expects the passage prefix; measuring without it would measure a
    # different input length than the real pipeline.
    payload = [f"passage: {c}" for c in chunks]
    model.encode(payload[:batch_size], batch_size=batch_size, show_progress_bar=False)

    started = time.perf_counter()
    model.encode(payload, batch_size=batch_size, show_progress_bar=False)
    elapsed = time.perf_counter() - started
    rate = len(payload) / elapsed

    total_chars = estimate_corpus_characters(inventory)
    total_chunks = int(total_chars / chars_per_chunk)

    notes = [
        f"Эмбеддинги: {model_name} на устройстве `{device}`, батч {batch_size}.",
        f"Токенизатор этого корпуса даёт {chars_per_token:.1f} символов на токен, "
        f"то есть фрагмент в {CHUNK_TOKENS} токенов — это ≈{chars_per_chunk} символов.",
        f"Замер по {len(payload)} фрагментам за {elapsed:.1f} с. "
        f"Это нижняя граница: батч не подбирался, а у модели предел "
        f"{model.max_seq_length} токенов, так что фрагмент в "
        f"{CHUNK_TOKENS} токенов в него укладывается.",
        f"Оценка объёма корпуса: ≈{total_chars / 1e6:.1f} млн символов "
        f"→ ≈{thousands(total_chunks)} фрагментов (страницы сканов посчитаны "
        f"по медианной плотности читаемых страниц).",
    ]
    return rate, total_chunks, notes


def estimate_corpus_characters(inventory: Inventory) -> float:
    """Characters the corpus would hold once every page is readable.

    Pages that currently have no text layer are counted at the median density
    of the pages that do: after OCR they will carry text too, and leaving them
    at zero would understate the indexing bill.
    """
    readable = [f for f in inventory.primary_files if f.text_layer == "good"]
    density = statistics.median([f.chars_per_page for f in readable]) if readable else 0.0

    total = 0.0
    for report in inventory.primary_files:
        pages = report.pages or 0
        if report.text_layer == "good":
            total += report.chars_per_page * pages
        else:
            total += density * pages
    return total
