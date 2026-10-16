"""Command line entry point: `nsu-rag`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from nsu_rag_llm.api.deps import get_llm_probe, get_qdrant_probe
from nsu_rag_llm.api.schemas import ComponentHealth
from nsu_rag_llm.config import Settings, get_settings, resolve_device
from nsu_rag_llm.ingest.courses import write as write_courses
from nsu_rag_llm.ingest.inventory import take_inventory
from nsu_rag_llm.ingest.inventory_report import CostEstimate, render, summarise_courses

app = typer.Typer(
    help="RAG question answering over NSU FIT course materials.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def main() -> None:
    """Keep Typer in sub-command mode.

    With a single command Typer would collapse it into the root, and
    `nsu-rag health` would stop working once the other commands land.
    """


async def _probe_all() -> dict[str, ComponentHealth]:
    settings = get_settings()
    qdrant, llm = await asyncio.gather(
        get_qdrant_probe(settings)(),
        get_llm_probe(settings)(),
    )
    return {"qdrant": qdrant, "llm": llm}


@app.command()
def health() -> None:
    """Check that Qdrant and the LLM endpoint answer. Exits non-zero if not."""
    settings = get_settings()
    results = asyncio.run(_probe_all())

    table = Table(title="nsu-rag health")
    table.add_column("component")
    table.add_column("target", overflow="fold")
    table.add_column("status")
    table.add_column("ms", justify="right")
    table.add_column("detail", overflow="fold")

    targets = {"qdrant": settings.qdrant_url, "llm": settings.llm_base_url}
    for name, result in results.items():
        colour = "green" if result.status == "ok" else "red"
        table.add_row(
            name,
            targets[name],
            f"[{colour}]{result.status}[/{colour}]",
            f"{result.latency_ms:.0f}",
            result.detail or "",
        )
    console.print(table)

    if any(r.status != "ok" for r in results.values()):
        raise typer.Exit(code=1)


@app.command()
def inventory(
    root: Annotated[
        Path | None,
        typer.Argument(help="Corpus root. Defaults to RAW_DIR."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where to write the report."),
    ] = None,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help="Also measure OCR and embedding throughput (needs the "
            "ingest and ml extras; takes a few minutes).",
        ),
    ] = False,
) -> None:
    """Describe every file in the corpus: pages, text layer, language, copies."""
    settings = get_settings()
    root = root or settings.raw_dir
    out = out or settings.inventory_file

    if not root.is_dir():
        console.print(f"[red]Нет такого каталога:[/red] {root}")
        raise typer.Exit(code=1)

    total = sum(1 for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Читаю {total} файлов", total=total)
        result = take_inventory(
            root,
            manifest=settings.manifest_file,
            on_file=lambda _: progress.advance(task),
        )

    if benchmark:
        costs = measure_costs(result, root, settings)
        costs.save(settings.inventory_costs_file)
    else:
        costs = CostEstimate.load(settings.inventory_costs_file)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result, costs), encoding="utf-8")
    # Refreshes the measured parts and leaves the author's titles and include
    # flags exactly as they were.
    courses_path = write_courses(result, settings.courses_file)

    primary = result.primary_files
    summary = Table(title="Инвентаризация корпуса")
    summary.add_column("предмет")
    summary.add_column("файлов", justify="right")
    summary.add_column("страниц", justify="right")
    summary.add_column("с текстом", justify="right")
    summary.add_column("доля", justify="right")
    summary.add_column("нужен OCR", justify="right")
    summary.add_column("язык")
    for course in summarise_courses(result):
        share = course.good_share
        colour = "green" if share >= 0.9 else "yellow" if share >= 0.5 else "red"
        summary.add_row(
            course.course,
            str(course.files),
            f"{course.pages}",
            f"{course.good_pages}",
            f"[{colour}]{share:.0%}[/{colour}]",
            str(course.scan_files),
            course.languages,
        )
    console.print(summary)
    console.print(
        f"файлов: {len(result.files)} (уникальных {len(primary)}, "
        f"дублей {len(result.files) - len(primary)})  |  "
        f"страниц: {sum(f.pages or 0 for f in primary)}  |  "
        f"строк манифеста без файла: {len(result.manifest_missing)}  |  "
        f"файлов вне манифеста: {len(result.unlisted)}"
    )
    console.print(f"Отчёт: [bold]{out}[/bold]")
    console.print(f"Каталог курсов: [bold]{courses_path}[/bold]")


def measure_costs(result, root: Path, settings: Settings) -> CostEstimate:
    """Time OCR and embedding on a small real sample, and say so out loud."""
    from nsu_rag_llm.ingest.benchmarks import (
        extract_sample_pages,
        measure_embedding,
        measure_ocr,
    )

    estimate = CostEstimate()
    estimate.ocr_pages = sum(f.pages or 0 for f in result.primary_files if f.needs_ocr)

    console.print("[bold]Замеряю OCR[/bold] на 5 случайных страницах сканов…")
    pages = extract_sample_pages(result, root, count=5)
    estimate.ocr_seconds_per_page, ocr_notes = measure_ocr(pages)
    estimate.notes += ocr_notes

    device = resolve_device(settings.device)
    console.print(f"[bold]Замеряю эмбеддинги[/bold] на устройстве {device}…")
    rate, chunks, embed_notes = measure_embedding(result, root, settings.embedding_model, device)
    estimate.embed_chunks_per_second = rate
    estimate.embed_chunks_total = chunks
    estimate.notes += embed_notes
    return estimate


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


@app.command()
def parse(
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", help="Parallel processes. Defaults to PARSE_WORKERS."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Ignore the cache and parse everything again."),
    ] = False,
    course: Annotated[
        str | None,
        typer.Option("--course", help="Only this course slug."),
    ] = None,
) -> None:
    """Parse the corpus (CORPUS_SCOPE) into cleaned Markdown pages."""
    from nsu_rag_llm.ingest.parse import run as run_parse
    from nsu_rag_llm.ingest.parse_report import render as render_parse_report

    settings = get_settings()
    workers = workers or settings.parse_workers
    if not settings.courses_file.exists():
        console.print("[red]Нет courses.yaml[/red] — сначала `nsu-rag inventory`.")
        raise typer.Exit(code=1)

    # The inventory is re-taken rather than read back from a report: content
    # hashes decide both the cache key and which copy of a duplicate is used,
    # and a stale answer to either would parse the wrong thing.
    root = settings.raw_dir
    total = sum(1 for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    with _progress() as progress:
        task = progress.add_task(f"Инвентаризация: {total} файлов", total=total)
        inventory_result = take_inventory(
            root, settings.manifest_file, on_file=lambda _: progress.advance(task)
        )

    with _progress() as progress:
        task = progress.add_task("Парсинг", total=None)

        def advance(document) -> None:
            progress.update(task, description=f"Парсинг: {document.source.name[:40]}")
            progress.advance(task)

        outcome = run_parse(
            inventory_result,
            root=root,
            courses_file=settings.courses_file,
            parsed_dir=settings.parsed_dir,
            workers=workers,
            force=force,
            only_course=course,
            on_parsed=advance,
            scope=settings.corpus_scope,
            ocr_engine=settings.ocr_engine,
        )

    settings.parse_report_file.parent.mkdir(parents=True, exist_ok=True)
    settings.parse_report_file.write_text(render_parse_report(outcome), encoding="utf-8")

    table = Table(title="Парсинг")
    for name, justify in [
        ("предмет", "left"),
        ("документов", "right"),
        ("страниц", "right"),
        ("символов", "right"),
        ("формул", "right"),
        ("код", "right"),
    ]:
        table.add_column(name, justify=justify)
    by_course: dict[str, list] = {}
    for result in outcome.results:
        by_course.setdefault(result.document.course, []).append(result)
    for slug, results in sorted(by_course.items()):
        pages = [p for r in results for p in r.pages]
        table.add_row(
            slug,
            str(len(results)),
            str(len(pages)),
            str(sum(p["chars"] for p in pages)),
            str(sum(p["formulas"] for p in pages)),
            str(sum(p["code_blocks"] for p in pages)),
        )
    console.print(table)
    fresh = sum(1 for r in outcome.results if not r.from_cache)
    console.print(
        f"разобрано: {fresh}, из кэша: {len(outcome.results) - fresh}, "
        f"пропущено: {len(outcome.skipped)}, с ошибкой: {len(outcome.failed)}  |  "
        f"{outcome.seconds:.0f} с"
    )
    console.print(f"Отчёт: [bold]{settings.parse_report_file}[/bold]")
    if outcome.failed:
        raise typer.Exit(code=1)


@app.command()
def ocr(
    course: Annotated[str | None, typer.Option("--course", help="Only this course slug.")] = None,
    status: Annotated[
        bool, typer.Option("--status", help="Only show how far the OCR has got.")
    ] = False,
) -> None:
    """Recognise the scans of the full corpus - for hours, resumably, in the background."""
    import sys
    import time

    from nsu_rag_llm.ingest.ocr import done_pages, make_converter, ocr_document
    from nsu_rag_llm.ingest.parse import ROUTE_OCR, cache_is_valid, expected_key, select_documents

    settings = get_settings()
    if settings.corpus_scope != "full":
        console.print(
            "[red]OCR нужен только полному корпусу:[/red] "
            "uv run --env-file corpus.full.env nsu-rag ocr"
        )
        raise typer.Exit(code=1)
    root = settings.raw_dir
    inventory_result = take_inventory(root, settings.manifest_file)
    documents, _ = select_documents(inventory_result, settings.courses_file, course, "full")
    scans = [d for d in documents if d.route == ROUTE_OCR]
    key = expected_key(ROUTE_OCR, settings.ocr_engine)
    finished = [d for d in scans if cache_is_valid(settings.parsed_dir / d.sha256, key)]
    todo = [d for d in scans if d not in finished]
    total = sum(d.pages or 0 for d in scans)
    already = sum(d.pages or 0 for d in finished) + sum(
        len(done_pages(settings.parsed_dir / d.sha256)) for d in todo
    )
    console.print(
        f"Сканов: {len(scans)} ({total} стр.), готово документов: {len(finished)}, "
        f"страниц распознано: {already} ({already / max(total, 1):.0%}); "
        f"движок {settings.ocr_engine}"
    )
    if status or not todo:
        return

    import importlib.util

    needed = ["docling"] + (["ocrmac"] if settings.ocr_engine == "ocrmac" else [])
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:
        console.print(f"[red]Не установлено: {', '.join(missing)}[/red] — uv sync --extra ingest")
        raise typer.Exit(code=1)
    unknown = [d for d in todo if not d.pages]
    for document in unknown:
        console.print(f"[yellow]⚠ не знаю числа страниц, пропускаю:[/yellow] {document.source}")
    todo = [d for d in todo if d.pages]
    # Smallest first: finished documents - and their chunks - early, and the
    # speed measured before the hours-long books start.
    todo.sort(key=lambda d: d.pages or 0)
    converter = make_converter(settings.ocr_engine)
    started, pages_done, seconds = time.perf_counter(), already, []

    def report(name: str, number: int, pages: int, took: float) -> None:
        nonlocal pages_done
        pages_done += 1
        seconds.append(took)
        recent = seconds[-50:]
        rate = sum(recent) / len(recent)
        left = (total - pages_done) * rate
        # Plain lines, not a progress bar: this runs for hours, into a log file.
        if pages_done % 10 == 0 or number == pages:
            print(
                f"{time.strftime('%H:%M:%S')} {pages_done}/{total} стр. "
                f"({pages_done / total:.1%}), {rate:.1f} с/стр., осталось ~{left / 3600:.1f} ч "
                f"| {name[:50]} {number}/{pages}",
                flush=True,
            )

    for document in todo:
        name, pages = document.source.name, document.pages or 0
        meta = ocr_document(
            root / document.source,
            settings.parsed_dir / document.sha256,
            pages,
            key,
            converter,
            document.sha256,
            str(document.source),
            on_page=lambda number, took, name=name, pages=pages: report(name, number, pages, took),
        )
        if meta["failed_pages"]:
            print(f"  ⚠ {name}: не распознано страниц: {meta['failed_pages']}", flush=True)
    elapsed = (time.perf_counter() - started) / 3600
    print(f"Готово за {elapsed:.1f} ч. Дальше: nsu-rag parse, nsu-rag ingest.", file=sys.stderr)


@app.command()
def ingest(
    course: Annotated[
        str | None,
        typer.Option("--course", help="Only this course slug; other courses are left alone."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Chunk and report, but do not touch the index."),
    ] = False,
) -> None:
    """Chunk the parsed documents and bring the vector index up to date."""
    import time

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        console.print("[red]Нужны extras ml:[/red] uv sync --extra ml")
        raise typer.Exit(code=1) from None

    from qdrant_client import QdrantClient

    from nsu_rag_llm.ingest.build import build_chunks, course_titles
    from nsu_rag_llm.ingest.chunking import MODEL_MAX_TOKENS, budget_for_model
    from nsu_rag_llm.ingest.indexing import (
        QdrantIndex,
        SentenceTransformerEmbedder,
        apply,
        plan,
        write_chunks,
    )
    from nsu_rag_llm.ingest.ingest_report import IngestRun
    from nsu_rag_llm.ingest.ingest_report import render as render_ingest
    from nsu_rag_llm.ingest.parse import select_documents

    settings = get_settings()
    device = resolve_device(settings.device)

    # Fail before minutes of chunking, not after them.
    index = None
    if not dry_run:
        # Longer than search's 30 s: a batch written while Qdrant builds its HNSW
        # graph for a hundred thousand points can take minutes on a laptop.
        client = QdrantClient(url=settings.qdrant_url, timeout=120)
        try:
            client.get_collections()
        except Exception as exc:  # noqa: BLE001 - any failure means the same to the user
            console.print(
                f"[red]Qdrant недоступен[/red] ({settings.qdrant_url}): {type(exc).__name__}. "
                "Поднимите его: docker compose up -d qdrant"
            )
            raise typer.Exit(code=1) from None
        index = QdrantIndex(client, settings.qdrant_collection)

    root = settings.raw_dir
    total = sum(1 for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    with _progress() as progress:
        task = progress.add_task(f"Инвентаризация: {total} файлов", total=total)
        inventory_result = take_inventory(
            root, settings.manifest_file, on_file=lambda _: progress.advance(task)
        )
    documents, _ = select_documents(
        inventory_result, settings.courses_file, course, settings.corpus_scope
    )

    # The embedding model's own tokenizer: a "token" here is exactly what the
    # model counts. It warns about texts over 512 tokens on every count, which
    # is noise when the point is to find out how long they are.
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    started = time.perf_counter()
    runnable = [d for d in documents if d.route is not None]
    with _progress() as progress:
        task = progress.add_task("Чанкинг", total=len(runnable))
        built, unparsed = build_chunks(
            documents,
            settings.parsed_dir,
            course_titles(settings.courses_file),
            count,
            settings.embedding_model,
            settings.chunk_tokens,
            settings.chunk_overlap_tokens,
            on_document=lambda _: progress.advance(task),
        )
    chunk_seconds = time.perf_counter() - started
    chunks = [c for d in built for c in d.chunks]
    write_chunks(chunks, settings.chunks_file, course=course)
    # What the model will actually read, to prove nothing gets truncated.
    embedding_tokens = [count(f"passage: {c.embedding_text}") + 2 for c in chunks]

    result = None
    if index is not None:
        work = plan(index, chunks, course)
        with _progress() as progress:
            task = progress.add_task(
                f"Эмбеддинги ({device}): {len(work.fresh)} новых", total=len(work.fresh) or 1
            )
            result = apply(
                index,
                lambda: SentenceTransformerEmbedder(
                    settings.embedding_model, device, settings.embed_batch_size
                ),
                work,
                on_batch=lambda n, _: progress.advance(task, n),
            )

    report = IngestRun(
        documents=built,
        unparsed=unparsed,
        model=settings.embedding_model,
        device=device,
        budget=budget_for_model(settings.chunk_tokens),
        overlap=settings.chunk_overlap_tokens,
        chunk_seconds=chunk_seconds,
        sync=result,
        embedding_tokens=embedding_tokens,
        model_limit=MODEL_MAX_TOKENS,
    )
    settings.ingest_report_file.parent.mkdir(parents=True, exist_ok=True)
    settings.ingest_report_file.write_text(render_ingest(report), encoding="utf-8")

    table = Table(title="Чанкинг и индексация")
    for name, justify in [
        ("курс", "left"),
        ("документов", "right"),
        ("страниц", "right"),
        ("чанков", "right"),
        ("токенов, медиана", "right"),
        ("p90", "right"),
    ]:
        table.add_column(name, justify=justify)
    by_course: dict[str, list] = {}
    for d in built:
        by_course.setdefault(d.document.course, []).append(d)
    for slug, docs in sorted(by_course.items()):
        tokens = sorted(c.tokens for d in docs for c in d.chunks) or [0]
        table.add_row(
            slug,
            str(len(docs)),
            str(sum(d.pages for d in docs)),
            str(len(tokens)),
            str(tokens[len(tokens) // 2]),
            str(tokens[int(0.9 * (len(tokens) - 1))]),
        )
    console.print(table)
    over = sum(t > MODEL_MAX_TOKENS for t in embedding_tokens)
    line = (
        f"чанков: {len(chunks)}, длиннее {MODEL_MAX_TOKENS} токенов: {over}"
        f"  |  чанкинг {chunk_seconds:.0f} с"
    )
    if result is not None:
        line += (
            f"  |  добавлено {result.added}, удалено {result.deleted}, "
            f"без изменений {result.unchanged}"
        )
        if result.added:
            line += f", {result.rate:.1f} чанков/с"
    console.print(line)
    console.print(f"Чанки: [bold]{settings.chunks_file}[/bold]")
    console.print(f"Отчёт: [bold]{settings.ingest_report_file}[/bold]")
    if unparsed:
        console.print(f"[yellow]Не разобрано документов: {len(unparsed)}[/yellow] — nsu-rag parse")


@app.command()
def search(
    question: Annotated[str, typer.Argument(help="What to look for.")],
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="bm25 | dense | hybrid | hybrid+rerank"),
    ] = "hybrid+rerank",
    course: Annotated[str | None, typer.Option("--course", help="Only this course slug.")] = None,
    k: Annotated[int, typer.Option("--k", help="How many results.")] = 10,
) -> None:
    """Search the indexed corpus and show where the answer is likely to be."""
    import time

    from nsu_rag_llm.retrieval.modes import MODES, Retrievers

    if mode not in MODES:
        console.print(f"[red]Неизвестный режим[/red] {mode!r}; есть: {', '.join(MODES)}")
        raise typer.Exit(code=1)
    settings = get_settings()
    if not settings.chunks_file.exists():
        console.print("[red]Нет data/chunks.jsonl[/red] — сначала nsu-rag ingest.")
        raise typer.Exit(code=1)

    retrievers = Retrievers(settings)
    started = time.perf_counter()
    retriever = retrievers.get(mode)
    retriever.search("прогрев", 1, course)  # load models before timing the query
    loaded = time.perf_counter() - started

    started = time.perf_counter()
    hits = retriever.search(question, k, course)
    elapsed = time.perf_counter() - started

    table = Table(title=f"{mode}" + (f", курс {course}" if course else ""))
    for name, justify in [
        ("#", "right"),
        ("score", "right"),
        ("курс", "left"),
        ("документ", "left"),
        ("стр.", "right"),
        ("фрагмент", "left"),
    ]:
        table.add_column(name, justify=justify, overflow="fold")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        pages = (
            str(chunk["page_start"])
            if chunk["page_start"] == chunk["page_end"]
            else f"{chunk['page_start']}–{chunk['page_end']}"
        )
        snippet = " ".join(chunk["text"].split())[:140]
        table.add_row(
            str(rank),
            f"{hit.score:.3f}",
            chunk["course"],
            chunk["doc_title"][:40],
            pages,
            snippet,
        )
    console.print(table)
    console.print(f"поиск {elapsed * 1000:.0f} мс (модели и индекс загружены за {loaded:.1f} с)")
    if mode == "hybrid+rerank" and not settings.reranker_enabled:
        console.print("[yellow]RERANKER_ENABLED=false: выполнен режим hybrid[/yellow]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question, in Russian.")],
    course: Annotated[str | None, typer.Option("--course", help="Only this course slug.")] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", "-m", help="bm25 | dense | hybrid | hybrid+rerank [RETRIEVAL_MODE]"),
    ] = None,
    agent: Annotated[
        bool, typer.Option("--agent", help="Answer with the agent: routing, re-querying, tools.")
    ] = False,
) -> None:
    """Answer from the handbooks, with a citation after every claim."""
    import sys
    import time
    from contextlib import aclosing

    import openai

    from nsu_rag_llm.config import configure_logging
    from nsu_rag_llm.generation.answer import Answerer, indexed_courses
    from nsu_rag_llm.retrieval.modes import MODES

    settings = get_settings()
    if mode is not None and mode not in MODES:
        console.print(f"[red]Неизвестный режим[/red] {mode!r}; есть: {', '.join(MODES)}")
        raise typer.Exit(code=1)
    if not settings.chunks_file.exists():
        console.print("[red]Нет data/chunks.jsonl[/red] — сначала nsu-rag ingest.")
        raise typer.Exit(code=1)
    courses = indexed_courses(settings.chunks_file)
    if course is not None and course not in courses:
        known = ", ".join(sorted(courses))
        console.print(f"[red]Курса {course!r} нет в индексе;[/red] есть: {known}")
        raise typer.Exit(code=1)
    # The summary below says what the INFO line would; warnings still show.
    configure_logging("WARNING")

    started = time.perf_counter()
    with console.status("Загружаю модели и индекс…"):
        answerer = Answerer.from_settings(settings, mode)
        answerer.warm_up()
        system = answerer
        if agent:
            from nsu_rag_llm.agent.graph import Agent
            from nsu_rag_llm.agent.llm import OpenAICompatibleTools
            from nsu_rag_llm.ingest.build import course_titles

            system = Agent(
                answerer=answerer,
                model=OpenAICompatibleTools(settings),
                titles=course_titles(settings.courses_file),
                settings=settings,
            )
    loaded = time.perf_counter() - started

    async def run():
        status = console.status("Ищу в методичках…")
        status.start()
        try:
            async with aclosing(system.events(question, course)) as events:
                async for event in events:
                    if event.kind == "sources":
                        documents = len({s.file for s in event.data})
                        status.update(f"Читаю фрагментов: {len(event.data)} из {documents} док.…")
                    elif event.kind == "token":
                        status.stop()
                        # Plain stdout: the answer's brackets are not Rich markup.
                        sys.stdout.write(event.data)
                        sys.stdout.flush()
                    else:
                        return event.data
        finally:
            status.stop()

    try:
        result = asyncio.run(run())
    except openai.APIConnectionError as exc:
        console.print(f"\n[red]LLM недоступна[/red] по {settings.llm_base_url}: {exc}")
        raise typer.Exit(code=1) from exc
    print()
    _print_answer_summary(result, loaded)


def _print_answer_summary(result, loaded: float) -> None:
    from rich.markup import escape

    console.print()
    if result.sources:
        cited = [s for s in result.sources if s.number in result.cited]
        console.print(
            f"[bold]Источники[/bold] (процитировано {len(cited)} из {len(result.sources)}):"
        )
        for source in cited:
            console.print(
                f"  Ф{source.number}  {escape(source.citation)}  [dim]{escape(source.file)}[/dim]",
                highlight=False,
            )
        unused = [f"Ф{s.number}" for s in result.sources if s.number not in result.cited]
        if unused and not result.refused:
            console.print(f"  [dim]не процитированы: {', '.join(unused)}[/dim]")
    for error in result.citation_errors:
        console.print(f"[red]✗ {escape(error)}[/red]")
    if result.dropped:
        console.print(f"[yellow]⚠ не вошли в бюджет контекста: {result.dropped} фрагм.[/yellow]")
    if result.truncated:
        console.print("[yellow]⚠ ответ обрезан: упёрся в LLM_MAX_OUTPUT_TOKENS[/yellow]")

    t = result.timings
    parts = [f"поиск {t.search_ms:.0f} мс"]
    if t.rerank_ms:
        parts.append(f"реранк {t.rerank_ms:.0f} мс")
    if t.first_token_ms is not None:
        parts.append(
            f"первый токен {t.first_token_ms / 1000:.1f} с "
            f"(из них модель читала промпт {t.llm_first_token_ms / 1000:.1f} с)"
        )
    parts.append(f"весь ответ {t.total_ms / 1000:.1f} с")
    console.print(" · ".join(parts), highlight=False)
    if result.prompt_tokens is not None:
        generating = (t.total_ms - (t.first_token_ms or 0)) / 1000
        speed = f", {result.completion_tokens / generating:.1f} ток/с" if generating > 0 else ""
        console.print(
            f"промпт {result.prompt_tokens} ток. (оценка {result.prompt_tokens_estimated}) · "
            f"ответ {result.completion_tokens} ток.{speed}",
            highlight=False,
        )
    for step in result.steps:
        console.print(f"[dim]· {escape(step)}[/dim]", highlight=False)
    console.print(
        f"[dim]режим {result.mode}"
        + (f", курс {result.course}" if result.course else "")
        + f"; модели и индекс загружены за {loaded:.1f} с[/dim]"
    )


@app.command()
def find(
    phrase: Annotated[str, typer.Argument(help="Words from the passage, as the book has them.")],
    course: Annotated[str | None, typer.Option("--course", help="Only this course slug.")] = None,
    limit: Annotated[int, typer.Option("--limit", help="At most this many pages.")] = 20,
) -> None:
    """Find the file and page where a phrase is - to write gold pages by.

    Searches the parsed text, scans' OCR included; an exact phrase, case and
    ё folded. Nothing is ranked: gold pages are yours to judge.
    """
    from rich.markup import escape

    from nsu_rag_llm.evaluation.find import find as find_pages
    from nsu_rag_llm.evaluation.find import indexed_documents

    settings = get_settings()
    full_chunks = settings.chunks_file.parent / "chunks-full.jsonl"
    indexes = {
        "mvp": indexed_documents(settings.chunks_file),
        "full": indexed_documents(full_chunks),
    }
    matches = find_pages(phrase, settings.parsed_dir, settings.raw_dir, indexes, course, limit)
    if not matches:
        console.print("Не нашлось. Попробуйте фразу короче или без формул.")
        raise typer.Exit(code=1)
    table = Table(title=f"«{escape(phrase)}»")
    columns = [("файл (для gold)", "left"), ("стр.", "right"), ("индекс", "left"), ("", "left")]
    for name, justify in columns:
        table.add_column(name, justify=justify, overflow="fold")
    for m in matches:
        where = ", ".join(m.indexes) or "не в индексе"
        origin = " · OCR" if m.origin in ("ocr", "ocr-layer") else ""
        table.add_row(escape(m.file), str(m.page), where + origin, escape(m.context))
    console.print(table)
    if len(matches) == limit:
        console.print(f"[dim]Показаны первые {limit}; --limit или --course сузят поиск.[/dim]")


@app.command("eval")
def evaluate(
    stage: Annotated[
        str, typer.Option("--stage", help="retrieval | generation | agent | judge")
    ] = "retrieval",
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="Questions file. Defaults to QUESTIONS_FILE."),
    ] = None,
    modes: Annotated[
        str,
        typer.Option("--modes", help="retrieval: comma-separated retrieval modes."),
    ] = "bm25,dense,hybrid,hybrid+rerank",
    validate_only: Annotated[
        bool, typer.Option("--validate-only", help="Check the questions file and stop.")
    ] = False,
    course_filter: Annotated[
        bool,
        typer.Option(
            "--course-filter/--no-course-filter",
            help="retrieval: also score every mode with the question's course as a filter.",
        ),
    ] = True,
    variants: Annotated[
        str,
        typer.Option("--variants", help="generation: labels from eval/models.yaml, or all."),
    ] = "all",
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="generation: retrieval mode [RETRIEVAL_MODE]."),
    ] = None,
    judge: Annotated[
        bool,
        typer.Option("--judge/--no-judge", help="generation, agent: judge the answers."),
    ] = True,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", help="judge: the author's labels [MANUAL_LABELS_FILE]."),
    ] = None,
) -> None:
    """Measure retrieval, answers, or the judge against the author's own data."""
    settings = get_settings()
    if stage == "judge":
        _eval_judge(settings, labels or settings.manual_labels_file)
        return
    if stage not in ("retrieval", "generation", "agent"):
        console.print(
            f"[red]Неизвестная стадия {stage!r}:[/red] retrieval, generation, agent, judge."
        )
        raise typer.Exit(code=1)
    if not settings.chunks_file.exists():
        console.print("[red]Нет data/chunks.jsonl[/red] — сначала nsu-rag ingest.")
        raise typer.Exit(code=1)
    path = questions or settings.questions_file
    if validate_only or stage in ("generation", "agent"):
        from nsu_rag_llm.evaluation.questions import validate
        from nsu_rag_llm.evaluation.runner import index_facts

        # Before any model loads: a missing file should cost a second, not twenty.
        indexed, _, _ = index_facts(settings.chunks_file)
        validation = validate(path, settings.raw_dir, indexed)
        if validate_only or not validation.ok:
            _report_validation(validation, path)
            return
    if stage == "retrieval":
        _eval_retrieval(settings, path, modes, course_filter)
    elif stage == "agent":
        _eval_agent(settings, path, judge)
    else:
        _eval_generation(settings, path, variants, mode or settings.retrieval_mode, judge)


def _report_validation(result, path: Path) -> None:
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    for error in result.errors:
        console.print(f"[red]✗ {error}[/red]")
    answerable = sum(q.answerable for q in result.questions)
    console.print(
        f"вопросов: {len(result.questions)} (с ответом {answerable}), "
        f"ошибок: {len(result.errors)}, предупреждений: {len(result.warnings)}"
    )
    if not result.ok:
        if not path.exists():
            console.print(
                "Вопросы пишет автор: формат — eval/README.md, пример — "
                "eval/questions.example.jsonl."
            )
        raise typer.Exit(code=1)


def _eval_retrieval(settings: Settings, path: Path, modes: str, course_filter: bool) -> None:
    from nsu_rag_llm.evaluation.runner import run_retrieval
    from nsu_rag_llm.retrieval.modes import MODES, Retrievers

    chosen = [m.strip() for m in modes.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in MODES]
    if unknown:
        console.print(f"[red]Неизвестные режимы:[/red] {', '.join(unknown)}")
        raise typer.Exit(code=1)
    retrievers = Retrievers(settings)
    run = run_retrieval(
        settings,
        chosen,
        retrievers.get,
        questions_file=path,
        with_filter=course_filter,
        on_mode=lambda mode, filtered: console.print(
            f"  {mode}" + (" (фильтр по курсу)" if filtered else "")
        ),
    )
    _report_validation(run.validation, path)

    table = Table(title="Оценка поиска")
    for name in ("режим", "recall@5", "recall@10", "±95%", "MRR@10", "мс/запрос"):
        table.add_column(name, justify="left" if name == "режим" else "right")
    for r in run.unfiltered:
        table.add_row(
            r.mode,
            f"{r.recall(5):.0%}",
            f"{r.recall(10):.0%}",
            f"{r.interval(10):.0%}",
            f"{r.mrr():.3f}",
            f"{r.median_ms():.0f}",
        )
    console.print(table)
    console.print(f"Отчёт: [bold]{run.report}[/bold]")


def _eval_generation(
    settings: Settings, path: Path, variants: str, mode: str, with_judge: bool
) -> None:
    import gc

    from nsu_rag_llm.evaluation.generation import load_variants
    from nsu_rag_llm.evaluation.generation_runner import run_generation
    from nsu_rag_llm.evaluation.judge import Judge, OpenAICompatibleJson
    from nsu_rag_llm.evaluation.ollama import Ollama
    from nsu_rag_llm.generation.answer import Answerer, indexed_courses
    from nsu_rag_llm.generation.llm import OpenAICompatibleChat, llm_token_counter
    from nsu_rag_llm.retrieval.modes import MODES

    if mode not in MODES:
        console.print(f"[red]Неизвестный режим[/red] {mode!r}; есть: {', '.join(MODES)}")
        raise typer.Exit(code=1)
    available = load_variants(settings.models_file)
    wanted = [v.strip() for v in variants.split(",") if v.strip()]
    by_label = {v.label: v for v in available}
    # In the order given: the order is part of an experiment (a laptop warms up).
    chosen = available if wanted == ["all"] else [by_label[w] for w in wanted if w in by_label]
    missing = sorted(set(wanted) - {"all"} - {v.label for v in available})
    if missing or not chosen:
        known = ", ".join(v.label for v in available)
        console.print(f"[red]Нет в {settings.models_file.name}:[/red] {missing}; есть: {known}")
        raise typer.Exit(code=1)

    # One pipeline to search with; released before any answering model loads.
    searcher: Answerer | None = Answerer.from_settings(settings, mode)

    def retrieve(question):
        assert searcher is not None
        return searcher.retrieve(question.question)

    def release() -> None:
        nonlocal searcher
        searcher = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    courses = indexed_courses(settings.chunks_file)
    counters: dict = {}

    def answerer_for(variant) -> Answerer:
        tuned = settings.model_copy(
            update={
                "llm_model": variant.model,
                "llm_reasoning_effort": variant.reasoning_effort,
                "llm_max_output_tokens": variant.max_output_tokens,
            }
        )
        if variant.tokenizer not in counters:
            counters[variant.tokenizer] = llm_token_counter(variant.tokenizer)
        return Answerer(
            search=lambda query, k, course: [],
            rerank=None,
            chat=OpenAICompatibleChat(tuned),
            count=counters[variant.tokenizer],
            settings=tuned,
            mode=mode,
            courses=courses,
        )

    judge_model = None
    if with_judge:
        judge_model = Judge(OpenAICompatibleJson(settings), settings.judge_model)
    tasks: dict[str, int] = {}
    with _progress() as progress:

        def on_progress(label: str, done: int, total: int) -> None:
            if label not in tasks:
                tasks[label] = progress.add_task(label, total=total)
            progress.update(tasks[label], completed=done)

        run = asyncio.run(
            run_generation(
                settings,
                chosen,
                mode,
                retrieve,
                answerer_for,
                judge_model,
                Ollama(settings.ollama_api_url),
                questions_file=path,
                release_retrieval=release,
                on_progress=on_progress,
            )
        )
    _report_validation(run.validation, path)

    table = Table(title=f"Оценка ответов ({mode})")
    for name in ("модель", "верность", "полнота", "ссылки", "отказ", "TTFT", "ток/с", "память"):
        table.add_column(name, justify="left" if name == "модель" else "right")
    for r in run.results:
        peak = r.peak_memory_bytes
        table.add_row(
            r.variant.label,
            _pct(r.faithfulness()),
            _pct(r.completeness()),
            _pct(r.citations_valid()),
            _pct(r.refusal_accuracy()),
            f"{r.ttft_ms() / 1000:.1f} с",
            f"{r.tokens_per_second():.1f}",
            "—" if peak is None else f"{peak / 1e9:.1f} ГБ",
        )
    console.print(table)
    console.print(f"Ответы и вердикты: [bold]{settings.eval_runs_dir}[/bold]")
    console.print(f"Отчёт: [bold]{run.report}[/bold]")


def _pct(value: float) -> str:
    return "—" if value != value else f"{value:.0%}"


def _eval_agent(settings: Settings, path: Path, with_judge: bool) -> None:
    """Phase 5: the phase 3 pipeline against the agent, end to end."""
    from nsu_rag_llm.agent.graph import Agent
    from nsu_rag_llm.agent.llm import OpenAICompatibleTools
    from nsu_rag_llm.evaluation.generation_runner import run_systems
    from nsu_rag_llm.evaluation.judge import Judge, OpenAICompatibleJson
    from nsu_rag_llm.generation.answer import Answerer
    from nsu_rag_llm.ingest.build import course_titles

    answerer = Answerer.from_settings(settings)
    answerer.warm_up()
    agent = Agent(
        answerer=answerer,
        model=OpenAICompatibleTools(settings),
        titles=course_titles(settings.courses_file),
        settings=settings,
    )

    async def pipeline(question):
        return await answerer.ask(question.question)

    async def agentic(question):
        return await agent.ask(question.question)

    judge_model = None
    if with_judge:
        judge_model = Judge(OpenAICompatibleJson(settings), settings.judge_model)
    tasks: dict[str, int] = {}
    with _progress() as progress:

        def on_progress(label: str, done: int, total: int) -> None:
            if label not in tasks:
                tasks[label] = progress.add_task(label, total=total)
            progress.update(tasks[label], completed=done)

        run = asyncio.run(
            run_systems(
                settings,
                {"pipeline": pipeline, "agent": agentic},
                judge_model,
                questions_file=path,
                on_progress=on_progress,
            )
        )
    _report_validation(run.validation, path)
    table = Table(title="Конвейер фазы 3 против агента")
    for name in ("система", "верность", "полнота", "ссылки", "отказ", "маршрут", "поисков"):
        table.add_column(name, justify="left" if name == "система" else "right")
    for r in run.results:
        table.add_row(
            r.variant.label,
            _pct(r.faithfulness()),
            _pct(r.completeness()),
            _pct(r.citations_valid()),
            _pct(r.refusal_accuracy()),
            _pct(r.router_accuracy()),
            f"{r.mean_searches():.1f}",
        )
    console.print(table)
    console.print(f"Отчёт: [bold]{run.report}[/bold]")


def _eval_judge(settings: Settings, labels: Path) -> None:
    from datetime import date

    from nsu_rag_llm.evaluation.calibration import calibrate, render

    calibration = calibrate(labels, settings.eval_runs_dir)
    for warning in calibration.labels.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    for error in calibration.labels.errors:
        console.print(f"[red]✗ {error}[/red]")
    if not calibration.labels.ok:
        if not labels.exists():
            console.print(
                "Разметку пишет автор: формат — eval/README.md, пример — "
                "eval/manual_labels.example.jsonl."
            )
        raise typer.Exit(code=1)
    table = Table(title="Судья против разметки автора")
    for name in ("вердикт", "n", "совпадение", "каппа"):
        table.add_column(name, justify="left" if name == "вердикт" else "right")
    for name, title in (("faithful", "верность"), ("complete", "полнота")):
        a = getattr(calibration, name)
        table.add_row(
            title,
            str(a.n),
            _pct(a.observed),
            "—" if a.kappa != a.kappa else f"{a.kappa:.2f}",
        )
    console.print(table)
    for item in calibration.unmatched:
        console.print(f"[yellow]⚠ нет вердикта судьи: {item}[/yellow]")
    settings.eval_reports_dir.mkdir(parents=True, exist_ok=True)
    report = settings.eval_reports_dir / f"{date.today().isoformat()}-judge.md"
    report.write_text(render(calibration), encoding="utf-8")
    console.print(f"Отчёт: [bold]{report}[/bold]")
