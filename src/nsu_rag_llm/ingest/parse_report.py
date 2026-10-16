"""Parse-quality report: what came out, and which pages a person should open.

Like the inventory report it lists file names and numbers, never the text of a
document, and it lives in `data/`, outside git.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date

from nsu_rag_llm.ingest.cleaning import CLEANING_REVISION
from nsu_rag_llm.ingest.inventory_report import thousands
from nsu_rag_llm.ingest.parse import DocumentResult, ParseRun
from nsu_rag_llm.ingest.parsers.pdf import parser_key

#: Fewer characters of prose than this and a page is "almost empty".
LOW_TEXT_CHARS = 150
#: The page's own text layer is compared with what came out of the parser. A
#: normal page keeps nearly all of it; below this share, text went missing.
KEPT_SHARE = 0.3
#: ...but only on pages with enough text for the comparison to mean anything.
LOST_MIN_SOURCE = 500
#: Broken glyphs are worth a person's time only when there are enough of them:
#: four U+FFFD on a 180-character page is 2%, and nothing anyone can fix.
HIGH_JUNK = 0.02
JUNK_MIN_CHARS = 20
#: Covers, title pages and blank endpapers are expected to be thin or broken.
EDGE_PAGES = 2


def classify(page: dict, page_count: int) -> tuple[str, str] | None:
    """(kind, reason) for a page worth a mention, or None.

    kind is `review` when nothing explains the page and a person should look,
    or `expected` when the page is thin for a known reason. The main test
    compares the page with its own text layer in the PDF: a page that is empty
    in the PDF is not the parser's fault, and a page the PDF has plenty of text
    for but the parser does not is exactly what someone should open.
    """
    at_edge = page["number"] <= EDGE_PAGES or page["number"] > page_count - EDGE_PAGES
    explained = page["pictures"] or page["formulas"]
    source = page.get("source_chars", 0)
    kept = page.get("text_chars", page["chars"])

    if page.get("ocr_error"):
        return "review", "OCR упал на странице"
    # A scan has no text layer to compare with: an empty page after OCR is
    # the engine finding nothing, not a page empty in the PDF.
    if page.get("origin") == "ocr" and page["chars"] < LOW_TEXT_CHARS:
        if at_edge:
            return "expected", "обложка или концевая"
        return "review", "OCR не нашёл текста"

    if page["junk_ratio"] > HIGH_JUNK and page.get("junk_chars", 0) >= JUNK_MIN_CHARS:
        if at_edge:
            return "expected", "обложка: шрифт без Unicode"
        return "review", f"мусор {page['junk_ratio']:.0%}"
    if source >= LOST_MIN_SOURCE and kept < KEPT_SHARE * source and not explained:
        return "review", "разбор потерял текст"
    if page["chars"] < LOW_TEXT_CHARS:
        if source < LOW_TEXT_CHARS:
            return "expected", "пустая и в самом PDF"
        if page["pictures"]:
            return "expected", "рисунок"
        if page["formulas"]:
            return "expected", "формулы"
        if at_edge:
            return "expected", "обложка или концевая"
        return "review", "почти пусто"
    return None


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _ranges(numbers: list[int]) -> str:
    """[3, 4, 5, 9] -> "3–5, 9"."""
    out: list[str] = []
    start = prev = None
    for n in sorted(numbers):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}–{prev}" if prev != start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}–{prev}" if prev != start else str(start))
    return ", ".join(out)


def document_flags(result: DocumentResult) -> dict[str, dict[str, list[int]]]:
    """{kind: {reason: [page numbers]}} for one document."""
    pages = result.pages
    flags: dict[str, dict[str, list[int]]] = {
        "review": defaultdict(list),
        "expected": defaultdict(list),
    }
    for page in pages:
        verdict = classify(page, len(pages))
        if verdict:
            kind, reason = verdict
            flags[kind][reason].append(page["number"])
    return flags


def render(run: ParseRun) -> str:
    results = sorted(run.results, key=lambda r: (r.document.course, str(r.document.source)))
    all_pages = [p for r in results for p in r.pages]
    fresh = [r for r in results if not r.from_cache]
    flags = {id(r): document_flags(r) for r in results}
    review_total = sum(len(v) for f in flags.values() for v in f["review"].values())

    out = [
        f"# Парсинг корпуса — {date.today().isoformat()}",
        "",
        "Отчёт не попадает в git: он перечисляет имена файлов из `data/`. "
        "Содержимого методичек здесь нет — только измерения.",
        "",
        f"Парсер `{parser_key()}`, очистка r{CLEANING_REVISION}.",
        "",
        "## Итог",
        "",
        f"- выбрано документов: **{len(run.selected)}**; разобрано сейчас: {len(fresh)}, "
        f"из кэша: {len(results) - len(fresh)}, пропущено: {len(run.skipped)}, "
        f"с ошибкой: {len(run.failed)}",
        f"- страниц: **{thousands(len(all_pages))}**, символов текста: "
        f"**{thousands(sum(p['chars'] for p in all_pages))}**",
        f"- заголовков: {thousands(sum(p['headings'] for p in all_pages))}, "
        f"формул: {thousands(sum(p['formulas'] for p in all_pages))} "
        f"(ещё {thousands(sum(p.get('formula_prose', 0) for p in all_pages))} «формул» "
        "оказались прозой — текст сохранён), "
        f"рисунков: {thousands(sum(p['pictures'] for p in all_pages))}, "
        f"листингов кода: {thousands(sum(p['code_blocks'] for p in all_pages))}",
        f"- колонтитулов убрала модель разметки: "
        f"{thousands(sum(p['running_boxes'] for p in all_pages))}, "
        f"добрала очистка: {thousands(sum(p['edge_lines_removed'] for p in all_pages))}; "
        f"склеено переносов: {thousands(sum(p['hyphens_joined'] for p in all_pages))}",
        f"- время: {run.seconds:.0f} с по часам; парсинг занял "
        f"{sum(r.parse_seconds for r in fresh):.0f} с в сумме по процессам-воркерам",
        f"- документов со сканом и чужим OCR-слоем: "
        f"{sum(r.document.text_origin == 'ocr-layer' for r in results)} "
        f"({thousands(sum(len(r.pages) for r in results if r.document.text_origin == 'ocr-layer'))}"
        " стр.) — помечены *(OCR-слой)*; распознанных нашим OCR: "
        f"{sum(r.document.text_origin == 'ocr' for r in results)} "
        f"({thousands(sum(len(r.pages) for r in results if r.document.text_origin == 'ocr'))}"
        " стр.) — *(наш OCR)*",
        f"- **страниц на ручную проверку: {review_total}**",
        "",
    ]

    if run.skipped or run.failed or run.undecided:
        out += ["## Не разобрано", ""]
        out += [f"- `{d.source}` — {d.skip_reason}" for d in run.skipped]
        out += [f"- `{path}` — ошибка: {error}" for path, error in run.failed.items()]
        if run.undecided:
            out += [
                f"- {len(run.undecided)} файлов включённых предметов нет в `courses.yaml` — "
                "запустите `nsu-rag inventory`, затем решите, включать ли их:"
            ]
            out += [f"  - `{path}`" for path in run.undecided]
        out += [""]

    out += [
        "## По документам",
        "",
        "Символов — медиана по страницам, без меток `[формула]` и `[рисунок]`.",
        "",
        "| документ | стр. | симв./стр. | заголовков | формул | рисунков | код |"
        " колонтитулов | переносов | на проверку | парсинг, с |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        pages = r.pages
        review = sum(len(v) for v in flags[id(r)]["review"].values())
        out.append(
            _row(
                [
                    f"`{r.document.source}`"
                    + {"ocr-layer": " *(OCR-слой)*", "ocr": " *(наш OCR)*"}.get(
                        r.document.text_origin, ""
                    ),
                    str(len(pages)),
                    f"{statistics.median([p['chars'] for p in pages]):.0f}" if pages else "-",
                    str(sum(p["headings"] for p in pages)),
                    str(sum(p["formulas"] for p in pages)),
                    str(sum(p["pictures"] for p in pages)),
                    str(sum(p["code_blocks"] for p in pages)),
                    str(sum(p["running_boxes"] + p["edge_lines_removed"] for p in pages)),
                    str(sum(p["hyphens_joined"] for p in pages)),
                    f"**{review}**" if review else "0",
                    f"{r.parse_seconds:.0f}" + (" (кэш)" if r.from_cache else ""),
                ]
            )
        )
    out += [""]

    out += [
        "## Страницы для ручной проверки",
        "",
        "Номера — страницы PDF-файла, считая с 1 (ADR-011). «Ожидаемо тонкие» объяснены "
        "содержимым — рисунок, формулы, обложка — и перечислены только для полноты.",
        "",
    ]
    any_listed = False
    for r in results:
        f = flags[id(r)]
        if not (f["review"] or f["expected"]):
            continue
        any_listed = True
        out.append(f"**`{r.document.source}`**")
        out.append("")
        for reason, numbers in sorted(f["review"].items()):
            out.append(f"- проверить — {reason}: стр. {_ranges(numbers)}")
        expected = ", ".join(
            f"{reason} ({len(numbers)})" for reason, numbers in sorted(f["expected"].items())
        )
        if expected:
            out.append(f"- ожидаемо тонкие: {expected}")
        out.append("")
    if not any_listed:
        out += ["Ни одной подозрительной страницы.", ""]

    out += [
        "## Как читать",
        "",
        "- `[формула]` — на этом месте была выносная формула. Текстовый слой PDF её не "
        "передаёт: старый рендер давал по символу на строку. Ответ может сослаться на "
        "страницу, где она стоит (ADR-012). Если в «формуле» 8+ слов, это абзац с "
        "инлайн-математикой, и его текст сохранён как есть.",
        "- `[рисунок]` — рисунок; слова, напечатанные внутри него, сохранены следом.",
        "- *(OCR-слой)* — скан, текстовый слой которого сделал чужой OCR: текст плотный, "
        "но с ошибками того распознавания (ADR-017).",
        "- Листинги кода — в блоках ```` ``` ````, строки и отступы восстановлены по координатам.",
        f"- «разбор потерял текст» — в текстовом слое страницы больше {LOST_MIN_SOURCE} "
        f"символов, а на выходе меньше {KEPT_SHARE:.0%} от них, и формул с рисунками, "
        "которые это объяснили бы, нет. Самый важный сигнал: страницу стоит открыть.",
        "- «пустая и в самом PDF» — текстового слоя почти нет и в исходнике: "
        "обычно пустой разворот перед главой.",
        f"- «мусор» — больше {HIGH_JUNK:.0%} и не меньше {JUNK_MIN_CHARS} битых символов "
        "(`U+FFFD`, области частного использования).",
        "",
    ]
    return "\n".join(out)
