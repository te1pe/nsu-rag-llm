"""Turn an Inventory into the Markdown report the author reads.

The report lists file names, page counts and measurements. It never quotes the
text of a document: the corpus is third-party material, and `data/` - this
report included - stays out of git.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from nsu_rag_llm.ingest.inventory import FileReport, Inventory

#: Above this, a single document is big enough to crowd out everything else in
#: a top-k result list, and big enough to dominate an indexing run.
GIANT_PAGES = 1500


@dataclass
class CourseSummary:
    course: str
    files: int
    pages: int
    good_pages: int
    scan_files: int
    duplicates: int
    giant_files: int
    languages: str
    usable_files: int
    usable_pages: int
    usable_languages: str
    """What an MVP index would actually get out of this course."""

    @property
    def good_share(self) -> float:
        return self.good_pages / self.pages if self.pages else 0.0


@dataclass
class CostEstimate:
    """Measured throughput, extrapolated over the corpus.

    Kept in a sidecar file next to the report: benchmarking takes minutes and
    needs the optional extras, so a plain inventory run must not silently
    throw away the last measurement.
    """

    ocr_seconds_per_page: float | None = None
    ocr_pages: int = 0
    embed_chunks_per_second: float | None = None
    embed_chunks_total: int = 0
    measured_on: str = ""
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if not self.measured_on:
            self.measured_on = date.today().isoformat()

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> CostEstimate | None:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError, TypeError:
            return None

    @property
    def ocr_hours(self) -> float | None:
        if self.ocr_seconds_per_page is None:
            return None
        return self.ocr_seconds_per_page * self.ocr_pages / 3600

    @property
    def embed_minutes(self) -> float | None:
        if not self.embed_chunks_per_second:
            return None
        return self.embed_chunks_total / self.embed_chunks_per_second / 60


def is_usable(report) -> bool:
    """Could this file go into the MVP index as it stands?

    It has to be readable without OCR, not a second copy of something already
    indexed, not a doorstop that would crowd out everything else in a top-k
    list, and not in English - the evaluation questions will be Russian, and
    an English reference manual answers none of them.
    """
    return (
        not report.is_duplicate
        and report.text_layer == "good"
        and (report.pages or 0) < GIANT_PAGES
        and report.language in {"ru", "mixed"}
    )


def summarise_courses(inventory: Inventory) -> list[CourseSummary]:
    grouped: dict[str, list[FileReport]] = defaultdict(list)
    for report in inventory.files:
        grouped[report.course].append(report)

    summaries = []
    for course, reports in sorted(grouped.items()):
        primary = [r for r in reports if not r.is_duplicate]
        usable = [r for r in primary if is_usable(r)]
        pages = sum(r.pages or 0 for r in primary)
        good = sum(r.pages or 0 for r in primary if r.text_layer == "good")
        languages = sorted({r.language for r in primary if r.language != "unknown"})
        summaries.append(
            CourseSummary(
                course=course,
                files=len(primary),
                pages=pages,
                good_pages=good,
                scan_files=sum(1 for r in primary if r.needs_ocr),
                duplicates=len(reports) - len(primary),
                giant_files=sum(1 for r in primary if (r.pages or 0) >= GIANT_PAGES),
                languages="/".join(languages) or "-",
                usable_files=len(usable),
                usable_pages=sum(r.pages or 0 for r in usable),
                usable_languages="/".join(sorted({r.language for r in usable})) or "-",
            )
        )
    return summaries


def propose_mvp(summaries: list[CourseSummary], want: int = 5) -> list[CourseSummary]:
    """Rank courses by how well they would carry phases 1b-4 as they stand.

    The ranking runs on `usable_pages` - Russian, readable, non-duplicate, not
    a doorstop - because that is what an index built today would contain. A
    course with ten thousand pages of English reference manual scores zero
    here, which is the point: it would cost the most and answer the least.
    """

    def score(s: CourseSummary) -> float:
        if s.usable_files < 2 or s.usable_pages < 300:
            return -1.0  # too thin to say anything about retrieval
        if s.usable_pages > 6000:
            return -1.0  # one subject would dominate every iteration
        # Prefer courses where little is left behind, and where the pages are
        # spread over several documents rather than one fat book.
        coverage = s.usable_pages / s.pages if s.pages else 0.0
        spread = min(s.usable_files, 8) / 8
        return 0.7 * coverage + 0.3 * spread

    ranked = sorted(summaries, key=lambda s: (-score(s), -s.usable_pages))
    return [s for s in ranked if score(s) > 0][:want]


def thousands(value: float) -> str:
    """Group digits with a non-breaking space.

    Doing this with a blanket `replace(",", " ")` over a whole sentence eats
    the commas in the Russian prose around the number.
    """
    return f"{value:,.0f}".replace(",", "\u00a0")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _summary_section(inventory: Inventory) -> list[str]:
    primary = inventory.primary_files
    good = [f for f in primary if f.text_layer == "good"]
    weak = [f for f in primary if f.text_layer == "weak"]
    none = [f for f in primary if f.text_layer == "none"]
    return [
        "## Итог",
        "",
        f"- файлов на диске: **{len(inventory.files)}**, из них уникальных: "
        f"**{len(primary)}** (дубликатов: {len(inventory.files) - len(primary)})",
        f"- страниц без дубликатов: **{thousands(sum(f.pages or 0 for f in primary))}**",
        f"- размер: **{sum(f.size_bytes for f in primary) / 2**30:.1f} ГБ**",
        f"- из текстовых файлов **{sum(f.text_origin == 'ocr-layer' for f in good)}** — "
        f"сканы с чужим OCR-слоем под картинкой "
        f"({thousands(sum(f.pages or 0 for f in good if f.text_origin == 'ocr-layer'))} стр.): "
        "текст плотный, но с ошибками того OCR",
        "",
        "| текстовый слой | файлов | страниц | что с этим делать |",
        "|---|---:|---:|---|",
        _row(
            [
                "good (≥500 симв./стр.)",
                str(len(good)),
                thousands(sum(f.pages or 0 for f in good)),
                "`pymupdf4llm`, быстрый путь",
            ]
        ),
        _row(
            [
                "weak (80–499)",
                str(len(weak)),
                thousands(sum(f.pages or 0 for f in weak)),
                "текст есть, но рваный — надёжнее OCR",
            ]
        ),
        _row(
            [
                "none (<80)",
                str(len(none)),
                thousands(sum(f.pages or 0 for f in none)),
                "скан — только OCR",
            ]
        ),
        "",
    ]


def _broken_section(inventory: Inventory) -> list[str]:
    broken = [f for f in inventory.files if f.error or f.encrypted]
    if not broken:
        return []
    out = ["### Не читаются", "", "| файл | причина |", "|---|---|"]
    out += [_row([f"`{f.path}`", "зашифрован" if f.encrypted else f.error or "?"]) for f in broken]
    return [*out, ""]


def _courses_section(summaries: list[CourseSummary]) -> list[str]:
    out = [
        "## По предметам",
        "",
        "«Пригодно» — то, что попало бы в индекс сегодня: читается без OCR, "
        "не дубликат, "
        f"меньше {GIANT_PAGES} страниц и по-русски.",
        "",
        "| предмет | файлов | страниц | с текстом | доля | пригодных файлов |"
        " пригодных страниц | нужен OCR | дублей | гигантов | язык |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        out.append(
            _row(
                [
                    f"`{s.course}`",
                    str(s.files),
                    thousands(s.pages),
                    thousands(s.good_pages),
                    f"{s.good_share:.0%}",
                    str(s.usable_files),
                    thousands(s.usable_pages),
                    str(s.scan_files),
                    str(s.duplicates),
                    str(s.giant_files),
                    s.languages,
                ]
            )
        )
    return [*out, ""]


def _duplicates_section(inventory: Inventory) -> list[str]:
    duplicates = [f for f in inventory.files if f.is_duplicate]
    out = ["## Дубликаты (совпадение sha256)", ""]
    if not duplicates:
        return [*out, "Дубликатов нет.", ""]
    wasted = sum(f.size_bytes for f in duplicates) / 2**30
    out += [
        f"Лишних копий: **{len(duplicates)}** ({wasted:.1f} ГБ). Основной "
        "считается копия из манифеста nsumedia, затем из более раннего "
        "семестра, затем с более коротким путём.",
        "",
        "| копия | основной |",
        "|---|---|",
    ]
    out += [_row([f"`{f.path}`", f"`{f.duplicate_of}`"]) for f in duplicates]
    return [*out, ""]


def _manifest_section(inventory: Inventory) -> list[str]:
    out = [
        "## Сверка с манифестом",
        "",
        f"- строк манифеста без файла на диске: **{len(inventory.manifest_missing)}**",
        f"- файлов без строки в манифесте: **{len(inventory.unlisted)}** — "
        "книги, лежавшие россыпью в корне `data/raw`",
        "",
    ]
    if inventory.manifest_missing:
        out += ["| предмет | файл | статус |", "|---|---|---|"]
        out += [_row([r.subject, f"`{r.filename}`", r.status]) for r in inventory.manifest_missing]
        out += [""]
    return out


def _costs_section(inventory: Inventory, costs: CostEstimate | None) -> list[str]:
    ocr_pages = sum(f.pages or 0 for f in inventory.primary_files if f.needs_ocr)
    out = [
        "## Оценка стоимости",
        "",
        f"Страниц, которым нужен OCR: **{thousands(ocr_pages)}**.",
        "",
    ]
    if costs is None:
        return [
            *out,
            "Время не измерялось. Запустите `nsu-rag inventory --benchmark` "
            "(нужны extras `ingest` и `ml`).",
            "",
        ]

    if costs.measured_on != date.today().isoformat():
        out += [f"Замеры от {costs.measured_on}, в этом прогоне не повторялись.", ""]

    if costs.ocr_seconds_per_page is not None:
        out.append(
            f"- **OCR: {costs.ocr_seconds_per_page:.1f} с/стр.** измерено; на "
            f"{thousands(costs.ocr_pages)} страниц это **≈ {costs.ocr_hours:.0f} ч** "
            "в один поток"
        )
    if costs.embed_chunks_per_second:
        out.append(
            f"- **эмбеддинги: {costs.embed_chunks_per_second:.0f} фрагм./с**; на "
            f"≈{thousands(costs.embed_chunks_total)} фрагментов это "
            f"**≈ {costs.embed_minutes:.0f} мин**"
        )
    out += ["", "Как мерили:", ""]
    out += [f"- {note}" for note in costs.notes]
    return [*out, ""]


def _mvp_section(summaries: list[CourseSummary]) -> list[str]:
    mvp = propose_mvp(summaries)
    out = [
        "## Предложение MVP-корпуса",
        "",
        "Ранжирование идёт по **пригодным** страницам, а не по общему объёму: "
        "это то, что реально попало бы в индекс сегодня. Предмет из десяти "
        "тысяч страниц английского справочника получает ноль — он стоил бы "
        "дороже всех и не ответил бы ни на один русский вопрос.",
        "",
        "Отсечки: минимум 2 файла и 300 страниц, максимум 6000 — чтобы один "
        "предмет не забивал каждую итерацию. Дальше — доля предмета, которая "
        "переживает фильтр, и разброс по документам.",
        "",
    ]
    if not mvp:
        return [*out, "Ни один предмет не проходит отсечки — смотри таблицу выше.", ""]

    out += [
        "| предмет | пригодных файлов | пригодных страниц | доля предмета | язык |",
        "|---|---:|---:|---:|---|",
    ]
    out += [
        _row(
            [
                f"`{s.course}`",
                str(s.usable_files),
                thousands(s.usable_pages),
                f"{s.usable_pages / s.pages:.0%}" if s.pages else "-",
                s.usable_languages,
            ]
        )
        for s in mvp
    ]
    out += [
        "",
        f"Итого: **{thousands(sum(s.usable_pages for s in mvp))} страниц** "
        f"в {sum(s.usable_files for s in mvp)} файлах — "
        "это выбор алгоритма, последнее слово за автором.",
        "",
    ]
    return out


def _files_section(inventory: Inventory) -> list[str]:
    out = [
        "## Все файлы",
        "",
        "| предмет | сем | файл | стр. | симв./стр. | слой | происхождение | мусор | язык |"
        " пригоден | sha256 |",
        "|---|---:|---|---:|---:|---|---|---:|---|---|---|",
    ]
    for f in sorted(inventory.files, key=lambda r: (r.course, r.semester or 0, str(r.path))):
        mark = " *(дубль)*" if f.is_duplicate else ""
        out.append(
            _row(
                [
                    f"`{f.course}`",
                    str(f.semester or "-"),
                    f"`{f.path.name}`{mark}",
                    str(f.pages or "-"),
                    f"{f.chars_per_page:.0f}",
                    f.text_layer,
                    f.text_origin,
                    f"{f.junk_ratio:.1%}",
                    f.language,
                    "да" if is_usable(f) else "нет",
                    f.sha256[:12],
                ]
            )
        )
    return [*out, ""]


def render(inventory: Inventory, costs: CostEstimate | None = None) -> str:
    """Build the whole report."""
    summaries = summarise_courses(inventory)
    parts = [
        f"# Инвентаризация корпуса — {date.today().isoformat()}",
        "",
        "Отчёт не попадает в git: он перечисляет имена файлов из `data/`. "
        "Содержимого методичек здесь нет — только измерения.",
        "",
        *_summary_section(inventory),
        *_broken_section(inventory),
        *_courses_section(summaries),
        *_duplicates_section(inventory),
        *_manifest_section(inventory),
        *_costs_section(inventory, costs),
        *_mvp_section(summaries),
        *_files_section(inventory),
    ]
    return "\n".join(parts)
