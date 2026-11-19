from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

GOOD_TEXT_CHARS_PER_PAGE = 500
WEAK_TEXT_CHARS_PER_PAGE = 80

IMAGE_BACKED_COVER = 0.9
OCR_LAYER_SHARE = 0.5

SAMPLE_PAGES = 12
SKIP_LEADING_PAGES = 2

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
JUNK_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
REPLACEMENT_CHAR = "\ufffd"


def is_junk(char: str) -> bool:
    if char.isspace():
        return False
    if char == REPLACEMENT_CHAR:
        return True
    return unicodedata.category(char) in JUNK_CATEGORIES


@dataclass
class FileReport:
    path: Path
    """Path relative to the corpus root."""

    course: str
    semester: int | None
    size_bytes: int
    sha256: str
    fmt: str

    pages: int | None = None
    encrypted: bool = False
    chars_per_page: float = 0.0
    junk_ratio: float = 0.0
    language: str = "unknown"
    image_backed_share: float = 0.0
    """Share of sampled pages whose text lies under a near-full-page picture."""
    error: str | None = None

    duplicate_of: Path | None = None
    manifest_subject: str | None = None
    manifest_semester: int | None = None

    @property
    def text_layer(self) -> str:
        if self.fmt != "pdf":
            return "n/a"
        if self.error or self.encrypted:
            return "unknown"
        if self.chars_per_page >= GOOD_TEXT_CHARS_PER_PAGE:
            return "good"
        if self.chars_per_page >= WEAK_TEXT_CHARS_PER_PAGE:
            return "weak"
        return "none"

    @property
    def needs_ocr(self) -> bool:
        return self.text_layer in {"none", "weak"}

    @property
    def text_origin(self) -> str:
        if self.fmt != "pdf" or self.text_layer in {"none", "unknown"}:
            return "n/a"
        return "ocr-layer" if self.image_backed_share >= OCR_LAYER_SHARE else "digital"

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None


@dataclass
class ManifestRow:
    year: str
    semester: str
    subject: str
    filename: str
    status: str

    @property
    def is_error(self) -> bool:
        return not self.status.startswith("уже есть")


@dataclass
class Inventory:
    files: list[FileReport] = field(default_factory=list)
    manifest_missing: list[ManifestRow] = field(default_factory=list)
    """Manifest rows with no file on disk - downloads that failed."""
    unlisted: list[FileReport] = field(default_factory=list)
    """Files on disk that no manifest row mentions."""

    @property
    def primary_files(self) -> list[FileReport]:
        return [f for f in self.files if not f.is_duplicate]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def text_stats(text: str) -> tuple[float, str]:
    if not text:
        return 0.0, "unknown"

    normalised = unicodedata.normalize("NFC", text)
    junk = sum(1 for char in normalised if is_junk(char))
    junk_ratio = junk / len(normalised)

    cyrillic = len(CYRILLIC.findall(normalised))
    latin = len(LATIN.findall(normalised))
    letters = cyrillic + latin
    if letters < 50:
        language = "unknown"
    else:
        share = cyrillic / letters
        language = "ru" if share > 0.7 else "en" if share < 0.3 else "mixed"
    return junk_ratio, language


def sample_indices(page_count: int) -> list[int]:
    first = SKIP_LEADING_PAGES if page_count > 2 * SKIP_LEADING_PAGES else 0
    body = page_count - first
    if body <= 0:
        return list(range(page_count))
    if body <= SAMPLE_PAGES:
        return list(range(first, page_count))
    step = body / SAMPLE_PAGES
    return [first + int(i * step) for i in range(SAMPLE_PAGES)]


def is_image_backed(page: pymupdf.Page) -> bool:
    area = page.rect.width * page.rect.height
    if not area:
        return False
    for info in page.get_image_info():
        drawn = pymupdf.Rect(info["bbox"]) & page.rect
        if drawn.width * drawn.height >= IMAGE_BACKED_COVER * area:
            return True
    return False


def inspect_pdf(path: Path, report: FileReport) -> None:
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                report.encrypted = True
                report.pages = doc.page_count
                return
            report.pages = doc.page_count
            if not doc.page_count:
                return
            sampled = sample_indices(doc.page_count)
            chunks, backed = [], 0
            for i in sampled:
                page = doc.load_page(i)
                chunks.append(page.get_text())
                backed += is_image_backed(page)
            report.image_backed_share = backed / len(sampled)
    except Exception as exc:  # noqa: BLE001 - one broken file must not stop the pass
        report.error = f"{type(exc).__name__}: {exc}"
        return

    sample = "".join(chunks)
    report.chars_per_page = len(sample) / len(sampled)
    report.junk_ratio, report.language = text_stats(sample)


def inspect_docx(path: Path, report: FileReport) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"
        return

    text = re.sub(r"<[^>]+>", " ", xml)
    report.chars_per_page = float(len(text))
    report.junk_ratio, report.language = text_stats(text)


def course_and_semester(relative: Path) -> tuple[str, int | None]:
    parts = relative.parts
    course = parts[0] if parts else "?"
    semester = None
    if len(parts) > 1 and parts[1].startswith("sem"):
        suffix = parts[1].removeprefix("sem")
        if suffix.isdigit():
            semester = int(suffix)
    return course, semester


def inspect_file(path: Path, root: Path) -> FileReport:
    relative = path.relative_to(root)
    course, semester = course_and_semester(relative)
    report = FileReport(
        path=relative,
        course=course,
        semester=semester,
        size_bytes=path.stat().st_size,
        sha256=sha256_of(path),
        fmt=path.suffix.lower().lstrip("."),
    )
    if report.fmt == "pdf":
        inspect_pdf(path, report)
    elif report.fmt == "docx":
        inspect_docx(path, report)
    else:
        report.error = "unsupported format"
    return report


def primary_key(report: FileReport) -> tuple[bool, int, int, str]:
    return (
        report.manifest_subject is None,
        report.semester if report.semester is not None else 99,
        len(str(report.path)),
        str(report.path),
    )


def mark_duplicates(reports: list[FileReport]) -> None:
    by_hash: dict[str, list[FileReport]] = defaultdict(list)
    for report in reports:
        by_hash[report.sha256].append(report)

    for copies in by_hash.values():
        if len(copies) < 2:
            continue
        primary = min(copies, key=primary_key)
        for copy in copies:
            if copy is not primary:
                copy.duplicate_of = primary.path


# кириллица, байт в байт как в манифесте
SUBJECT_SLUG = {
    "Линал": "linal",
    "Матанализ": "matan",
    "Матлог": "matlog",
    "Прога": "proga",
    "Схемотехника": "shemotehnika",
    "Физика": "fizika",
    "ДГМА": "dgma",
    "Дискретка": "diskretka",
    "ЛОП": "lop",
    "ЭВМ": "evm",
    "ОПП": "opp",
    "ОСи": "os",
    "Сети": "seti",
    "ТВиМС": "tvims",
    "ТФКП и Диффуры": "tfkp-diffury",
    "БД": "bd",
    "ВычМат": "vychmat",
    "МТК": "mtk",
    "Элтех": "eltech",
    "ОТУ": "otu",
    "Теоркод": "teorkod",
    "ИСИС": "isis",
}


def load_manifest(path: Path) -> list[ManifestRow]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            ManifestRow(
                year=row["курс"].strip(),
                semester=row["семестр"].strip(),
                subject=row["предмет"].strip(),
                filename=row["файл"].strip(),
                status=row["статус"].strip(),
            )
            for row in csv.DictReader(handle)
        ]


def reconcile(inventory: Inventory, rows: list[ManifestRow]) -> None:
    by_name: dict[str, list[FileReport]] = defaultdict(list)
    for report in inventory.files:
        by_name[report.path.name].append(report)

    claimed: set[int] = set()

    def claim(row: ManifestRow, candidates: list[FileReport]) -> FileReport | None:
        wanted_slug = SUBJECT_SLUG.get(row.subject)
        wanted_sem = int(row.semester) if row.semester.isdigit() else None
        free = [c for c in candidates if id(c) not in claimed]
        if not free:
            return None
        for test in (
            lambda c: c.course == wanted_slug and c.semester == wanted_sem,
            lambda c: c.course == wanted_slug,
            lambda _: True,
        ):
            match = next((c for c in free if test(c)), None)
            if match is not None:
                return match
        return None

    for row in rows:
        candidates = by_name.get(row.filename, [])
        if not candidates:
            inventory.manifest_missing.append(row)
            continue
        chosen = claim(row, candidates)
        if chosen is None:
            continue
        chosen.manifest_subject = row.subject
        chosen.manifest_semester = int(row.semester) if row.semester.isdigit() else None
        claimed.add(id(chosen))

    inventory.unlisted = [f for f in inventory.files if id(f) not in claimed]


def take_inventory(
    root: Path,
    manifest: Path | None = None,
    on_file: Callable[[Path], None] | None = None,
) -> Inventory:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    reports = []
    for path in paths:
        reports.append(inspect_file(path, root))
        if on_file is not None:
            on_file(path)
    inventory = Inventory(files=reports)
    if manifest is not None:
        reconcile(inventory, load_manifest(manifest))
    mark_duplicates(inventory.files)
    return inventory
