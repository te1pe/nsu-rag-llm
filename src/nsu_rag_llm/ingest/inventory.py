"""Take stock of the corpus before touching it.

The point of this pass is to answer, per file: can it be read at all, does it
carry a usable text layer, what language is it in, and is it a copy of
something already seen. Everything downstream - which parser to route a file
to, what the OCR bill will be, which subjects make a sensible MVP - follows
from those answers, so they are measured rather than assumed.

Nothing here writes to `data/raw`: the corpus is read-only.
"""

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

# A page of typeset prose holds roughly 1500-3000 characters. A scan whose
# only "text" is a stray watermark lands near zero. The middle band is the
# interesting one: a partial or broken text layer that would quietly produce
# garbage chunks if it were trusted.
GOOD_TEXT_CHARS_PER_PAGE = 500
WEAK_TEXT_CHARS_PER_PAGE = 80

#: A page whose text sits under a picture covering at least this share of it
#: is a scan that somebody else ran OCR on. Its text layer can be perfectly
#: dense and still carry that OCR's mistakes, so it is reported separately:
#: phase 2 breaks retrieval metrics down by where the text came from.
IMAGE_BACKED_COVER = 0.9
#: A document is an OCR layer when at least this share of its sampled pages is.
OCR_LAYER_SHARE = 0.5

#: How many pages to sample per document when estimating the text layer.
SAMPLE_PAGES = 12
#: Front matter is often a scanned cover even in a born-digital file.
SKIP_LEADING_PAGES = 2

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
#: What a broken text layer leaves behind. Control, format, surrogate,
#: private-use and unassigned code points are never real prose; a PDF with a
#: damaged ToUnicode map typically extracts as private-use glyphs, and a
#: mis-declared encoding produces U+FFFD. Whitespace is category Cc but is of
#: course legitimate, so it is excluded.
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
    """Everything the inventory learned about one file."""

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

    #: Filled in after every file has been seen.
    duplicate_of: Path | None = None
    manifest_subject: str | None = None
    manifest_semester: int | None = None

    @property
    def text_layer(self) -> str:
        """How much of a text layer this file has: good, weak, or none."""
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
        """`digital` for born-digital text, `ocr-layer` for a scan someone OCR'd."""
        if self.fmt != "pdf" or self.text_layer in {"none", "unknown"}:
            return "n/a"
        return "ocr-layer" if self.image_backed_share >= OCR_LAYER_SHARE else "digital"

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None


@dataclass
class ManifestRow:
    """One line of the nsumedia manifest."""

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
    """The whole corpus, plus what could not be matched up."""

    files: list[FileReport] = field(default_factory=list)
    manifest_missing: list[ManifestRow] = field(default_factory=list)
    """Manifest rows with no file on disk - downloads that failed."""
    unlisted: list[FileReport] = field(default_factory=list)
    """Files on disk that no manifest row mentions."""

    @property
    def primary_files(self) -> list[FileReport]:
        """Files excluding duplicate copies."""
        return [f for f in self.files if not f.is_duplicate]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def text_stats(text: str) -> tuple[float, str]:
    """Return the junk ratio and the dominant script of a text sample."""
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
    """Pick pages spread through the document, skipping the front matter.

    Front matter is only worth skipping when there is a body to skip it for:
    in a three-page handout, dropping two pages would leave the estimate
    resting on a single page.
    """
    first = SKIP_LEADING_PAGES if page_count > 2 * SKIP_LEADING_PAGES else 0
    body = page_count - first
    if body <= 0:
        return list(range(page_count))
    if body <= SAMPLE_PAGES:
        return list(range(first, page_count))
    step = body / SAMPLE_PAGES
    return [first + int(i * step) for i in range(SAMPLE_PAGES)]


def is_image_backed(page: pymupdf.Page) -> bool:
    """Does a single picture cover (nearly) the whole page?

    `get_image_info` reports where images are drawn without decoding their
    pixels, so this stays cheap even on 2000x2600 scans.
    """
    area = page.rect.width * page.rect.height
    if not area:
        return False
    for info in page.get_image_info():
        drawn = pymupdf.Rect(info["bbox"]) & page.rect
        if drawn.width * drawn.height >= IMAGE_BACKED_COVER * area:
            return True
    return False


def inspect_pdf(path: Path, report: FileReport) -> None:
    """Fill in page count, text-layer strength, junk ratio and language."""
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
    """Read the document body straight out of the archive.

    A single DOCX does not justify pulling in a parsing library at this stage;
    phase 1b hands the file to docling anyway.
    """
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
    """Read the course slug and semester out of `<slug>/sem<N>/...`."""
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
    """Sort key for choosing which copy of a duplicate is the real one.

    The copies are byte-identical, so nothing about the content can decide;
    what differs is provenance. A copy the nsumedia manifest lists was filed
    under a subject by whoever built the course materials, while a loose book
    was filed by the reorganisation (ADR-005) - so the manifest copy wins. Then
    the earliest semester, then the shortest path, purely to be deterministic.
    """
    return (
        report.manifest_subject is None,
        report.semester if report.semester is not None else 99,
        len(str(report.path)),
        str(report.path),
    )


def mark_duplicates(reports: list[FileReport]) -> None:
    """Point every copy at one primary, chosen by `primary_key`.

    Must run after `reconcile`, because the choice depends on which copies the
    manifest lists.
    """
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


#: Subject name in the manifest -> course slug on disk (see ADR-005).
#: The keys are Cyrillic and must match the manifest byte for byte.
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
    """Read the nsumedia manifest.

    It is UTF-8 with a BOM and quotes filenames containing commas, so it is
    read with `utf-8-sig` through the csv module rather than split by hand.
    """
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
    """Match files against manifest rows, both ways.

    The manifest describes the nsumedia tree and nothing else, so a file with
    no row is a book that was sitting loose in the root of `data/raw`, and a
    row with no file at all is a download that never completed.

    A book can be listed twice - the same title is set in two semesters of one
    subject, and a copy sits in each semester's folder. Each row therefore
    claims a file of its own: matching on the name alone would hand both rows
    the same copy and leave the other looking unlisted.
    """
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
            # The file exists but every copy is already spoken for: the
            # manifest lists it more times than it appears on disk.
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
    """Walk the corpus and describe every file in it.

    `on_file` is called after each file so a caller can show progress: reading
    2.6 GB takes long enough that silence would look like a hang.
    """
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    reports = []
    for path in paths:
        reports.append(inspect_file(path, root))
        if on_file is not None:
            on_file(path)
    inventory = Inventory(files=reports)
    # Reconcile first: which copy of a duplicate is primary depends on which
    # copies the manifest lists.
    if manifest is not None:
        reconcile(inventory, load_manifest(manifest))
    mark_duplicates(inventory.files)
    return inventory
