# Architecture decision records

Short records of the decisions that were not obvious, each with the
alternatives that were rejected and why. Newest last.

## Index

**Environment and configuration**

- ADR-001. Python 3.14 rather than the 3.12 named in CLAUDE.md
- ADR-002. Sampling parameters live in configuration, not in the Modelfile
- ADR-003. Thinking is switched off with `reasoning_effort: "none"`
- ADR-004. `num_ctx 8192` is pinned in `ollama/Modelfile`

**The corpus: inventory and layout**

- ADR-005. Course materials are laid out as `data/raw/<slug>/sem<N>/`
- ADR-007. A document's text layer is judged by characters per sampled page
- ADR-008. The MVP is ranked on "usable pages", not on total pages
- ADR-010. `courses.yaml` is regenerated without touching the author's edits
- ADR-016. The primary copy of a duplicate comes from the manifest, and include decisions follow the content
- ADR-017. Where a text layer came from is recorded: born-digital or somebody else's OCR

**Parsing and OCR**

- ADR-009. OCR runs through RapidOCR with the language code `iso:ru`
- ADR-011. A page number is the 1-based position of the page in the PDF file
- ADR-012. PDFs are parsed by pymupdf4llm's layout model, with two repairs
- ADR-013. The parse cache is keyed by content hash and parser version; cleaning is never cached
- ADR-014. Two parse processes, not four
- ADR-015. The running-header safety net only touches the outermost line, on most pages
- ADR-018. Parse quality is judged against each page's own text layer
- ADR-019. Private-use glyphs are mapped only when their font is known
- ADR-030. The full corpus is a second index built by the same pipeline

**Chunking and indexing**

- ADR-020. Chunks follow headings where the text allows, and never outgrow the model
- ADR-021. Indexing is incremental because chunk ids come from content

**Retrieval**

- ADR-022. BM25 is written out, and stems each word in its own language
- ADR-023. Hybrid search fuses by rank; the reranker re-reads a shortlist

**Answering**

- ADR-024. The prompt is measured with the model's own tokenizer before it is sent
- ADR-025. The model cites a label; the code checks it and writes the citation
- ADR-026. Ten fragments per answer; no fragments, no model

**Evaluation**

- ADR-028. The judge rules on facts it can check, in JSON, twice per answer
- ADR-029. Models are compared alone, on the same fragments, by a rule fixed in advance

**The agent**

- ADR-031. The agent is a small graph of plain functions; LangGraph holds the edges

**The service and its packaging**

- ADR-006. Health probes are injected as callables, and run concurrently
- ADR-027. One pipeline behind the CLI, `/ask` and a Server-Sent Events stream
- ADR-032. The container answers too: CPU search, hybrid only, the index mounted

---

## ADR-001. Python 3.14 rather than the 3.12 named in CLAUDE.md

**Context.** CLAUDE.md specifies Python 3.12. No 3.12 was installed on the
machine, so the version had to be chosen deliberately rather than inherited.
The stack includes `torch`, `sentence-transformers` and `docling`, which lag
behind new Python releases, so "newest" is not automatically safe.

**Decision.** Python 3.14.7, installed and pinned through uv (`.python-version`).

**How it was checked, rather than assumed.** The full dependency set resolved
under 3.14 to 139 packages. Of those, 100 are pure Python and 37 ship
`cp314` wheels for macOS arm64. Exactly two had no such wheel:
`antlr4-python3-runtime` (source distribution only) and `multiprocess` (mac
wheels for other Python versions only). Both were installed into a scratch
3.14 environment and built from source in seconds, with the Xcode command line
tools already present. Nothing in the stack blocks 3.14.

**Alternatives rejected.**
- *3.12, as written in CLAUDE.md.* Would have to be downloaded anyway, and
  starting a new project two releases behind buys nothing here.
- *3.15.* Only a release candidate at the time of writing.

---

## ADR-002. Sampling parameters live in configuration, not in the Modelfile

**Context.** `qwen3.5:9b` carries its own defaults: `temperature 1`,
`top_p 0.95`, `top_k 20` and `presence_penalty 1.5`. These are inherited by
`qwen3.5-9b-8k`. A presence penalty punishes tokens that have already appeared,
and this service is required to repeat a citation after *every* claim, so a
penalty of 1.5 works directly against the output format. A temperature of 1 is
also wrong for extractive answering.

**Decision.** Sampling is set per request from `Settings`
(`LLM_TEMPERATURE=0.0`, `LLM_TOP_P=1.0`, `LLM_PRESENCE_PENALTY=0.0`). The
Modelfile carries only `num_ctx`.

**Why not the Modelfile.** Phase 4 compares several models. If each model's
Modelfile carried its own sampling, the comparison would confound the model
with its sampling and the result would not support the choice it is meant to
justify. One knob, set identically for every candidate, keeps that comparison
honest.

**Cost.** Every call site must pass the parameters; forgetting one silently
reinstates the model's own defaults. Mitigated by building the request in one
place in `generation/`.

---

## ADR-003. Thinking is switched off with `reasoning_effort: "none"`

**Context.** `ollama show` reports that the model has a `thinking` capability
whose levels are `false, true` and whose **default is `true`**. Reasoning
traces cost tokens out of an 8192-token budget and are not wanted in a RAG
answer. The question was how to switch it off over the OpenAI-compatible
endpoint specifically, not over Ollama's native API.

**Decision.** Send `reasoning_effort: "none"` in the request body;
`LLM_REASONING_EFFORT` carries it.

**Evidence.** Ollama's OpenAI-compatibility documentation lists
`reasoning_effort` among the supported fields and states that boolean-only
models map `"none"` to `false`. Measured on one identical question:

| Request | `message.reasoning` | completion tokens |
|---|---|---|
| no parameter | full reasoning trace | 205 |
| `reasoning_effort: "none"` | `null` | 3 |

**Alternatives rejected.**
- *`PARAMETER think false` in the Modelfile.* Not a documented Modelfile
  parameter, and it would bake the choice into the model, which ADR-002 avoids.
- *`--think=false` on the CLI.* Does not apply; the service speaks HTTP.

---

## ADR-004. `num_ctx 8192` is pinned in `ollama/Modelfile`

**Context.** Ollama defaults to a 4096-token context. A RAG prompt with
retrieved chunks exceeds that, and the overflow is dropped **silently** — no
error, just a quietly truncated prompt and a worse answer.

**Decision.** `ollama/Modelfile` sets `PARAMETER num_ctx 8192`, and the model
is built with `ollama create qwen3.5-9b-8k -f ollama/Modelfile`.

**Why in the Modelfile, when ADR-002 argues the opposite for sampling.** The
context window is a property of how the model is *loaded*, not of an individual
request: it decides how much memory the server reserves. Keeping it in the
Modelfile means it survives restarts and reproduces for anyone who clones the
repository. `LLM_CONTEXT_TOKENS` mirrors the same number so that the prompt
builder plans against the real window; the two must be changed together.

**Verified.** `ollama ps` reports `CONTEXT 8192` and `100% GPU`, 5.5 GB
resident.

---

## ADR-005. Course materials are laid out as `data/raw/<slug>/sem<N>/`

**Context.** CLAUDE.md describes `data/raw/<course_slug>/...`. In practice the
directory held 30 files loose at the top level (first year, first semester) and
a `nsumedia/` tree organised as `<year> курс/<N> семестр/<Subject>/`. Nothing
could attribute a file to a course automatically.

**Decision.** One directory per subject, semester as a sub-directory:
`data/raw/matan/sem1/...`. 170 files, 23 subjects. The semester therefore comes
out of the path as metadata for free, and a subject taught across several
semesters stays a single retrieval filter.

**Slugs** are transliterations of the short names the author already uses
(`matan`, `matlog`, `diskretka`, `tvims`), because they are typed as a CLI flag
and several of them — ДГМА, ЛОП, МТК, ОТУ, ИСИС — have no meaningful English
equivalent to translate to. One exception: ОСи is `os`, matching the example in
CLAUDE.md.

**Alternatives rejected.**
- *A `file → course` mapping in `courses.yaml`.* Leaves the mapping to be
  maintained by hand and lets it drift from the filesystem.
- *A directory per subject-and-semester (`matan-sem1`).* Splits one subject
  across several filters, so a question about calculus would have to be asked
  twice.

**Reversible.** `data/.reorg-undo.sh` restores the original layout.

---

## ADR-006. Health probes are injected as callables, and run concurrently

**Context.** `/health` reports on Qdrant and on the LLM endpoint. FastAPI
resolves dependencies sequentially, so a dependency that returns an
*already-awaited* probe result would make the two probes run one after the
other — two slow dependencies would add up.

**Decision.** `get_qdrant_probe` and `get_llm_probe` return async callables.
The endpoint runs them with `asyncio.gather`. Each probe is wrapped in
`measure()`, which applies a per-dependency deadline and converts any exception
into a status rather than a 500.

**Consequences.** Tests override one factory to inject a fake probe, so the two
failure axes can be exercised independently and without a network. A test
asserts the wall-clock time of two slow probes, which would fail if the
concurrency regressed.

**Why 503 and not 200.** A health endpoint that answers 200 while a dependency
is down is worse than no health endpoint. The body still names the component
that failed and how long each probe took.

---

## ADR-007. A document's text layer is judged by characters per sampled page

**Context.** Routing decides everything downstream: a file sent to
`pymupdf4llm` when it is really a scan produces empty chunks, and a file sent
to OCR when it has a perfectly good text layer costs two seconds a page for
nothing. The corpus has 169 PDFs and no metadata saying which is which.

**Decision.** Extract text from up to 12 pages spread through the document,
skipping the first two as likely front matter, and divide by the number of
pages sampled. Above 500 characters per page the layer is `good`, below 80 it
is `none`, and in between it is `weak`.

**Why those numbers.** A page of typeset prose in this corpus holds 1500-3000
characters, so 500 is comfortably below any real page of text. A scan whose
only text is a watermark or a stamped page number lands in the tens. The band
between them is the dangerous one - a partial or broken layer - and it is
called out separately rather than being forced into one of the two verdicts.
Measured on the corpus: 97 files `good`, 4 `weak`, 43 `none`.

**Sampling rather than reading everything** keeps the pass at eleven seconds
for 2.1 GB, which matters because it is re-run after every change. Front
matter is skipped only when the document is longer than four pages, so a
three-page handout is not judged on a single page.

**Alternative rejected.** *Looking for embedded images instead.* Many
born-digital files are full of figures, and some scans embed their page as a
single image with an invisible OCR layer already present. Character density
measures the thing that actually matters.

---

## ADR-008. The MVP is ranked on "usable pages", not on total pages

**Context.** Phases 1b-4 run on a subset of the corpus. The first ranking used
each subject's share of readable pages, and proposed `evm` - which is 5594
pages of Intel's English architecture manual, exactly the kind of document
CLAUDE.md warns will crowd out everything else in a top-k result.

**Decision.** Rank on `usable_pages`: pages in files that read without OCR,
are not a duplicate copy, are under 1500 pages, and are in Russian. Require at
least 2 files and 300 pages, and no more than 6000. Among survivors, prefer
subjects where a large share of the material passes the filter and where the
pages are spread over several documents.

**Why.** The number being ranked is now the number of pages an index built
today would actually contain. A subject that is mostly English reference
manual scores zero rather than first, which is the correct answer: it would
cost the most to index and answer none of the Russian evaluation questions.

**Consequence.** The proposal is advisory. `courses.yaml` carries an `include`
flag per subject and per file, the author sets them, and the generator
preserves whatever they set.

---

## ADR-009. OCR runs through RapidOCR with the language code `iso:ru`

**Context.** The OCR bill had to be measured, not guessed, and docling offers
several engines. Two things were found by trying rather than by reading:
docling installs RapidOCR by default, and RapidOCR rejects a bare `ru`.

**Decision.** `iso:ru`, on the onnxruntime backend, one language per pass.

**What the engine actually said.** Russian exists in RapidOCR's onnxruntime
backend but not in its torch backend. Asking for `["iso:ru", "iso:en"]` does
not fail - the engine keeps the first language and silently drops the rest, so
a two-language request would have been measured under a misleading label. The
request is therefore Russian only, and English-only scans need their own pass
in phase 4.5.

**Measured.** 2.0 seconds per page (median of 4 pages, 1.9-2.8 s), about 1360
characters recovered per page, after a 6-second one-off model load. Over the
20 040 pages that need OCR that is roughly 11 hours single-threaded - which is
the whole reason phases 1b-4 run on the MVP subset instead.

**Not evaluated yet.** `ocrmac`, which would use the macOS Vision framework
and could be faster on this machine, is not installed. Worth measuring in
phase 4.5 before committing eleven hours.

---

## ADR-010. `courses.yaml` is regenerated without touching the author's edits

**Context.** The file mixes two kinds of content: measurements that go stale
after every corpus change, and decisions only the author can make - the course
titles that appear in citations, and the `include` flags that define the MVP.
Regenerating it naively would wipe the decisions; never regenerating it would
leave the measurements wrong.

**Decision.** The generator reads the existing file first. Course titles are
lifted as **raw text lines** rather than through a YAML round-trip, because
the author writes notes after them (`# ПРОВЕРЬ точное название`) and a
round-trip drops every comment. `include` values are read as YAML and written
back. Everything else is regenerated.

**Consequence.** Running the inventory twice produces byte-identical files,
and a flag the author flipped survives the next run. Both are covered by
tests, because the failure mode - silently discarding someone's work - is not
one you notice by looking at the output.

**Alternative rejected.** *`ruamel.yaml` for comment-preserving round-trips.*
A dependency, and a heavier mechanism than the one thing that needs
preserving here.

---

## ADR-011. A page number is the 1-based position of the page in the PDF file

**Context.** Citations say `с. 34`, and the evaluation matches retrieved pages
against the pages the author marks as gold. Both break unless everyone means the
same page. A book has up to three candidates: the page's position in the file,
the label the PDF may define for it (roman numerals for front matter, say), and
the number printed on the page itself - which is offset from the position
whenever a cover and a table of contents come first.

**Decision.** The canonical number is the position in the file, counting from
1 - what a viewer's "go to page" field takes. It is stored as `number` on every
page record. A label the PDF defines is stored alongside as `label`; the
printed number is not recoverable in general and is not used.

**Measured.** Of the 33 MVP files, 3 define page labels, and in none of them
does the label differ from the position at mid-document. Labels are therefore
rare here, and the printed number is the one that would mislead: CLAUDE.md now
tells the author to take gold pages from the viewer's page field.

**Alternatives rejected.** *Printed page numbers* - only a guess could extract
them, from whatever text happens to sit in the running header. *PDF labels* -
absent from 30 of 33 files.

---

## ADR-012. PDFs are parsed by pymupdf4llm's layout model, with two repairs

**Context.** `pymupdf4llm` 1.28 ships two renderers: a layout path, where an
ONNX model bundled in the wheel classifies every region of a page, and the
older path, which infers headings from font sizes or from the PDF outline. They
were compared on the same 25 mid-document pages of three MVP books (operating
systems, probability, calculus), and on two pages of C listings:

| | older renderer | layout model |
|---|---|---|
| headings found | 1, 0, 0 | 4, 15, 5 |
| running headers | left in the text | dropped |
| paragraphs | broken into short lines | whole |
| display formulas | one symbol per line, with `��` | dropped entirely |
| code listings | fenced, line breaks kept | flattened into one 651-char line |
| speed, warm | 0.125 s/page | 0.107 s/page |

Neither is usable as it stands: one shreds formulas, the other destroys code,
which CLAUDE.md forbids.

**Decision.** The layout model, repaired using the character offset (`pos`) it
reports for every region of its own output:

* a `formula` region becomes the marker `[формула]` where the formula stood;
* a `text` region whose characters are at least 60% monospaced is re-extracted
  line by line into a fenced block, with indentation recovered from each line's
  x-coordinate and the width of one character on that listing.

On the two listing pages this keeps exactly as many `;`-terminated lines as the
older renderer (8 and 18) and restores indentation, which the older renderer
lost as well. On a probability page it leaves three `[формула]` markers and no
false listings.

**Why a marker and not the formula's text.** The text layer of a display
formula is the shredded form above: no retriever can match it and no model can
read it. The marker keeps the sentence around it intact and lets an answer point
at the page - "the formula is on p. 82" - which is what a reader needs anyway.

**Also decided.** `use_ocr=False`: the layout path OCRs pages it finds empty
by default, in English. OCR is phase 4.5.

**Formula regions that are really prose (revision 3).** Comparing output with
each page's text layer (ADR-018) showed 84 pages with formulas keeping under
30% of their text, some none. The layout model had labelled whole paragraphs
- prose with inline maths, a definition, a theorem - as formulas; one such
region covered 52% of a page and held 50 Russian words, where genuine display
formulas hold at most a handful ("при", "где"). A formula region with at least
8 words of four or more letters is now kept as text, exactly as the PDF has
it, lines joined. 373 regions across the MVP turned out to be prose; 91 000
non-space characters came back; pages under 30% fell from 84 to 30, and the
1st percentile of text kept on formula pages rose from 19% to 32%.

The threshold is a trade-off, not a truth: a short sentence of five words set
inside a formula region is still replaced by the marker. The worst remaining
page keeps 27% of its text, beside three regions of five words each.

**Known limitation, measured.** The monospace test also fires on stray glyphs:
the three MVP maths courses (4 238 pages) produced 11 "listings", every one a
single line of at most fourteen letters - a symbol set in a typewriter font. The
operating-systems books show why no size threshold fixes this: of their 4 254
listings, 1 817 are single lines, median 23 characters, many of them genuine
prototypes like `pid_t fork(void);`. Any cut-off that removed the false 11
would unfence hundreds of real ones. They are left as they are: a one-line
block is harmless to retrieval and cannot be split by the chunker.

**Performance, measured.** The first full run took 30 minutes, not the 15
expected, and four books accounted for most of it at 0.8-1.4 s per page. The
repair was the cause, not the layout model: `get_text("dict")` returns every
image's pixels by default, and those four are scans with a full-page picture
(1772x2598 px and similar), so each region check decoded the whole page - 80 to
150 ms, up to 17 times per page. Extracting without image blocks
(`TEXTFLAGS_DICT & ~TEXT_PRESERVE_IMAGES`) costs 6 ms and returns identical
lines, since image blocks carry none; the repair step fell from 1.35 to 0.013
s per page. That the output is unchanged was checked on the whole corpus, not
a sample: every `raw.jsonl` was saved, everything re-parsed with `--force`,
and the files compared byte for byte.

**Consequence.** `pymupdf4llm` moves to the default dependencies: parsing is
core, and its tests must run in CI. The layout models (about 50 MB of ONNX)
ship inside the wheel and make no network calls, so the tests stay offline.

---

## ADR-013. The parse cache is keyed by content hash and parser version; cleaning is never cached

**Context.** Parsing the MVP takes minutes; re-running it on every change to
the cleaning rules would make those rules expensive to iterate on. CLAUDE.md
asks for a cache keyed by the sha256 of the source.

**Decision.** `data/parsed/<sha256>/` holds `raw.jsonl` - the parser's output -
and `meta.json`, which records the parser key (`pymupdf4llm-layout/<version>/
r<PARSER_REVISION>`). A folder is trusted only if the key matches. Cleaning
reads `raw.jsonl` and is re-done on every run, writing `pages.jsonl` (for phase
1c) and `document.md` (for a person to read).

**Details that matter.**
* The key covers the library version and our own revision number, so an upgrade
  or a change to the repairs re-parses once, automatically.
* Files are written to a temporary name and renamed, and `meta.json` is written
  last: an interrupted run leaves a folder without `meta.json`, which the next
  run parses again rather than trusting half a file. A test covers this.
* Keyed by content, a duplicate copy is never parsed twice, and moving a file
  does not invalidate its cache.

**Alternative rejected.** *Caching the cleaned output too.* It would need its
own invalidation, and cleaning 10 000 pages takes seconds.

---

## ADR-014. Two parse processes, not four

**Context.** Parsing is CPU-bound and files are independent, so it runs in a
process pool. The obvious size - one process per spare core - was measured,
not assumed, on the 718 pages of the probability course:

| processes | wall time | CPU time |
|---|---|---|
| 1 | 64 s | 191 s |
| 2 | **49 s** | 266 s |
| 4 | 55 s | 362 s |

A single process already keeps three cores busy: the ONNX runtime inside the
layout model runs each page on its own thread pool. Four processes fight over
eight cores and finish later while burning twice the CPU.

**Decision.** `PARSE_WORKERS=2`. Documents are submitted longest first, so the
biggest book does not end up running alone while the other process idles.

**Alternative rejected.** *Capping the ONNX runtime's threads per process.* The
session is created inside `pymupdf.layout` with no way to pass options through
`pymupdf4llm`; patching it would break on the next release of either library.

---

## ADR-015. The running-header safety net only touches the outermost line, on most pages

**Context.** The layout model drops the running headers and footers it
recognises - 1016 of them on the probability course. Cleaning adds a second,
text-based pass for any it misses: a line recurring at a page edge. To make page
12's header match page 13's, digits are normalised away. Tests on synthetic
pages showed what that costs: "Задача 12." and "Задача 13." become the same
line too, and with two edge lines per page, every line of a short page counts
as an edge.

**Decision.** Only the first and the last non-empty line of a page are
candidates, and a candidate must recur on at least half the pages - literally
"most pages", as CLAUDE.md says. Markers, table rows and code fences are never
candidates.

**Why so strict.** The risk is asymmetric. A missed header leaves a short line
of noise in one chunk; a removed line of content is information gone for good.
And the safety net rarely matters: on the probability course the most frequent
first or last line of any document appears on at most 8% of its pages, so it
removed nothing - correctly.

---

## ADR-016. The primary copy of a duplicate comes from the manifest, and include decisions follow the content

**Context.** Phase 1a picked the primary copy of a duplicate by page count and
path length. CLAUDE.md asks for the nsumedia copy to win. Page count could never
decide anything anyway - the copies are byte-identical - and the shortest path
often belongs to a loose book, the one copy with no provenance.

**Decision.** The primary is the copy the manifest lists, then the earliest
semester, then the shortest path. `mark_duplicates` therefore runs after the
manifest is reconciled.

**The consequence that needed its own decision.** Fixing the rule moved one
book's primary from `matan/sem1` (a loose copy) to `matan/sem2` (the manifest
copy). `courses.yaml` had recorded `include: true` on the old path and `false`
on the new one - duplicates are always written as `false` - so a naive reading
would have silently dropped the book from the MVP.

An include decision is therefore read across every copy of the same content:
since copies are always written as `false`, a `true` on any of them is the
author's. Both the regeneration of `courses.yaml` and the parse stage's
selection use this rule. After the fix the MVP held the same 33 documents,
with that one attributed to its manifest copy. A test replays exactly this
sequence.

---

## ADR-017. Where a text layer came from is recorded: born-digital or somebody else's OCR

**Context.** Investigating the slow books (ADR-012) showed that some files the
inventory calls `good` are scans: a full-page picture with an invisible OCR
text layer on top, added by whoever scanned the book. Their text is dense - so
the character-density test passes - but it carries that OCR's errors. CLAUDE.md
asks phase 2 to break retrieval metrics down by source type, "text / OCR";
without this label those books would silently count as text.

**Decision.** The inventory checks each sampled page for a single picture
covering at least 90% of it. A document where at least half the sampled pages
are like that has `text_origin = ocr-layer`; otherwise `digital`. The label is
shown in the inventory report and in `courses.yaml` (`origin:`), and copied
onto every parsed page (`origin` in `pages.jsonl`), so that chunks inherit it.

**Cheap by construction.** `page.get_image_info()` reports where images are
drawn without decoding them, so the check adds nothing noticeable to the
inventory's eleven seconds.

**Measured.** In the MVP, **4 of 33 documents are scans under an OCR layer -
2 510 of 10 606 pages, almost a quarter**: "Антидемидович", Тер-Крикоров and
Шабунин's calculus course, Самарский and Гулин's numerical methods, and
Иртегов's "Введение в операционные системы". Across the corpus, 9 of the 97
files with a good text layer are like this (5 023 pages). Without the label, a
quarter of the MVP would have been counted as clean text in phase 2.

These are the same four books that made the first parse slow (ADR-012): their
pictures run to 4-10 megapixels a page.

**Alternative rejected.** *Deciding per page.* More precise for a book that
mixes scanned and born-digital pages, but it needs the check inside the parser
and a full re-parse; phase 2 compares source types, and a document-level label
answers that.

---

## ADR-018. Parse quality is judged against each page's own text layer

**Context.** The first quality report flagged 86 of 10 606 pages, mostly by
comparing a page's length with its document's median. Triaged by hand, almost
none were parser failures: blank versos before a chapter (the PDF's text layer
holds only a page number), 1 480 check marks from one book's support tables
counted as junk, covers set in fonts with no Unicode mapping, and short pages
carrying four or five `U+FFFD`. A list that cries wolf trains its reader to
skip it. And the median comparison answered the wrong question: whether the
page is short, not whether the parser lost anything.

**Decision.** The parser records `source_chars` - the non-space characters in
the page's own text layer - and the report compares the output with it:

* **text lost**: the PDF has at least 500 characters on the page, fewer than
  30% came out, and no formula or picture explains the gap - review;
* **blank in the PDF itself**: next to nothing in the text layer either -
  expected, not flagged;
* **junk**: more than 2% broken glyphs *and* at least 20 of them - review,
  unless it is a cover or an endpaper.

**Result.** 5 pages to review out of 10 606, 125 explained (67 blank in the
PDF, 50 dominated by formulas, 4 covers, 4 figures). On the 6 576 pages
without formulas the output keeps a median of 100% of the text layer, and 96%
at the 1st percentile: the parser loses no prose.

**The same comparison found a real loss** on pages with formulas: 84 kept under
30% of their text, some none at all. The layout model had labelled whole
paragraphs of prose with inline maths as formulas - see ADR-012.

---

## ADR-019. Private-use glyphs are mapped only when their font is known

**Context.** 1 721 characters in the MVP output were private-use code points.
Symbol fonts place their glyphs at `U+F0NN` for byte `0xNN`, so the meaning
depends on the font: `U+F0FC` is a check mark in Wingdings and a piece of a
brace in Symbol. 86% of them were one glyph - Wingdings' check mark, in the
function-support tables of Rago and Stevens's UNIX book, where each one means
"supported". Left as they were, those tables said nothing.

**Decision.** At parse time, while the font is still known, a page's
private-use characters are mapped through a table of glyphs *measured in this
corpus*: Wingdings `U+F0FC` → ✓, and in Symbol-encoded fonts (SymbolMT,
EuclidSymbol) `U+F0AE` → → and `U+F0B7` → •. A character set in a font the
table does not know, or in two fonts that disagree on the same page, is left
alone. `MT-Extra` uses its own encoding and was deliberately not guessed at.

**Result.** 1 481 check marks restored; private-use characters down from 1 721
to 189 - pieces of large brackets inside formulas, which have no meaning of
their own.

**Alternative rejected.** *A complete Wingdings/Symbol table applied during
cleaning.* Cleaning no longer knows the font, so it would map every `U+F0FC`
to a check mark - including the brace pieces.

---

## ADR-020. Chunks follow headings where the text allows, and never outgrow the model

**Context.** CLAUDE.md asks for chunks cut at headings, long sections cut
between paragraphs up to ~400 tokens with ~50 of overlap, and listings and
tables never cut through the middle. The parsed MVP, measured with the
embedding model's own tokenizer, does not cooperate with a literal reading:

* **Heading levels mean nothing across books.** Bratko's Prolog and Stevens's
  network programming use level 1 only (124 and 382 headings); Williams's C++
  book puts 1 605 run-in labels - "Объявление", "Результат" - at level 6. A
  rule like "levels 1-3 are boundaries" works for none of them.
* **Sections are tiny or huge.** 7 059 sections: median 383 tokens, but 18%
  are under 50 tokens, and one runs to 51 637 with no heading at all.
* **Listings and tables outgrow any chunk.** 76 of 4 265 listings and 257 of
  694 tables exceed 450 tokens; one table is 5 656. multilingual-e5-base
  reads 512 tokens and silently truncates the rest.
* **28% of pages start mid-sentence.**

**Decision.**
1. A heading starts a new chunk only once the current one holds at least 100
   tokens; smaller sections are packed together, and the chunk is labelled
   with the *common ancestor* of the sections it holds.
2. Prose is cut between paragraphs, then between sentences, into pieces of at
   most budget − overlap, so that the carried overlap always fits.
3. A listing or table that fits goes whole. One that cannot is split along its
   structure: a listing at blank lines between its parts, else between lines;
   a table between rows, with its header repeated - unless the header alone
   is over half the budget. Never inside a line or a row, except as a last
   resort for text with no structure left (a 629-character table cell without
   a space).
4. A paragraph broken by a page break is joined back, across the two pages.
5. Overlap is an exact suffix of the previous chunk: its closing sentences up
   to ~50 tokens, or - when the last sentence alone is longer - its closing
   words, marked "…". It never opens the next section, and a lone formula
   marker is not carried.
6. What is embedded is `course › document › section` plus the text. The text
   budget leaves room for a context line capped at 64 tokens, so every chunk
   fits the model's window by construction.

**Result.** 20 190 chunks; median 329-365 tokens per course; 16 under 50
tokens. What the model reads - prefix, context line, text - peaks at 487
tokens, p99 462: **none of the 20 190 is truncated**. Overlap is present at
96% of the 12 512 boundaries inside a section that follow prose; of the 469
without it, 335 are by design - the next chunk opens with a listing or table,
or the previous one ends on a bare marker - and 134 (1%) are unexplained.

**Found by measuring, not by reading the code.** Two bugs cost overlap at most
boundaries: prose was cut to the full budget, leaving no room for the carried
text; and every heading discarded the carried text, even a small one that did
not start a chunk (2 559 boundaries). Along the way: the layout model
sometimes labels an 884-character bold paragraph as a heading, so a
"heading" over 150 characters is read as text. A first measurement of overlap
also undercounted, because re-joined sentences no longer matched the original
line breaks - which is why the overlap is now an exact suffix.

**Alternatives rejected.**
* *Fixed-size windows.* Simple, but cut sentences, listings and table rows.
* *Headings as hard boundaries.* 18% of chunks would be a line or two.
* *Semantic chunking by embedding similarity.* It would need the model at
  chunking time and makes boundaries hard to explain - and to test.

---

## ADR-021. Indexing is incremental because chunk ids come from content

**Context.** Embedding the MVP takes about twenty minutes; the full corpus
will take hours. Re-embedding everything after every change to parsing or
chunking would make iterating on either impractical. CLAUDE.md asks for
incremental indexing of changed chunks only.

**Decision.** A chunk's id is a UUID derived from the embedding model's name,
the sha256 of its document, its first page and the exact text that is
embedded. Syncing is then set arithmetic against the ids already in Qdrant:
embed the new ones, delete the missing ones, leave the rest untouched - not
re-embedded, not even read. The model is loaded only if something is new.

**Consequences.**
* An edit re-embeds only the chunks it touched; changing the model or a
  course title changes the ids it affects, so those are rebuilt.
* Excluding a document in `courses.yaml` deletes its points on the next run.
* A run with `--course` compares and deletes only that course's points, and
  rewrites only that course's lines in `data/chunks.jsonl`. Without that
  scope, a single-course run would have treated every other course as stale -
  both traps were caught before the first real run and are covered by tests.

**Verified on the real index.** First run: 20 190 points added. Second run:
0 added, 0 deleted, the model never loaded, 76 s instead of 22 min. A run over
one course: 0 changes, and the collection still holds all 20 190 points.

**Embeddings.** multilingual-e5-base with its `passage: ` prefix, normalised,
cosine distance. Measured on MPS on 512 real passages: batch 32 gives 24.1
chunks/s, 64 gives 22.9, 128 gives 17.9; float16 is 9% faster and moves vectors
by up to 1 − 0.99954 in cosine - not worth it, so float32 and batch 32.

**What the benchmark missed.** The real run averaged 16.1 chunks/s, not 24.
Writing to Qdrant costs 0.16 s per batch of 256 - 2%. The rest is heat: this
MacBook Air has no fan, and three consecutive ten-second runs already fall
25.4 → 24.5 → 23.1 chunks/s. Sustained, it settles near 16. For planning the
full corpus (phase 4.5), the sustained rate is the one that counts: at the
MVP's 1.9 chunks per page, 59 630 pages make about 113 000 chunks, which is
about two hours - not the 93 minutes phase 1a extrapolated from a burst.

**Alternatives rejected.**
* *Ids from position (document + chunk number).* Inserting one chunk early in
  a document would shift every id after it and re-embed the rest.
* *Rebuilding the collection on every run.* Correct, and twenty minutes each.

---

## ADR-022. BM25 is written out, and stems each word in its own language

**Context.** CLAUDE.md asks for BM25 with Snowball stemming "by the language of
the fragment". The fragments of this corpus are rarely one language: a Russian
paragraph about `fork()`, a C listing with Russian comments. A chunk-level
language flag would send English identifiers through the Russian stemmer, or
the reverse.

**Decision.**
* **The language is decided per word, by its alphabet**: Cyrillic words go to
  the Russian Snowball stemmer, Latin words to the English one, anything else
  - numbers, identifiers with digits - stays as it is. Questions are tokenised
  the same way, so a word always meets its own stem.
* `ё` is folded to `е` (the corpus spells words both ways); one-character
  tokens are dropped (formula variables and one-letter prepositions, which
  match everywhere).
* The context line - "course › document › section" - is indexed with the
  text, as it is embedded with it: a section title is often the best match a
  question has.
* The scoring is Lucene's: k1 = 1.2, b = 0.75, and an idf that never goes
  negative, so a very common term adds a little instead of subtracting.

**Why write BM25 at all.** It is a dozen lines of arithmetic over numpy arrays,
and every part of it is explainable at an interview: the inverted index, the
length normalisation, the idf. Measured on the MVP: 20 190 chunks and 65 764
terms indexed in 2.4 s, 0.8 ms per query. At that cost the index is rebuilt
from `data/chunks.jsonl` on every start rather than cached - there is nothing
to invalidate.

**Alternatives rejected.**
* *`rank_bm25`.* Scores every document per query in Python, and hides the
  tokenisation this corpus needs control over.
* *Qdrant sparse vectors with a BM25 model.* A second model, and its own
  stemming instead of the Snowball stemmers CLAUDE.md names.
* *Elasticsearch or OpenSearch.* A server to run for 20 000 chunks - against
  "не усложнять".

---

## ADR-023. Hybrid search fuses by rank; the reranker re-reads a shortlist

**Context.** Four modes are required: BM25, dense, their fusion, and the fusion
re-ordered by the `bge-reranker-v2-m3` cross-encoder, which must be possible
to switch off.

**Decision.**
* **Fusion by Reciprocal Rank Fusion**, k = 60 (Cormack et al., 2009), over the
  top 50 of each retriever. BM25 scores and cosine similarities live on
  unrelated scales; RRF looks only at ranks, so no calibration is needed. Ties
  are broken by chunk id, so a run is reproducible.
* **The reranker re-reads the fused top 30**, passage and context line
  together, and returns the top k. A cross-encoder reads question and passage
  at once, which is why it orders a shortlist better than two independently
  computed vectors - and why it cannot search the whole corpus.
* `RERANKER_ENABLED=false` turns "hybrid+rerank" into "hybrid" everywhere, for
  a machine without the memory or the time for the heaviest model here.
* A dense search whose request fails in transport - a timeout, a dropped
  connection - is retried twice. The first full evaluation died two hours in
  on a single timed-out request: the laptop had slept with it in flight.

**Measured on an M3 MacBook Air, MPS.** Per query: BM25 0.8 ms, dense 70 ms,
hybrid 47 ms, **hybrid + rerank about 4 s** - 30 question/passage pairs through
a 568M-parameter model. float16 is *slower* on MPS (4 565 ms against 4 116,
with an identical top 10), and batches of 32 are slower than 16. The cost is
linear in the number of candidates, which is therefore the lever: phase 3
decides it against the time-to-first-token budget, once the evaluation shows
what reranking 30 candidates buys over fewer.

**How retrieval is scored** (phase 2 evaluation, `nsu-rag eval`):
* a hit is a chunk from the gold document whose pages overlap the gold pages;
  documents are matched **by content hash**, so a question naming any copy of
  a duplicate is matched by the chunks of the indexed copy;
* recall@k of a question is the share of its gold sources found in the top k
  - partial for a multi-hop question - and MRR@10 uses the first hit;
* unanswerable questions are left to phase 4, which measures refusals;
* every mode is also scored with the question's course as a filter, which
  measures what phase 5's router could gain;
* the report states a 95% interval for recall@10, because with fifty questions
  a difference of a few points between modes is noise;
* the report holds question ids and numbers only, so it can be committed.

**Alternatives rejected.**
* *Fusing scores* - min-max normalised BM25 plus cosine, weighted. The weight
  is one more parameter to tune on fifty questions, and one outlier score
  stretches a query's whole normalisation.
* *Reranking everything the retrievers return.* Up to 100 pairs through the
  cross-encoder: at the measured, linear cost, about 13 s a query.
* *An LLM as the reranker.* The same 9B model that answers would have to read
  30 passages before it starts reading them again to answer.

---

## ADR-024. The prompt is measured with the model's own tokenizer before it is sent

**Context.** The model's window is 8 192 tokens (`num_ctx` in the Modelfile,
ADR-004). A prompt that does not fit is not rejected: the server cuts it
silently, and the model answers from whatever survived. CLAUDE.md requires the
prompt builder to stay inside the budget and to log what it had to leave out.
The answer's own 1 024 tokens (`LLM_MAX_OUTPUT_TOKENS`) come out of the same
window.

**Decision.**
* **Tokens are counted with the answering model's tokenizer**, loaded from
  Hugging Face (`LLM_TOKENIZER=Qwen/Qwen3.5-9B`: the tokenizer files, about
  23 MB, never the weights). Ollama 0.34 has no tokenize endpoint
  (`/api/tokenize` answers 404), so there is nothing to ask the server.
* **Calibrated against the server.** On ten prompts built from real chunks,
  the tokenizer's count of the message texts plus 17 matched the
  `prompt_tokens` Ollama reported *exactly* on all nine Russian prompts, and
  came one short on an English one. 17 is the chat template's own tokens
  (`CHAT_OVERHEAD`); a margin of 32 more absorbs that drift and the joins
  between fragments, which a tokenizer does not count additively.
* **The budget**: 8 192 - 1 024 (answer) - 17 - 32 = **7 119 tokens** for the
  rules, the fragments and the question.
* **Fragments go in by rank while they fit.** One that does not fit is skipped
  whole - never cut mid-definition - and a shorter one further down may still
  go in; numbering stays dense, so the model never sees a gap. Every skipped
  fragment is logged with its chunk id, and its count is returned with the
  answer (`dropped`).
* **Checked on every answer.** The streamed reply ends with the server's own
  `prompt_tokens`; it is logged next to the estimate, with a warning if it
  ever exceeds the budget. On the phase 3 smoke run - ten questions, ten
  fragments each, 3 948 to 5 123 prompt tokens - the estimate was above the
  truth by exactly 10 tokens every time: the separators between fragments
  are counted as two tokens and cost one.

**Alternatives rejected.**
* *Characters divided by a constant.* Measured on 3 000 indexed chunks, the
  median characters per Qwen token is 3.9 in Russian prose, 2.8 in Russian
  with formulas, 3.6 in code and 2.4 in English. A constant safe for the
  densest kind wastes over a third of the window on prose; one tuned to
  prose overflows it on formulas by as much.
* *`tiktoken`.* OpenAI's vocabularies, not Qwen's: the counts would be
  confidently wrong.
* *Sending first and reading `prompt_tokens` afterwards.* By then the prompt
  has already been cut.
* *Trimming each fragment to fit.* A fragment is a chunk that was built to be
  self-contained (ADR-020); half of one is worse than a different whole one.

---

## ADR-025. The model cites a label; the code checks it and writes the citation

**Context.** CLAUDE.md asks for a citation after every claim, in the form
`[course · «document» · p. N]`, and counts a citation of a fragment that does
not exist as an error.

**Decision.**
* **Fragments are labelled `[Ф1]`…`[Ф10]` in the prompt**, each followed by its
  context line and pages. The model is told to cite the label; the code checks
  every label against the fragments the model was actually given and replaces
  it with the full citation. Course titles, document titles and page numbers
  are never generated, so they cannot be misspelt or invented.
* **A letter, not a bare number.** `[1]` cannot be told apart from mathematics:
  `[0, 1]` is a segment, and three of the five MVP courses are mathematical.
  The parser takes `[Ф1]`, `[Ф1, Ф3]`, `[Ф1; 3]`, `[ф2]`, `[Фрагмент 2]`, and
  leaves every other bracket alone; the tests list the cases, `[0, 1]`,
  `[a, b]`, `[формула]` and `[Рис. 2]` among them.
* **Errors are visible and counted.** A label with no fragment behind it
  appears in the answer as `[нет фрагмента Ф9]`, goes into `citation_errors`
  and the log; so does an answer - other than a refusal - with no citation at
  all. Whether *every claim* is cited is for phase 4's judge to score; a regex
  cannot tell where one claim ends.
* **Expanded while streaming.** Text is passed through as it arrives, except
  from an open bracket onwards, which is held until it closes (and is expanded
  if it is a label) or runs past 48 characters (and is prose). The reader waits
  a few tokens at most, and a test checks, at every possible split of a
  sample, that streaming produces exactly what rendering the whole text does.
* Two fragments from the same page cited together print one citation.

**Why not let the model write the full citation.** Besides the typos: a label
is 4 tokens of output, a full citation 31-36 (counted with the model's
tokenizer on synthetic titles of real length). At about 15 tokens/s, an answer
with five citations would spend some ten seconds more writing them.

**Alternatives rejected.**
* *Bare numbers, `[1]`* - ambiguous in mathematics, as above.
* *Structured output* - a JSON list of claims with source ids. No streaming,
  a new way to fail for a 9B model (invalid JSON), and the reader wants prose.
* *Attribution after the fact* - matching each sentence to its most similar
  fragment. It says which fragment *resembles* a sentence, not which one the
  model used, and it is a second retrieval problem to evaluate.

---

## ADR-026. Ten fragments per answer; no fragments, no model

**Context.** How many fragments the model reads, from which retrieval, and
what it says when the handbooks do not answer. Recall wants more fragments;
the wait for the first word wants fewer, because the model reads the whole
prompt before it writes anything.

**Decision.**
* **Ten fragments** (`ANSWER_FRAGMENTS=10`), which the reranker picks from the
  hybrid top 30 (`RETRIEVAL_MODE=hybrid+rerank`, ADR-023).
  * Ten is the depth phase 2 measures recall at: recall@10 is the chance that
    the answer is in front of the model at all.
  * Reading time, measured on the M3 Air with the model loaded and memory to
    spare: 2 fragments (782 prompt tokens) 5.1 s to the first token, 6 (1 912)
    9.8 s, 10 (3 124) 11.0 s, 14 (4 488) 12.9 s. Going from six to ten costs
    1.2 s.
  * Short of memory, it is a different machine. The phase 3 smoke run - ten
    questions, with Xcode and its debugger, a messenger and a browser open,
    12-17 GB of swap in use - read the same ten fragments (3 900-5 100
    tokens) in 29-47 s, at 98-149 tokens/s. The reranker took 15-19 s where
    on its own, the same hour, it takes 4.5 s - its 2.1 GB of weights had been
    paged out while the model wrote - and dense search 2-6 s instead of 0.1:
    the first word came after 49-77 s. With `RERANKER_ENABLED=false`
    (four of the questions) search took 0.1-0.5 s and the first word came
    after 27-71 s. Reading the prompt is the largest part either way.
  * It is a default for phase 4 to test against answer quality, not a result.
* **Refusal.** The model is told to reply with exactly «В методичках этого
  нет.», recognised whatever the quotes, markup or case, and whatever follows.
  If search returns nothing - a course filter with no match, or no fragment
  fits the budget - the service refuses without asking the model: there would
  be nothing to cite, so an answer could only come from the model's memory.
* **Deterministic.** Thinking off (ADR-003) and temperature 0 (ADR-002): the
  same question gets the same answer, which phase 4's comparisons rely on.

**Alternatives rejected.**
* *Five fragments* - seconds faster, but every fragment left out is a chance
  that the answer never reaches the model; recall@5 against recall@10 in
  phase 2's report will say what that costs.
* *Filling the budget* - about 17 fragments fit in 7 119 tokens, but the
  reranker's seventeenth is rarely relevant, and each one adds reading time
  and gives a 9B model more to lose its way in.
* *A threshold on the reranker's score*, refusing below it without asking the
  model. That is phase 5's agent; the threshold needs labelled questions to be
  set, and there are none yet.

---

## ADR-027. One pipeline behind the CLI, `/ask` and a Server-Sent Events stream

**Context.** Phase 3 asks for `POST /ask` (JSON), `POST /ask/stream` (SSE) and
`nsu-rag ask`, with latency logged per stage: search, rerank, first token,
whole answer.

**Decision.**
* **The pipeline is one async generator of events**: `sources`, then `token`
  after `token`, then `done`. The stream forwards them as they come; `/ask` and
  the CLI wait for `done`. One code path, tested once.
* **Sources go first**, before the model starts: reading the prompt takes the
  model ten seconds and more on this laptop, and a client can show meanwhile
  which pages are being read.
* **FastAPI's own SSE support** (`EventSourceResponse`, `ServerSentEvent`, in
  the installed FastAPI 0.141): no `sse-starlette`, and keep-alive pings come
  with it. Each event's JSON is written with `ensure_ascii=False`, so
  `curl -N` shows Russian rather than `\u` escapes.
* **POST, not GET**: the question is a body. The browser's `EventSource` can
  only GET; a web client reads this stream with `fetch` instead.
* **Checked before the stream starts.** Once a streaming endpoint runs, its
  `200` is on the wire, so the checks live in one dependency: the body is
  validated, then the pipeline is loaded, then the course is checked - `422`,
  listing the indexed courses. The order took a fix: FastAPI calls a
  dependency's own dependencies *before* validating the body, so with the
  pipeline as a sub-dependency an empty question waited 20 s for the models,
  and would have got a `503` had they failed to load. The loading is now a
  call inside the dependency, which FastAPI makes only for a valid body; a
  test pins it down. A failure after the `200`, such as the LLM going away
  mid-answer, ends the stream with an `error` event. On `/ask` an LLM failure
  is a `502`: something upstream failed, not this service.
* **Loaded by the first question**, not at start-up: the models take some 20 s
  to load, `/health` must answer at once, and a container without the ML
  extras must still start. A failed load is a `503`, retried by the next
  question. The load ends with a warm-up search, so the first real question
  is not charged for kernel compilation.
* **Search runs in a worker thread, one question at a time.** Search and
  rerank are synchronous and heavy; `asyncio.to_thread` keeps the event loop
  answering. A lock serialises them: the embedder, the cross-encoder and
  their fast tokenizers are not meant to be called from two threads at once.
  Generation is not locked; Ollama schedules concurrent requests itself.
* **Timings.** `search_ms` and `rerank_ms` - how long each stage took;
  `llm_first_token_ms` - from the request to the model to its first token,
  i.e. reading the prompt; `first_token_ms` and `total_ms` - from the
  question's arrival, what a reader actually waits. One INFO line per answer
  adds prompt tokens (estimated and reported), answer tokens and tokens/s.
* **A reader who leaves stops the model.** The model's stream is closed the
  moment its reader stops listening (`contextlib.aclosing` - without it the
  connection lives on until the garbage collector finalises the generator; a
  test tells the two apart), and Ollama then cancels the generation. Checked
  on Ollama 0.34.3: a stream closed after 20 chunks of a 1 000-token request
  is logged as a cancelled task, stopped at 60 tokens, prompt included.

**Alternatives rejected.**
* *WebSocket* - two-way, which a question and its answer do not need.
* *Loading the models at start-up* - `/health` would be silent for 20 s, and
  a broken index would stop the service from starting instead of being
  reported.
* *A task queue with polling* - against "не усложнять", and the tokens would
  still need a stream.

---

## ADR-028. The judge rules on facts it can check, in JSON, twice per answer

**Context.** Phase 4 scores answers with an LLM judge on its own endpoint
(`JUDGE_*`): faithfulness, completeness against the author's reference answer,
correct citations, refusals on unanswerable questions - and calibrates the
judge against the author's own labels with agreement and Cohen's kappa.

**Decision.**
* **Two questions per answer, two calls.** Faithfulness and citations need the
  fragments - labelled exactly as the answering model saw them - and the raw
  answer, labels included (rendered citations are ambiguous: two fragments of
  one page print the same). Completeness needs only the question, the
  reference and the answer, a far shorter prompt.
* **Binary verdicts, like the author's labels**, so that kappa can be
  computed between the two.
* **JSON against a schema, lists before verdicts.** The judge fills
  `unsupported`, `miscited`, `uncited` (or `missing`) and only then
  `faithful` / `citations_correct` (or `complete`): a small model decides
  better once it has written down what it found, and the lists show a reader
  *why*. Ollama honours `response_format` with a JSON schema - checked live on
  0.34.3. A reply that does not parse is a **judge failure**, counted and
  reported, never read as a verdict; a verdict contradicting its own lists is
  counted too.
* **What code can check, code checks.** Refusals are recognised by the
  refusal phrase, not by the judge, and never sent to it; citations of
  fragments that do not exist are counted from the answer itself
  ("citations valid"); the judge rules only on whether a real citation
  supports its claim ("citations supported").
* **Where a refusal counts.** Against completeness on an answerable question,
  for refusal accuracy on an unanswerable one; faithfulness is measured over
  the answers actually given. A metric over no answers is "—", never 0.
* **Answers stay out of git.** They quote the handbooks, so each run keeps
  them in `data/eval-runs/<date>-<mode>-<model>/`; the report in
  `eval/reports/` has ids and numbers only.
* **Calibration** reads the author's `eval/manual_labels.jsonl` (id, run,
  faithful, complete) against the judge's verdicts on the same answers and
  reports agreement and kappa for each verdict separately - a judge can be
  good at one and useless at the other.

**Checked live** on a synthetic fragment and two answers: the judge listed
the invented claim as unsupported and the answer as unfaithful, passed the
faithful one, and named the point missing from the reference - 2-14 s per
verdict with a short prompt. One definition was adjusted after watching it:
a claim with a label but no support is a citation error too, and the judge
records it under `unsupported` only, so that combination is not counted as a
contradiction.

**Known bias.** The default judge, `qwen3.5-9b-8k`, is also the default
answering model, and models favour their own writing. A judge from another
family - `gemma4-12b-8k`, or a stronger model behind an API - is one setting
away (`JUDGE_MODEL`, `JUDGE_BASE_URL`), and calibration is what shows which
to trust.

**Alternatives rejected.**
* *A 1-5 score per answer.* A 9B judge's scores are poorly calibrated, and
  they cannot be compared with the author's yes/no labels.
* *One judge call per claim* (FActScore-style decomposition). More precise,
  but several times the calls - with some fifty questions and four models the
  judging would take most of a day on this laptop - for a verdict the lists
  already approximate.
* *RAGAS or DeepEval.* Their prompts and parsing are hidden behind the
  library, their faithfulness is that same per-claim decomposition, and every
  line here has to be explainable.

---

## ADR-029. Models are compared alone, on the same fragments, by a rule fixed in advance

**Context.** Phase 4 compares the answering models - `qwen3.5:4b`,
`qwen3.5:9b` with and without thinking, `gemma4:12b`, each with an 8k window -
on quality (faithfulness, completeness, citations, refusals) and on cost
(time to first token, tokens per second, peak memory), and picks the default.

**Decision - how they are compared.**
* **Same fragments.** Every question is searched once; every model answers
  from those fragments. A difference between two rows is the model's.
* **One model in memory at a time.** The search models are released before
  the first model loads, and every model starts with the others unloaded
  (`keep_alive: 0`): on 16 GB, co-resident models page each other out and
  the slowdown lands on whoever runs second (ADR-026).
* **Speed from the model's side.** TTFT runs from the request to the first
  token of the answer; tokens per second from the first output of any kind,
  because thinking tokens are completion tokens too - Ollama streams them in
  a separate `reasoning` field, checked live. Peak memory is the largest size
  `/api/ps` reports while the model answers.
* **Each model's own tokenizer** fits its prompt (ADR-024); the estimate is
  checked against the size the server reports on every answer.
* **Thinking gets room to think**: `max_output_tokens` 3072 for that variant,
  which leaves some 5 000 tokens for fragments instead of 7 100.
* **The judge runs after all generation**, so its model never shares memory
  with one being timed.

**The rule, fixed before any quality number exists.** Take the most faithful
model; any model whose faithfulness is within the 95% interval of the best
counts as equally faithful; among those, prefer the higher completeness, then
the lower TTFT. Thinking is worth its latency only if it lifts faithfulness
or completeness beyond that interval. Fixing the rule first keeps the choice
from being fitted to whatever the numbers turn out to be.

**Measured - cost, on ad-hoc questions, not the evaluation set.** Quality
waits for the author's questions; cost does not need them. Two runs, each on
the same fragments for every model, `--no-judge`:

| model | TTFT | tokens/s | peak memory | answers |
|---|---:|---:|---:|---|
| *run 1 - morning, machine fresh, 10 questions* | | | | |
| `qwen3.5-4b` | 18.3 s | 22.6 | 3.25 GB | 10 of 10 |
| *run 2 - hours of load, battery at 10-25% on a weak charger, 3 questions* | | | | |
| `gemma4-12b` | 79.1 s | 4.0 | 7.83 GB | 3 of 3 |
| `qwen3.5-9b` with thinking | - | - | 5.62 GB | **0 of 3** |
| `qwen3.5-9b` | 50.5 s | 6.1 | 5.62 GB | 3 of 3 |
| `qwen3.5-4b` | 31.3 s | 10.1 | 3.25 GB | 3 of 3 |

* **The same model ran twice as slow in the second run** - `qwen3.5-4b`,
  18.3 s against 31.3 s to the first token. A fanless laptop after hours of
  load, and a battery the system was protecting, are part of any measurement
  here; only rows of one run compare - which is why the second run measured
  every model under the same conditions, the morning's model last.
* **Thinking answered nothing.** On all three questions `qwen3.5-9b` spent the
  whole 3 072-token budget thinking and never started the answer - seven
  minutes each. A larger budget would leave too little of the 8k window for
  fragments: thinking and this design do not fit together.
* **`gemma4-12b` costs 2.5 times `qwen3.5-4b`'s time** and 7.8 GB - half the
  machine - before the search models and anything else.
* **Every answer that came was cited correctly** (no citation of a fragment
  that does not exist), and the prompt estimate was 9-12 tokens above the
  server's count for every model: the budget holds for the Gemma tokenizer
  and chat template too, not just Qwen's.

**Decision - the default, for now.** A cost ceiling comes before the quality
rule: an answer that starts after a minute, or never, is not a service on
this laptop. That rules out thinking and `gemma4-12b`. Between `qwen3.5-4b`
and `qwen3.5-9b` the rule above decides once the author's questions exist;
until then the default stays `qwen3.5-9b-8k`, the model every calibration so
far was made with. If the 4B model's faithfulness falls within the 9B's
interval, it wins: 1.6 times faster to the first word and 2.4 GB lighter.

**Alternatives rejected.**
* *Each model with its own retrieval run.* The same search, repeated, costs
  4-15 s per question per model and adds nothing: retrieval is deterministic.
* *All models loaded together.* They do not fit in 16 GB; the measurements
  would be of swapping.
* *Choosing by speed now, quality later.* Speed is cheap to measure and
  tempting to decide on, but a fast model that invents is worse than useless
  here: the whole service is its citations.

---

## ADR-030. The full corpus is a second index built by the same pipeline

**Context.** Phase 4.5 brings in everything the MVP left out: scans that need
OCR, the remaining courses, English documents - and asks whether recall and
faithfulness change when the index holds ten times more text, and what to do
with the giant references that could crowd the rest out.

**Decision - the second index.**
* **A corpus scope, not a second codebase.** `CORPUS_SCOPE=full` selects
  every unique document - the include flags in `courses.yaml` describe the
  MVP, and the full corpus is by definition what they left out - and
  `corpus.full.env` points the chunks file and the Qdrant collection
  elsewhere. The MVP index stays untouched, so the same questions can be put
  to both; reports for the full index are named `<date>-retrieval-full.md`.
* **The parse cache is shared**, keyed by content: a document parsed for the
  MVP is not parsed again.
* **OCR is a command of its own**, `nsu-rag ocr`, because it takes hours. It
  recognises one page at a time and appends each to the document's cache the
  moment it is done; a restart skips the pages already there, and a page the
  engine fails on is recorded as empty and counted instead of stopping the
  run. It logs progress with an estimate of the time left, and `--status`
  answers "how far has it got" without disturbing it.
* **Whole-page OCR** (`OcrMode.FULL_PAGE`) for every scan: these documents
  have no text layer, or one the inventory rejected.
* **Text from our OCR is marked `origin: ocr`**, apart from `ocr-layer` (a
  scan someone else recognised) and `digital`, so the retrieval report breaks
  recall down by where the text came from.

**Decision - the OCR engine: Apple's Vision**, through docling's `ocrmac`
(`OCR_ENGINE=ocrmac`), which ADR-009 left to be measured. On the same nine
random scan pages, after a warm-up page:

| engine | seconds per page | characters per page |
|---|---:|---:|
| Vision (`ocrmac`), Russian + English | 0.88 (0.72-1.25) | 1 166 |
| RapidOCR, Russian | 1.63 (0.97-2.56) | 1 014 |

Read side by side, RapidOCR's Russian-only model put Latin letters into
Russian words, got the case of SQL keywords wrong and lost stretches of
mathematical prose; Vision reads Russian and English in one pass - the
problem ADR-009 left open for English scans - and kept both. The run itself:
47 scanned documents, 20 040 pages, 7.9 hours, not one page the engine failed
on. That is 0.8-0.9 s a page with the machine to itself and several seconds
while the embedding model shared the GPU; phase 1a had estimated eleven hours
with RapidOCR. docling's own markers for what it could not read are
replaced with the parser's `[формула]` and `[рисунок]`, which the prompt
explains to the model and the parse report counts.

**Decision - the giant references stay, at full weight.** The full corpus is
145 unique documents and 59 630 pages in 22 courses; 24 documents (17 394
pages) are in English; indexed, it is 130 483 chunks. One document is 8% of
all the pages - Intel's x86 manual, 4 778 pages - and with the other
architecture documents it makes 18% of the chunks; next come the C++20 draft
standard (1 815 pages) and a dozen textbooks of 1 000-1 700. Measured before
deciding, without any author's question - top ten of the MVP index against
the full one, for the same ad-hoc questions:

* **Questions on the MVP's courses** (ten): the top ten stayed 70% the same
  (median). Most of the rest are the scanned textbooks of the same courses,
  recognised now - before the scans were in, 90% had stayed - and the best
  reranker score held or rose. Other courses took 15 places in 100 -
  "virtual memory" also found the architecture books, the expectation of a
  random variable an algorithms book's probability appendix, rightly - and
  **the giant references took none**.
* **Questions only the full corpus can answer** (five): databases, C++ and
  digital circuits found their own course in 7-10 places of 10; control
  theory - four documents, 200 pages - lost to electrical engineering, which
  covers transfer functions too.
* **The giants are not crowding anything out; they are hard to reach.**
  Search stays in the language of the question: a Russian question on x86
  addressing modes got no English fragment in its top ten, the same question
  in English got ten; a Russian question naming an instruction (`CMPXCHG`)
  got six from the manual, its English twin ten.

So nothing is dropped or down-weighted: the manuals take no room from other
answers, and they are the only source when a question names an instruction or
an identifier. What they show is the opposite problem - English documents are
out of reach of Russian questions - and its fix belongs to retrieval, not to
the corpus: a second search with the question translated, fused like BM25
and dense are. That is next, not now: measuring it needs the author's
questions.

**What the full corpus taught, on the way.**
* **Two documents' text layers held control characters.** Fonts with no
  Unicode mapping filled 209 of their pages with garbage, U+0085 ("next
  line") among it - the parse report flags every one of those pages.
  `str.splitlines()` breaks lines at U+0085, U+2028 and U+2029, and
  `json.dumps(ensure_ascii=False)` leaves them unescaped, so a page's JSON
  line was cut mid-string and the whole parse stopped. Every JSON Lines
  reader now splits at `\n` alone (`nsu_rag_llm/jsonl.py`) - a bug the MVP's
  documents had simply never triggered.
* **Indexing died at 71%** on one write that timed out while Qdrant, in a VM
  short of memory, built its HNSW graph for 74 000 points. Writes are
  idempotent - ids come from content - so they are now retried like searches
  (ADR-023), and the second run embedded only the 31 000 chunks missing.
* **Handwritten notes do not OCR.** Vision reads print; one set of
  handwritten lecture notes came out at a median of 316 characters a page,
  with dozens of pages empty. The parse report lists them for review; they
  stay in the index for whatever was read.
* **OCR and embedding share the machine badly.** Vision slowed from under a
  second a page to several while the embedding model ran on the GPU; the two
  long jobs are best run one after the other.

**Alternatives rejected.**
* *One index, with a payload flag for the MVP subset.* The comparison needs
  the MVP's BM25 statistics too, and they would be computed over the whole
  corpus.
* *OCR inside `nsu-rag parse`.* A command that sometimes takes a minute and
  sometimes a night is a command nobody dares to run.
* *Changing the author's include flags to widen the corpus.* They are the
  author's decisions about the MVP (ADR-016), and the MVP index must stay
  reproducible.

---

## ADR-031. The agent is a small graph of plain functions; LangGraph holds the edges

**Context.** Phase 5: a router picks the course from the question and filters
the search; when the reranked results are weak, the query is rewritten and
searched again, and otherwise the agent refuses; tools `search_course_materials`
and `python_calc` (safe arithmetic, no `eval`); metrics compared with phase 3.
CLAUDE.md keeps the RAG core free of LangChain wrappers.

**Decision.**
* **Five nodes** - route, retrieve, rewrite, generate, refuse - and one
  conditional edge after retrieve. The nodes are plain async functions over
  this project's own retrieval (`Answerer.retrieve`) and an OpenAI-compatible
  client; LangGraph contributes the state, the edges and the loop, nothing
  of LangChain's model or retriever abstractions.
* **The router answers from a JSON enum** of the indexed courses plus "none",
  so it cannot name a course that does not exist; an unreadable reply means no
  filter. A course the user gave is never overridden.
* **"Weak" is the reranker's best score** below `AGENT_MIN_SCORE`. The
  cross-encoder reads question and passage together, so its score is a
  judgement of relevance, unlike an RRF score, which only ranks - with the
  reranker off, the agent never refuses on score. The threshold, 0.1, comes
  from ten course questions and eight unrelated ones: the best score was
  0.976-0.998 on every course question and 0.0008-0.023 on the unrelated
  ones, so anything in between separates them. It is low on purpose: a
  wrong refusal costs a student the answer, a missed one only the seconds
  in which the model reads the fragments and refuses by itself. (One
  question I had filed as unrelated, on unification in logic programming,
  scored 0.94: the course's handbook does cover it.) The author's
  unanswerable questions - plausible ones, by design - are what the
  threshold should finally be tuned on.
* **Two rewrites at most**, the second without the router's filter: a wrong
  route is the likeliest reason a good question finds nothing.
* **A refusal costs no generation**: the phase 3 prompt would only produce
  the same refusal, ten seconds later.
* **Tools inside generation.** `python_calc` walks the expression's syntax
  tree and allows numbers, operators, `pi`/`e` and a list of math functions;
  names, attributes, calls to anything else, strings and comprehensions are
  refused before anything runs, and exponents, factorials and result sizes
  are bounded (`((10**1000)**1000)**1000` is arithmetic too). A search from
  the model adds at most three fragments, labelled after the first ten, and
  only within a reserve of 1 500 tokens kept free in the prompt.
* **No token streaming** for the agent: a reply may turn out to be a tool
  call, known only when it is complete. The API sends the whole answer as
  one `token` event.

**Checked live** on five ad-hoc questions: the router sent an operating
systems question to `os` and a combinatorics one to `tvims`, and answered
"none" for a football question; on the combinatorics question the model
called `python_calc` by itself and cited the fragment with the definition;
the football question scored 0.004-0.005 on all three searches and was
refused with no generation. A warm router call is 4-5 s.

**Measured against phase 3** by `nsu-rag eval --stage agent`: both systems
end to end on the author's questions, asked without a course as a student
would, judged the same way, plus the router's accuracy and the searches per
question.

**Alternatives rejected.**
* *LangChain's agent executor with its chat-model and retriever wrappers.*
  It hides the prompts and the retrieval this project measures, against
  CLAUDE.md's rule for the RAG core.
* *Letting the model decide everything through tools* (a ReAct loop from the
  start). Each extra decision is an extra 9B call on a laptop, and the fixed
  path - route, search, check - is where the phase 3 failures are expected.
* *`eval` with a restricted namespace for the calculator.* Restricted `eval`
  has been escaped many times over (`().__class__.__bases__...`); walking the
  tree allows only what is listed.

---

## ADR-032. The container answers too: CPU search, hybrid only, the index mounted

**Context.** Phase 0's image served `/health` alone: the search models live in
the `ml` extra, and the index is built natively on the Mac, where Metal is.
A reader of this repository who runs `docker compose up` should get a service
that answers.

**Decision.**
* **The image takes an `EXTRAS` build argument**; `docker-compose.yml` builds
  it with `ml`. CI builds it without, which keeps its check fast and still
  proves the Dockerfile.
* **torch from PyTorch's CPU index on Linux.** The PyPI wheel for Linux
  depends on CUDA libraries - gigabytes, for a container that has no GPU. A
  `[tool.uv.sources]` entry with a `sys_platform == 'linux'` marker sends
  Linux to `download.pytorch.org/whl/cpu` and leaves the Mac on PyPI's wheel
  with Metal; one lock file serves both.
* **Hybrid search only, in the container** (`RERANKER_ENABLED=false`): the
  cross-encoder reads thirty passages per question, some 4 s on the Mac's
  GPU and far longer on a container's CPU.
* **The index is mounted, not copied**: `./data` read-only - the chunks for
  BM25 and the course titles - and Qdrant is reached over the compose
  network. Nothing from the corpus is ever in the image.
* **Models download on the first question** into a named volume, kept
  between runs; the image creates the cache directory owned by its user, so
  that the volume takes that owner.

**Checked.** The image builds with the `ml` extra - 2.3 GB, torch fetched as
the CPU build - and the container's `/health` reaches Qdrant over the compose
network and Ollama on the host. The first attempt at `/ask` did not finish:
with the OCR of the full corpus running and 9 GB of swap in use, OrbStack's VM
had no memory to load the models into and stopped answering until it was
stopped. With the long jobs done, the container answered: its first question
took 194 s, some two minutes of it downloading and loading the models; the
answer itself - hybrid search in 115 ms on the CPU, the host's Ollama reading
the prompt in 35 s - came back with seven citations and no citation errors.
The lesson stands in the README: a 16 GB laptop runs this service beside
long jobs natively, not in a container.

**Alternatives rejected.**
* *Models baked into the image.* Gigabytes in every rebuild, for files that
  never change.
* *Ollama in the container too.* No Metal in a container on macOS: the 9B
  model on a CPU is not a service (ADR-004 context).
* *A second Dockerfile for the answering image.* Two files to keep in step
  for one build argument.
