"""Chunking rules, on synthetic pages, with one word counted as one token.

A fake counter keeps the budgets small enough to reason about by eye; the real
pipeline passes the embedding model's tokenizer.
"""

from __future__ import annotations

import itertools
import re

from nsu_rag_llm.ingest.chunking import (
    DocInfo,
    assign_sections,
    chunk_document,
    hard_split,
    join_across_pages,
    page_blocks,
    split_listing,
    split_table,
    tail,
)


def words(text: str) -> int:
    return len(text.split())


DOC = DocInfo(
    course="os",
    course_title="Операционные системы",
    doc_title="Процессы и потоки",
    file="os/sem4/processes.pdf",
    sha256="0" * 64,
    semester=4,
    origin="digital",
)


def sentences(n: int, topic: str = "процесс") -> str:
    """n distinct sentences of six words each."""
    return " ".join(f"Предложение {i} описывает {topic} номер {i}." for i in range(n))


def chunk(pages: list[str], **kwargs) -> list:
    options = {"budget": 40, "overlap": 8, "min_before_break": 12} | kwargs
    return chunk_document(DOC, list(enumerate(pages, start=1)), words, "fake-model", **options)


def test_blocks_are_told_apart() -> None:
    page = "# Глава 1\n\nАбзац текста.\n\n```\nint x;\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |"

    kinds = [b.kind for b in page_blocks(page, 1)]

    assert kinds == ["heading", "text", "code", "table"]


def test_a_heading_the_length_of_a_paragraph_is_a_paragraph() -> None:
    """The layout model sometimes marks a whole bold paragraph as a heading."""
    page = "# " + "очень " * 60 + "длинная строка"

    assert [b.kind for b in page_blocks(page, 1)] == ["text"]


def test_a_paragraph_broken_by_a_page_is_joined() -> None:
    blocks = join_across_pages([(1, "Процесс переходит в состояние"), (2, "ожидания ввода.")])

    assert len(blocks) == 1
    assert blocks[0].text == "Процесс переходит в состояние ожидания ввода."
    assert (blocks[0].page_start, blocks[0].page_end) == (1, 2)


def test_a_page_starting_a_new_sentence_is_not_joined() -> None:
    blocks = join_across_pages([(1, "Первое предложение."), (2, "Второе предложение.")])

    assert len(blocks) == 2


def test_a_word_hyphenated_across_pages_is_rejoined() -> None:
    blocks = join_across_pages([(1, "Происходит пере-"), (2, "ключение контекста.")])

    assert blocks[0].text == "Происходит переключение контекста."


def test_sections_nest_by_heading_level() -> None:
    blocks = page_blocks("# Глава\n\n## Раздел\n\nТекст.\n\n# Другая глава\n\nЕщё текст.", 1)
    assign_sections(blocks)

    paths = [b.path for b in blocks if b.kind == "text"]
    assert paths == [("Глава", "Раздел"), ("Другая глава",)]


def test_a_short_document_is_one_chunk_with_its_context() -> None:
    chunks = chunk(["# Потоки\n\nПоток — единица планирования."])

    assert len(chunks) == 1
    assert chunks[0].section_path == ["Потоки"]
    assert chunks[0].context == "Операционные системы › Процессы и потоки › Потоки"
    assert chunks[0].embedding_text.startswith(chunks[0].context)


def test_no_chunk_exceeds_the_budget() -> None:
    chunks = chunk([sentences(40), sentences(40, "поток")])

    assert len(chunks) > 5
    assert max(c.tokens for c in chunks) <= 40


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    chunks = chunk([sentences(30)])

    for previous, current in itertools.pairwise(chunks):
        last_sentence = re.split(r"(?<=\.)\s+", previous.text)[-1]
        assert current.text.startswith(last_sentence)


def test_small_sections_are_packed_together_under_their_common_section() -> None:
    """Run-in labels like "Объявление" and "Результат" must not become chunks
    of their own, and the chunk is labelled with the section holding both."""
    page = (
        "# Функция join\n\n"
        "## Объявление\n\nvoid join();\n\n"
        "## Результат\n\nЖдёт завершения потока.\n\n"
        "## Исключения\n\nsystem_error при ошибке."
    )

    # The real threshold is 100 tokens, and a reference entry like this one is
    # about thirty: scaled to one word per token, that is a threshold of 30.
    chunks = chunk([page], min_before_break=30)

    assert len(chunks) == 1
    assert chunks[0].section_path == ["Функция join"]


def test_a_heading_starts_a_new_chunk_once_the_current_one_is_big_enough() -> None:
    page = f"# Первый раздел\n\n{sentences(3)}\n\n# Второй раздел\n\n{sentences(2, 'поток')}"

    chunks = chunk([page])

    assert [c.section_path for c in chunks] == [["Первый раздел"], ["Второй раздел"]]
    assert chunks[1].text.startswith("# Второй раздел")


def test_overlap_never_leaks_into_the_next_section() -> None:
    page = f"# Первый\n\n{sentences(10)}\n\n# Второй\n\n{sentences(2, 'поток')}"

    chunks = chunk([page])
    second_section = [c for c in chunks if c.section_path == ["Второй"]]

    assert second_section
    assert "процесс" not in second_section[0].text


def test_a_heading_is_never_left_at_the_end_of_a_chunk() -> None:
    chunks = chunk([f"{sentences(6)}\n\n# Новый раздел\n\n{sentences(6, 'поток')}"])

    for c in chunks:
        assert not c.text.rstrip().splitlines()[-1].startswith("#")


def test_a_listing_that_fits_stays_whole() -> None:
    listing = "```\nint main(void) {\n    return 0;\n}\n```"

    chunks = chunk([f"Пример программы.\n\n{listing}"])

    assert listing in chunks[0].text
    assert chunks[0].has_code


def test_a_long_listing_is_split_between_its_parts_not_inside_a_line() -> None:
    parts = [f"void f{i}(void) {{\n    call_{i}(a, b, c);\n}}" for i in range(12)]
    listing = "```\n" + "\n\n".join(parts) + "\n```"

    pieces = split_listing(listing, budget=20, count=words)

    assert len(pieces) > 1
    body = [line for p in pieces for line in p.splitlines() if not line.startswith("```")]
    original = [line for line in listing.splitlines() if not line.startswith("```")]
    assert [line for line in body if line] == [line for line in original if line]
    assert all(p.startswith("```\n") and p.endswith("\n```") for p in pieces)


def test_a_long_table_is_split_between_rows_with_its_header_repeated() -> None:
    header = "| функция | POSIX | Linux |\n|---|---|---|"
    rows = [f"| call_{i} | да | да |" for i in range(30)]
    table = header + "\n" + "\n".join(rows)

    pieces = split_table(table, budget=40, count=words)

    assert len(pieces) > 1
    assert all(p.startswith(header) for p in pieces)
    assert sum(p.count("| call_") for p in pieces) == 30


def test_a_table_with_an_enormous_header_is_split_without_repeating_it() -> None:
    # 25 "tokens": more than half the budget, so repeating it would leave no
    # room for rows - but still small enough to appear intact once.
    header = "| " + " | ".join(f"колонка{i}" for i in range(12)) + " |"
    table = header + "\n" + "\n".join(f"| строка {i} |" for i in range(20))

    pieces = split_table(table, budget=40, count=words)

    assert sum(p.count(header) for p in pieces) == 1


def test_a_piece_with_no_structure_left_is_still_cut_to_size() -> None:
    blob = "x" * 5000  # a table cell without a single space

    pieces = hard_split(blob, budget=10, count=lambda s: len(s) // 100 + 1)

    assert "".join(pieces) == blob
    assert all(len(p) // 100 + 1 <= 10 for p in pieces)


def test_pages_are_recorded_on_every_chunk() -> None:
    chunks = chunk([sentences(8), sentences(8, "поток"), sentences(8, "сигнал")])

    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert all(c.page_start <= c.page_end for c in chunks)


def test_ids_are_stable_and_change_only_where_the_text_changed() -> None:
    pages = [f"# Раздел {i}\n\n{sentences(5, f'тема{i}')}" for i in range(6)]
    before = chunk(pages)
    again = chunk(pages)
    pages[3] = pages[3].replace("тема3", "изменено")
    after = chunk(pages)

    assert [c.id for c in before] == [c.id for c in again]
    changed = {c.id for c in before} ^ {c.id for c in after}
    touched = [c for c in before if c.id in changed]
    assert touched and all(c.page_start <= 4 <= c.page_end for c in touched)


def test_a_long_last_sentence_carries_its_end_as_a_marked_fragment() -> None:
    """Whole sentences when they fit; otherwise the closing words, marked."""
    long_sentence = "Теорема утверждает " + "что " * 30 + "функция ограничена."

    assert tail(f"Короткое. {long_sentence}", tokens=8, count=words).startswith("…")
    assert tail("Первое. Второе предложение.", tokens=8, count=words) == (
        "Первое. Второе предложение."
    )


def test_a_lone_marker_is_not_carried_over() -> None:
    page = " ".join(f"Утверждение {i} о свойствах функции." for i in range(20)) + "\n\n[формула]"
    chunks = chunk([page, "продолжение " * 30])

    assert not any(c.text.startswith("[формула]") for c in chunks[1:])


def test_overlap_survives_a_small_heading_that_does_not_start_a_chunk() -> None:
    """Regression: every heading used to throw the carried overlap away, even
    when the heading was too small to start a chunk of its own - which cost 2
    559 of the MVP's chunk boundaries their overlap."""
    page = (
        f"# Раздел\n\n{sentences(6)}\n\nКороткий абзац продолжения.\n\n"
        f"## Подпись\n\nЕщё немного текста."
    )

    chunks = chunk([page], min_before_break=30)

    assert len(chunks) == 2
    closing = re.split(r"(?<=\.)\s+", chunks[0].text)[-1]
    assert chunks[1].text.startswith(closing)


def test_the_sentence_before_a_formula_is_carried_with_it() -> None:
    page = sentences(6) + "\n\nПлотность суммы равна\n\n[формула]\n\n" + sentences(6, "поток")

    chunks = chunk([page])

    boundary = next(c for c in chunks[1:] if "Плотность" in c.text)
    assert boundary.text.startswith("Плотность суммы равна")
