"""Cleaning rules, on synthetic text. The corpus never appears in tests."""

from __future__ import annotations

from nsu_rag_llm.ingest.cleaning import (
    FORMULA,
    clean_document,
    clean_page,
    edge_key,
    normalise_unicode,
    running_lines,
    split_fenced,
)

LISTING = "```\nint main(void) {\n    return  0;   \n}\n```"


def test_soft_hyphens_and_zero_width_characters_are_removed() -> None:
    # Invisible, but they split one word into two tokens for BM25.
    assert normalise_unicode("про\u00adцесс") == "процесс"
    assert normalise_unicode("по\u200bток\ufeff") == "поток"


def test_ligatures_are_expanded() -> None:
    assert normalise_unicode("\ufb01le \ufb02ow") == "file flow"


def test_superscripts_survive_normalisation() -> None:
    """NFKC would turn x² into x2 and quietly rewrite a formula; NFC must not."""
    assert normalise_unicode("x² + y³ = z½") == "x² + y³ = z½"


def test_a_word_broken_across_lines_is_joined() -> None:
    text, stats = clean_page("Операционная система управляет пере-\nключением процессов.")

    assert "переключением" in text
    assert stats.hyphens_joined == 1


def test_a_hyphen_before_a_capital_letter_is_kept() -> None:
    """A compound broken at its own hyphen continues in upper case."""
    text, stats = clean_page("Северо-\nЗападный округ")

    assert "Северо-\nЗападный" in text
    assert stats.hyphens_joined == 0


def test_list_items_are_not_mistaken_for_hyphenation() -> None:
    text, _ = clean_page("Состояния процесса:\n- готов\n- выполняется")

    assert "- готов" in text


def test_whitespace_is_tidied_in_prose() -> None:
    text, _ = clean_page("Процесс   и\u00a0поток.   \n\n\n\n\nДалее.")

    assert text == "Процесс и поток.\n\nДалее."


def test_code_listings_are_left_exactly_as_they_were() -> None:
    page = f"Пример:\n\n{LISTING}\n\nКонец   примера."

    text, _ = clean_page(page)

    assert LISTING in text  # the double space and trailing spaces inside survive
    assert "Конец примера." in text


def test_hyphenation_is_not_joined_inside_code() -> None:
    page = "```\nx = a -\nb\n```"

    text, stats = clean_page(page)

    assert text == page
    assert stats.hyphens_joined == 0


def test_an_unclosed_fence_protects_the_rest_of_the_page() -> None:
    pieces = split_fenced("текст\n```\ncode   here\n")

    assert pieces == [(False, "текст\n"), (True, "```\ncode   here\n")]


def test_edge_keys_match_across_page_numbers() -> None:
    assert edge_key("ГЛАВА 6. Распределения   82") == edge_key("ГЛАВА 6. Распределения 83")
    assert edge_key("**§ 7.** _Свёртка_") == edge_key("§ 7. Свёртка")


TOPICS = [
    "планировщик",
    "семафор",
    "мьютекс",
    "прерывание",
    "страница памяти",
    "сегмент",
    "сигнал",
    "канал",
    "сокет",
    "дескриптор",
    "таймер",
]


def _book(pages: int, header: str | None = "ГЛАВА 3. ПРОЦЕССЫ", footer: bool = True) -> list[str]:
    """Synthetic pages whose content lines differ in words, as real ones do.

    Content that differed only in a number would look like a running header to
    any rule that normalises page numbers away - that case is tested on its own.
    """
    out = []
    for n in range(1, pages + 1):
        a, b = TOPICS[n % len(TOPICS)], TOPICS[(n * 5 + 3) % len(TOPICS)]
        lines = [f"{header} {n}"] if header else []
        lines += [f"Здесь речь о том, как {a} связан с тем, что называют {b}.", f"Итог: {b}."]
        if footer:
            lines.append(f"— {n} —")
        out.append("\n".join(lines))
    return out


def test_running_headers_and_page_numbers_are_removed() -> None:
    cleaned, stats = clean_document(_book(10))

    assert all("ГЛАВА 3" not in page for page in cleaned)
    assert all("—" not in page for page in cleaned)
    assert sum(s.edge_lines_removed for s in stats) == 20
    assert all(page.count("\n") == 1 for page in cleaned)  # both content lines left


def test_a_line_repeated_in_the_middle_of_pages_is_content() -> None:
    """Only the outermost lines of a page can hold a running header."""
    pages = [f"Начало: {TOPICS[n]}.\nТеорема Байеса.\nКонец: {TOPICS[-n - 1]}." for n in range(10)]

    cleaned, stats = clean_document(pages)

    assert all("Теорема Байеса." in page for page in cleaned)
    assert sum(s.edge_lines_removed for s in stats) == 0


def test_numbered_content_at_the_top_of_many_pages_is_kept() -> None:
    """Digits are normalised so page numbers match, which makes "Задача 12."
    and "Задача 13." the same line. A problem book can start many pages with
    a problem; below a majority of pages, that must stay content."""
    pages = _book(10, header=None, footer=False)
    for n in range(4):  # 40% of pages
        pages[n] = f"Задача {n + 12}.\n" + pages[n]

    cleaned, stats = clean_document(pages)

    assert sum(page.startswith("Задача") for page in cleaned) == 4
    assert sum(s.edge_lines_removed for s in stats) == 0


def test_a_header_on_too_few_pages_is_kept() -> None:
    pages = _book(10, header=None, footer=False)
    pages[0] = "Предисловие\n" + pages[0]
    pages[1] = "Предисловие\n" + pages[1]

    assert running_lines(pages) == set()


def test_short_documents_are_not_second_guessed() -> None:
    assert running_lines(_book(3)) == set()


def test_placeholders_are_never_taken_for_running_lines() -> None:
    """In a formula-heavy text the marker sits at a page edge constantly."""
    pages = [f"{FORMULA}\nТекст страницы {n}.\n{FORMULA}" for n in range(10)]

    cleaned, _ = clean_document(pages)

    assert all(page.startswith(FORMULA) for page in cleaned)


def test_code_fences_at_page_edges_are_never_removed() -> None:
    pages = [f"```\ncall_{n}();\n```\nПояснение {n}." for n in range(10)]

    cleaned, _ = clean_document(pages)

    assert all(page.startswith("```") for page in cleaned)
