"""The prompt: rules in the system message, labelled fragments and the question.

Russian throughout: the question, the handbooks and the answer are Russian, and
a 9B model follows instructions in the language it is answering in more
reliably than in a language it has to switch from.
"""

from __future__ import annotations

import re

from nsu_rag_llm.generation.citations import Fragment

#: The exact refusal. Its wording matters twice: the model is told to use it
#: verbatim, and phase 4 scores refusals by recognising it.
REFUSAL = "В методичках этого нет."

SYSTEM = f"""Ты отвечаешь на вопросы студента по методическим пособиям его курсов.

Правила:
1. Отвечай только по фрагментам из методичек, которые даны ниже. Не добавляй сведений, которых во фрагментах нет, даже если они тебе известны.
2. После каждого утверждения ставь ссылку на фрагмент, из которого оно взято, — его метку в квадратных скобках: [Ф1], [Ф2] или [Ф1, Ф3]. Используй только метки из списка фрагментов.
3. Если во фрагментах нет ответа на вопрос, ответь одной фразой: «{REFUSAL}» — и больше ничего не пиши.
4. «[формула]» во фрагменте — место, где стоит формула, которую текст не передаёт. Не восстанавливай её, а сошлись на фрагмент: читатель найдёт формулу на странице.
5. Отвечай по-русски, по существу и без вступлений."""  # noqa: E501 - prose, not code

FRAGMENTS_HEADER = "Фрагменты:"
QUESTION_PREFIX = "Вопрос: "


def fragment_block(fragment: Fragment) -> str:
    """One labelled fragment: where it is from, then its text."""
    chunk = fragment.chunk
    return f"{fragment.label} {chunk['context']} (с. {fragment.pages})\n{chunk['text']}"


def user_message(question: str, fragments: list[Fragment]) -> str:
    blocks = "\n\n".join(fragment_block(f) for f in fragments)
    return f"{FRAGMENTS_HEADER}\n\n{blocks}\n\n{QUESTION_PREFIX}{question}"


def messages(question: str, fragments: list[Fragment]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(question, fragments)},
    ]


def is_refusal(answer: str) -> bool:
    """Did the model decline? Tolerant of quotes, markup, case and what follows."""
    normalised = re.sub(r"[«»\"'*_]", "", answer).strip().lower()
    return normalised.startswith(REFUSAL.lower().rstrip("."))
