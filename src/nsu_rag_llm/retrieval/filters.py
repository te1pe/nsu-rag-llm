"""The course filter, in the form Qdrant wants it."""

from __future__ import annotations

from qdrant_client import models


def course_filter(course: str | None) -> models.Filter | None:
    """Restrict a vector search to one course; None searches everything.

    `course` is a keyword payload index (ADR-021), so Qdrant filters while it
    searches instead of discarding results after the fact.
    """
    if course is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="course", match=models.MatchValue(value=course))]
    )
