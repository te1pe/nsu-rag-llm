from __future__ import annotations

from qdrant_client import models


def course_filter(course: str | None) -> models.Filter | None:
    if course is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="course", match=models.MatchValue(value=course))]
    )
