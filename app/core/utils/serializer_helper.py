from typing import Any

from langchain_core.documents import Document


def search_results_to_dict(document: Document, score=None) -> dict[str, Any]:
    result = {
        "page_content": document.page_content,
        "metadata": document.metadata,
        # Add all fields of the Document class here
    }
    if score is not None:
        result["score"] = score
    return result