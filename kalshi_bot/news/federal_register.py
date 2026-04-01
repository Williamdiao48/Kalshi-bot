import logging
from typing import Any

import aiohttp

FEDERAL_REGISTER_API_BASE = "https://www.federalregister.gov/api/v1"


async def fetch_documents(
    session: aiohttp.ClientSession,
    *,
    agency_slug: str,
    per_page: int = 50,
) -> list[dict[str, Any]]:
    """Fetch recent documents from the Federal Register for a given agency.

    Args:
        session:     Shared aiohttp session.
        agency_slug: Federal Register agency identifier (e.g. "environmental-protection-agency").
        per_page:    Number of documents to request (max 1000 per API docs).

    Returns:
        List of document dicts, newest first.
    """
    url = f"{FEDERAL_REGISTER_API_BASE}/documents.json"
    params = {
        "conditions[agencies][]": agency_slug,
        "order": "newest",
        "per_page": per_page,
        "fields[]": ["document_number", "title", "abstract", "html_url", "publication_date", "agencies"],
    }

    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            docs = data.get("results", [])
            logging.info(
                "Federal Register [%s]: fetched %d documents.", agency_slug, len(docs)
            )
            return docs
    except aiohttp.ClientResponseError as exc:
        logging.error(
            "Federal Register HTTP error %s for agency '%s': %s",
            exc.status, agency_slug, exc.message,
        )
    except aiohttp.ClientError as exc:
        logging.error("Federal Register request error for agency '%s': %s", agency_slug, exc)

    return []
