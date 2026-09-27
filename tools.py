"""
tools.py – SerpAPI web search tool for the Weather Agent.
"""

import os
import requests
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_URL = "https://serpapi.com/search"


def search_web(query: str, num_results: int = 5) -> dict:
    """
    Search the web using SerpAPI and return structured results.

    Args:
        query:       The search query string.
        num_results: How many organic results to return (default 5).

    Returns:
        A dict with keys:
          - "results": list of {"title", "link", "snippet"}
          - "error":   str (only present on failure)
    """
    if not SERPAPI_KEY:
        return {
            "error": (
                "SERPAPI_KEY is not set. Create a .env file (see .env.example) "
                "with SERPAPI_KEY=<your key> from https://serpapi.com/manage-api-key"
            )
        }

    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "engine": "google",
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        results = [
            {
                "title":   r.get("title", ""),
                "link":    r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in organic[:num_results]
        ]
        return {"results": results}

    except requests.exceptions.Timeout:
        return {"error": "SerpAPI request timed out."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"SerpAPI HTTP error: {e}"}
    except Exception as e:
        return {"error": f"SerpAPI error: {e}"}


def format_search_results(data: dict) -> str:
    """Convert search_web() output into a readable string for the LLM."""
    if "error" in data:
        return f"Search error: {data['error']}"

    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['snippet']}")
        lines.append(f"   Source: {r['link']}")
    return "\n".join(lines)


# ── Ollama tool schema ──────────────────────────────────────────────────────
# Used when calling Ollama's /api/chat with tools support.
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current information. "
            "Use this to get live weather data, forecasts, or any up-to-date info."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
    },
}
