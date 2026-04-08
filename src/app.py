"""
app.py — Patent Search Flask Application
=========================================
Serves the patent search UI. Search logic will be wired in later.
Run with:  python src/app.py
"""

from pathlib import Path
import json
import re

from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Placeholder data (replaced when real retrieval is wired in)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_JSONL = PROJECT_ROOT / "data" / "interim" / "parsed_patents.jsonl"

MOCK_RESULTS = [
    {
        "ucid": "WO-2000002299-A1",
        "title": "Improved Power Supply Assembly for Hand-Held Communications Device",
        "abstract": "A portable phone has an internal battery and an external battery pack that is "
                    "releasably attachable to the phone. A control unit controls connection of the "
                    "respective batteries depending on detection of the external battery voltage.",
        "ipc": "H02J 7/00",
        "score": 95.33,
    },
    {
        "ucid": "WO-2000001192-A2",
        "title": "Call Admission Control System for Wireless ATM Networks",
        "abstract": "A system which determines whether to accept a wireless connection to a network "
                    "device receives a request, determines a nominal cell rate at which cells are "
                    "exiting from a buffer, and decides whether to accept based on buffer space.",
        "ipc": "H04Q 11/04",
        "score": 87.63,
    },
    {
        "ucid": "WO-2000000725-A1",
        "title": "Engine System Employing an Unsymmetrical Cycle",
        "abstract": "An engine system employs an unsymmetrical thermodynamic cycle to improve "
                    "fuel efficiency. The system includes a novel combustion chamber design "
                    "that allows different compression and expansion ratios.",
        "ipc": "F02B 41/00",
        "score": 74.12,
    },
]


def _snippet(text: str, max_words: int = 35) -> str:
    """Truncate text to max_words words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def search_patents(query: str) -> list[dict]:
    """
    Placeholder search — returns mock results filtered by query keyword.
    Replace this function body with real BM25F / dense retrieval later.
    """
    q = query.lower()
    results = []
    for r in MOCK_RESULTS:
        haystack = (r["title"] + " " + r["abstract"]).lower()
        if any(word in haystack for word in q.split()):
            results.append(r)
    # If nothing matches, return all mock results so the UI is always populated
    return results if results else MOCK_RESULTS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("index.html")

    results = search_patents(query)

    # Add snippet field
    for r in results:
        r["snippet"] = _snippet(r.get("abstract", ""))

    return render_template(
        "results.html",
        query=query,
        results=results,
        total=len(results),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
