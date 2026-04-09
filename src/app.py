"""
app.py — Patent Search Flask Application
Run with:  python src/app.py
"""

import re
from flask import Flask, render_template, request, abort
from markupsafe import Markup, escape
from retrieval import search, get_patent

app = Flask(__name__)

METHODS = {
    "r2": "BM25F",
    "r3": "BM25F → Dense Re-rank",
    "r4": "Weighted Fusion (α=0.5)",
    "r5": "RRF (k=60)",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _highlight(text: str, query_tokens: set) -> Markup:
    """Wrap query-matching words in <strong> tags."""
    if not text or not query_tokens:
        return Markup(escape(text))
    words  = text.split()
    result = []
    for word in words:
        clean = TOKEN_RE.search(word)
        if clean and clean.group().lower() in query_tokens:
            result.append(f"<strong>{escape(word)}</strong>")
        else:
            result.append(str(escape(word)))
    return Markup(" ".join(result))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search_route():
    query  = request.args.get("q", "").strip()
    method = request.args.get("method", "r2")

    if method not in METHODS:
        method = "r2"
    if not query:
        return render_template("index.html")

    results       = search(query, method=method, top_k=10)
    query_tokens  = {t.lower() for t in TOKEN_RE.findall(query)}

    # Add highlighted snippet to each result
    for r in results:
        r["snippet_html"] = _highlight(r["snippet"], query_tokens)
        r["title_html"]   = _highlight(r["title"],   query_tokens)

    return render_template(
        "results.html",
        query=query,
        results=results,
        total=len(results),
        method=method,
        method_label=METHODS[method],
        methods=METHODS,
    )


@app.route("/patent/<path:ucid>")
def patent_detail(ucid):
    patent = get_patent(ucid)
    if patent is None:
        abort(404)
    return render_template("patent.html", patent=patent)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
