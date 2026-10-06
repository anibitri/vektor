#!/usr/bin/env python3
"""Fetches the RAG corpus: the arXiv abstracts in information retrieval (cs.IR)
submitted from January to June 2025, written as a /rag/ingest request body.

arXiv's descriptive metadata (titles and abstracts) is free to use under CC0
(https://info.arxiv.org/help/api/tou.html). The API asks for at most one request
every three seconds, so this takes about 15 seconds.

Usage: python3 scripts/fetch_corpus.py data/rag-corpus.json
"""

import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

QUERY = "cat:cs.IR AND submittedDate:[202501010000 TO 202506302359]"
PAGE = 500
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def fetch_page(start):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": QUERY, "start": start, "max_results": PAGE,
        "sortBy": "submittedDate", "sortOrder": "ascending"})
    with urllib.request.urlopen(url, timeout=120) as response:
        return ET.fromstring(response.read())


def main(out_path):
    docs = {}
    start, total = 0, None
    while total is None or start < total:
        for attempt in range(4):  # the API sometimes returns an empty page; try again
            feed = fetch_page(start)
            entries = feed.findall(ATOM + "entry")
            if entries:
                break
            time.sleep(3 * (attempt + 2))
        total = int(feed.find(OPENSEARCH + "totalResults").text)
        for entry in entries:
            # "http://arxiv.org/abs/2501.01234v2" -> "2501.01234"
            arxiv_id = entry.find(ATOM + "id").text.rsplit("/abs/", 1)[1].rsplit("v", 1)[0]
            title = " ".join(entry.find(ATOM + "title").text.split())
            abstract = " ".join(entry.find(ATOM + "summary").text.split())
            docs[arxiv_id] = {"id": arxiv_id, "title": title, "text": f"{title}. {abstract}"}
        start += PAGE
        print(f"{len(docs)} of {total} abstracts", file=sys.stderr)
        time.sleep(3)
    with open(out_path, "w") as f:
        json.dump({"documents": [docs[k] for k in sorted(docs)]}, f)
    print(f"wrote {out_path}: {len(docs)} abstracts", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
