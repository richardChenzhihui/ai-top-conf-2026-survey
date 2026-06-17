#!/usr/bin/env python3
"""Fetch all papers for a conference from OpenReview API (ICLR 2026 / NeurIPS 2025).
No PDF download. Paginates through all notes, extracts title + abstract.
"""
import argparse, json, os, time
import requests

BASE_URL = "https://api2.openreview.net/notes"

def fetch_all(venueid: str, conf: str, year: str) -> list:
    offset, limit, total = 0, 1000, None
    all_papers = []

    session = requests.Session()
    session.headers.update({"User-Agent": "academic-research-script/1.0"})

    while True:
        params = {
            "content.venueid": venueid,
            "limit": limit,
            "offset": offset,
        }
        print(f"  Fetching offset={offset} ...", end=" ", flush=True)
        r = session.get(BASE_URL, params=params, timeout=60)
        if r.status_code == 429:
            print("rate-limited, sleeping 30s ...")
            time.sleep(30)
            continue
        r.raise_for_status()
        data = r.json()

        if total is None:
            total = data.get("count", 0)
            print(f"Total count: {total}")

        notes = data.get("notes", [])
        print(f"got {len(notes)} notes", flush=True)

        for note in notes:
            c = note.get("content", {})

            def val(field):
                v = c.get(field, "")
                return v.get("value", "") if isinstance(v, dict) else (v or "")

            title    = val("title")
            abstract = val("abstract")
            venue    = val("venue")

            if not title:
                continue

            # track from venue string: "ICLR 2026 Poster" -> "Poster"
            track = ""
            for tok in ["Oral", "Spotlight", "Poster", "Highlight"]:
                if tok.lower() in venue.lower():
                    track = tok
                    break

            paper_id = f"{conf.upper()}{year[-2:]}_{note.get('id', '')}"
            all_papers.append({
                "paper_id":   paper_id,
                "title_en":   title,
                "year":       year,
                "conference": conf.upper(),
                "track":      track,
                "abstract":   abstract,
                "keywords":   "",
                "authors":    ", ".join(
                    (a.get("value", a) if isinstance(a, dict) else a)
                    for a in (val("authors") if isinstance(c.get("authors"), list) else [])
                ),
            })

        offset += len(notes)
        if total is not None and offset >= total:
            break
        if not notes:
            break

        time.sleep(0.5)

    return all_papers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venueid", required=True, help="e.g. ICLR.cc/2026/Conference")
    ap.add_argument("--conf",    required=True, help="iclr | neurips")
    ap.add_argument("--year",    required=True, help="2026 | 2025")
    ap.add_argument("--output",  required=True)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Fetching {args.conf.upper()} {args.year} from OpenReview ...")
    papers = fetch_all(args.venueid, args.conf, args.year)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    with_abs = sum(1 for p in papers if p.get("abstract"))
    print(f"\nSaved {len(papers)} papers -> {args.output}")
    print(f"  With abstract: {with_abs}")
    print(f"  Without abstract: {len(papers) - with_abs}")

if __name__ == "__main__":
    main()
