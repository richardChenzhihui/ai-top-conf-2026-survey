#!/usr/bin/env python3
"""Fetch all CVPR 2026 papers (title + abstract) from two static JSON files on the virtual site.
No PDF download, no HTML scraping — just two HTTP GETs.
"""
import argparse, json, os, sys
import requests

META_URL  = "https://cvpr.thecvf.com/static/virtual/data/cvpr-2026-orals-posters.json"
ABSTR_URL = "https://cvpr.thecvf.com/static/virtual/data/cvpr-2026-abstracts.json"

def fetch(url: str, label: str) -> dict:
    print(f"Fetching {label} ...", flush=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f"  -> {len(json.dumps(data))//1024} KB received", flush=True)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/cvpr2026_raw.json")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    meta      = fetch(META_URL, "cvpr-2026-orals-posters.json")
    abstracts = fetch(ABSTR_URL, "cvpr-2026-abstracts.json")

    results = meta.get("results", [])
    print(f"Total papers in meta: {len(results)}", flush=True)

    papers = []
    no_abstract = 0
    for p in results:
        pid = str(p["id"])
        abstract = abstracts.get(pid, "")
        if not abstract:
            no_abstract += 1

        track = (p.get("decision") or "")
        track = track.replace("Accept (", "").replace(")", "").strip()

        papers.append({
            "paper_id":   f"CVPR26_{pid}",
            "title_en":   p.get("name", ""),
            "year":       "2026",
            "conference": "CVPR",
            "track":      track,
            "abstract":   abstract,
            "keywords":   ", ".join(p.get("keywords") or []),
            "authors":    ", ".join(a.get("fullname", "") for a in (p.get("authors") or [])),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(papers)} papers -> {args.output}")
    print(f"  With abstract: {len(papers) - no_abstract}")
    print(f"  Without abstract: {no_abstract}")

if __name__ == "__main__":
    main()
