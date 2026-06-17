#!/usr/bin/env python3
"""Split a raw JSON paper list into fixed-size batch files for classification agents."""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True, help="raw JSON file (list of papers)")
    ap.add_argument("--out-dir",    required=True, help="directory to write batch_NNNN.json files")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--print-paths", action="store_true", help="print all output paths (for workflow args)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        papers = json.load(f)

    paths = []
    for i in range(0, len(papers), args.batch_size):
        batch = papers[i : i + args.batch_size]
        fname = f"batch_{i // args.batch_size:04d}.json"
        fpath = os.path.join(args.out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        paths.append(os.path.abspath(fpath))

    print(f"Split {len(papers)} papers into {len(paths)} batches of {args.batch_size} -> {args.out_dir}")

    if args.print_paths:
        print("\n--- PATHS (copy for workflow args) ---")
        print(json.dumps(paths))

if __name__ == "__main__":
    main()
