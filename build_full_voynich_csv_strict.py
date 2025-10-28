# build_full_voynich_csv_strict.py  (ASCII only)
# Strict IVTFF -> CSV converter that keeps only real Voynich EVA lines.

import os
import re
import argparse
import pandas as pd

# Defaults match folder layout
DEFAULT_INPUT_DIR   = "input"
DEFAULT_OUTPUT_CSV  = os.path.join(DEFAULT_INPUT_DIR, "voynich_full_transcription.csv")

# Robust folio markers commonly found in IVTFF/interlinear files
TAG_FOLIO = re.compile(r"<f0*([0-9]{1,3})([rv])(?:[0-9]*)?(?:\.P\.[0-9]+;[A-Z])?>", re.I)
PAGE_HASH = re.compile(r"^\s*#\s*page\s*0*([0-9]{1,3})([rv])", re.I)
PAGE_EQ   = re.compile(r"^\s*=\s*0*([0-9]{1,3})([rv])\s*=", re.I)
FALLBACK  = re.compile(r"\bf0*([0-9]{1,3})([rv])\b", re.I)

# Seed patterns typical of Voynich EVA tokens
SEED_PATTERNS = re.compile(
    r"(qokaiin|qokeedy|qokedy|qokar|ykal|ydar|ykor|daiin|aiin|shedy|chedy|shor|chol|okar|okain|\bdy\b|\bar\b)"
)

# Tight EVA set and helpers
EVA_LETTERS = set("acdefghiklmnopqrsty")  # no b j u v w x z
EVA_TOKEN   = re.compile(r"^[acdefghiklmnopqrsty]+$")

# English/meta words we never want
BAD_WORDS = {
    "last","edited","identification","title","first","page","folio","author","beinecke",
    "yale","rene","stolfi","landini","annotation","notes","copyright","translation",
    "intro","index","contents","section","table","figure","updated","revision"
}

# Known EVA units to anchor "looks like Voynich"
EVA_ANCHORS = {
    "daiin","aiin","qokaiin","qokain","qokeedy","qokedy","qokar","okar","okain","shedy",
    "chedy","shor","chol","ykal","dy","ar"
}

def clean_text(s):
    s = re.sub(r"<[^>]+>", " ", s)        # drop tags
    s = re.sub(r"\[[^\]]+\]", " ", s)     # drop bracket notes
    s = re.sub(r"[^a-zA-Z\s\.]", " ", s)  # keep letters, space, dot
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def looks_like_english(line_lower):
    tokens = set(line_lower.split())
    return any(w in tokens for w in BAD_WORDS)

def eva_purity_ratio(line_lower):
    toks = [t for t in re.split(r"[.\s]+", line_lower) if t]
    if not toks:
        return 0.0, 0, 0
    eva_like = sum(1 for t in toks if EVA_TOKEN.match(t))
    return eva_like / max(1, len(toks)), eva_like, len(toks)

def has_eva_anchor(line_lower):
    return any(a in line_lower for a in EVA_ANCHORS)

def is_probably_eva(raw_line):
    if not SEED_PATTERNS.search(raw_line):
        return False
    cln = clean_text(raw_line)
    if not cln or looks_like_english(cln):
        return False
    ratio, eva_like, total = eva_purity_ratio(cln)
    if ratio < 0.8:   # require 80%+ EVA-only tokens
        return False
    if not has_eva_anchor(cln):
        return False
    return True

def folio_sort_key(folio):
    m = re.match(r"^0*([0-9]+)([rv])$", folio)
    if m:
        num  = int(m.group(1))
        side = m.group(2)
        side_ord = 0 if side == 'r' else 1
        return (num, side_ord)
    return (9999, 9)

def build_full_csv(ivtt_path, out_csv=DEFAULT_OUTPUT_CSV):
    if not os.path.exists(ivtt_path):
        raise FileNotFoundError("IVTFF file not found: %s" % ivtt_path)

    folio    = None
    buckets  = {}

    with open(ivtt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith("#"):
                continue

            line = raw.rstrip("\n")
            m    = TAG_FOLIO.search(line) or PAGE_HASH.search(line) or PAGE_EQ.search(line)
            if m:
                folnum  = m.group(1).lstrip("0")
                folside = m.group(2).lower()
                folio   = folnum + folside
                buckets.setdefault(folio, [])
                continue

            if not folio:
                m2 = FALLBACK.search(line)
                if m2:
                    folnum  = m2.group(1).lstrip("0")
                    folside = m2.group(2).lower()
                    folio   = folnum + folside
                    buckets.setdefault(folio, [])
                else:
                    continue

            if is_probably_eva(line):
                buckets[folio].append(clean_text(line))

    rows = []
    for fol, chunks in buckets.items():
        text = " ".join(chunks).strip()
        if text:
            rows.append({"folio": fol, "text": text})

    rows = sorted(rows, key=lambda r: folio_sort_key(r["folio"]))
    df   = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[OK] wrote %d folios to %s" % (len(df), out_csv))
    return df

def main():
    parser = argparse.ArgumentParser(description="Strict IVTFF -> CSV converter (folio,text)")
    parser.add_argument("--ivtt", default=os.path.join(DEFAULT_INPUT_DIR, "LSI_ivtff_0d.txt"),
                        help="Path to IVTFF file (default: input/LSI_ivtff_0d.txt)")
    parser.add_argument("--out",  default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV path (default: input/voynich_full_transcription.csv)")
    args = parser.parse_args()
    build_full_csv(args.ivtt, args.out)

if __name__ == "__main__":
    main()
