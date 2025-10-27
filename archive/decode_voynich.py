"""
decode_voynich.py
Decodes Voynich Manuscript folios using the Comprehensive Method and Mapping (CMM)
Author: Kenneth Young, PhD (2025)
"""

import pandas as pd
import argparse
import os
import re

# -------------------------------------------------------
# CONFIGURATION DEFAULTS
# -------------------------------------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEFAULT_FILE = "voynich_full_transcription.csv"

def make_char_map():
    return {
        "q":"quando","o":"oleum","k":"clavis","e":"et","d":"dare","y":"vitalis","a":"aqua","i":"in",
        "s":"sanguis","h":"humus","c":"caput","t":"tempus","r":"radialis","l":"luna","n":"",
        "f":"femina","p":"pax"
    }

def make_unit_map():
    k="clavis"
    return {
        "ydar":"vitalis dare aqua radialis",
        "ykor":"vitalis " + k + " oleum radialis",
        "qokaiin":"quando oleum " + k + " aqua in",
        "qokeedy":"quando oleum " + k + " et dare vitalis",
        "shedy":"sanguis humus et dare vitalis",
        "shor":"sanguis humus oleum radialis",
        "ykal":"vitalis " + k + " aqua luna",
        "daiin":"dare aqua in","aiin":"aqua in","dy":"dare vitalis",
        "ar":"aqua radialis","paxar":"pax aqua radialis"
    }

def tokenizer(line, unit_map, char_map):
    tokens = []
    for word in str(line).split():
        w = word.lower().strip()
        if not w:
            continue
        if w in unit_map:
            tokens.extend(unit_map[w].split())
            continue
        i = 0
        while i < len(w):
            matched = False
            for L in range(min(7, len(w)-i), 1, -1):
                seg = w[i:i+L]
                if seg in unit_map:
                    tokens.extend(unit_map[seg].split())
                    i += L
                    matched = True
                    break
            if matched: 
                continue
            ch = w[i]
            if ch in char_map:
                val = char_map[ch]
                if val: 
                    tokens.append(val)
            i += 1
    return tokens

def smoothen(tokens):
    out = []
    for t in tokens:
        out.append(t)
        if t in {"quando","dare","aqua","oleum"}:
            out.append("fac")
        if t in {"vitalis","aqua","oleum"}:
            out.append("usa")
    return out

def decode_text(eva_text):
    char_map = make_char_map()
    unit_map = make_unit_map()
    toks = tokenizer(eva_text, unit_map, char_map)
    toks = smoothen(toks)
    latin = " ".join(toks)
    return latin

def decode_file(input_path, output_path, folio=None):
    df = pd.read_csv(input_path, dtype=str).fillna("")
    if folio:
        df = df[df["folio"].str.lower() == folio.lower()]
        if df.empty:
            raise ValueError("Folio %s not found in %s." % (folio, input_path))
    results = []
    for _, row in df.iterrows():
        f = row["folio"]
        t = row["text"]
        latin = decode_text(t)
        results.append({"folio": f, "eva": t, "latin": latin})
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print("[OK] Decoded folios saved to %s" % output_path)

def main():
    parser = argparse.ArgumentParser(description="Decode Voynich Manuscript using CMM")
    parser.add_argument("--infile", default=os.path.join(INPUT_DIR, DEFAULT_FILE), help="Input CSV path")
    parser.add_argument("--folio", default=None, help="Target folio (optional)")
    parser.add_argument("--outfile", default=os.path.join(OUTPUT_DIR, "decoded_folios.csv"), help="Output CSV path")
    args = parser.parse_args()
    decode_file(args.infile, args.outfile, args.folio)

if __name__ == "__main__":
    main()
