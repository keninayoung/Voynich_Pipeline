# voynich_pipeline.py  (ASCII safe) -- CMM v6
# - Context-aware multi-glyph mapping
# - Longest-first tokenization
# - Clause-level segmentation and one implied verb per clause
# - Diversity selection to avoid repetitive English
# - Robust rendering (no index errors)
# - Validation metrics + sensitivity test (+ optional BERT perplexity)
# - Entropy histogram figure saved to output/

import os, re, json, argparse, random
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- paths ----------------
BASE_DIR   = os.getcwd()
INPUT_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMG_DIR    = os.path.join(INPUT_DIR, "images")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- EVA cleaning ----------------
EVA_WORD = re.compile(r"^[acdefghiklmnopqrsty]+$")
SEED = re.compile(r"(qokaiin|qokeedy|qokedy|qokar|okar|okain|ykal|ydar|ykor|daiin|aiin|shedy|chedy|shor|chol|\bdy\b|\bar\b)")

BAD = {
    "last","edited","identification","title","first","page","folio","author","beinecke","yale",
    "rene","stolfi","landini","annotation","notes","copyright","translation","intro","index",
    "contents","section","table","figure","updated","revision"
}


def _folio_sort_cols(df, folio_col="folio"):
    def _parse(fid):
        s = str(fid).strip().lower()
        m = re.search(r"(\d+)\s*([rv]?)", s)
        if not m:
            return (10**9, 9, s)
        num = int(m.group(1))
        side = m.group(2) if m.group(2) in ("r","v") else ""
        side_ord = {"r":0, "v":1}.get(side, 2)
        return (num, side_ord, s)
    keys = df[folio_col].map(_parse)
    df = df.copy()
    df["_n"]  = keys.map(lambda t: t[0])
    df["_sv"] = keys.map(lambda t: t[1])
    df["_s"]  = keys.map(lambda t: t[2])
    df = df.sort_values(["_n","_sv","_s"]).drop(columns=["_n","_sv","_s"])
    return df

def _clean(s):
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"[^A-Za-z\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _eva_ratio(line):
    toks = [t for t in re.split(r"[.\s]+", line) if t]
    if not toks: return 0.0
    ok = sum(1 for t in toks if EVA_WORD.match(t))
    return ok / float(len(toks))

def _looks_english(s):
    toks = set(s.split())
    return any(w in toks for w in BAD)

def keep_eva_line(raw):
    if not SEED.search(raw or ""): return False
    s = _clean(raw)
    if not s or _looks_english(s): return False
    return _eva_ratio(s) >= 0.80

def sanitize_eva_field(text):
    lines = re.split(r"[\r\n]+", str(text))
    kept = [_clean(l) for l in lines if keep_eva_line(l)]
    if kept: return " ".join(kept)
    s = _clean(text)
    return s if (not _looks_english(s) and _eva_ratio(s) >= 0.80) else ""

# ---------------- context and mapping ----------------
CTX = {
    "astral":    {"68r","69r","70r","71r","72r","73r","75r"},
    "baths":     {"75r","78r","78v","85r"},
    "poultice":  {"99r","99v","100r","102r"},
    "closing":   {"116r","116v"},
}
def context_for_folio(fid):
    f = fid.lower()
    if f in CTX["astral"]:   return "astral"
    if f in CTX["poultice"]: return "poultice"
    if f in CTX["baths"]:    return "baths"
    if f in CTX["closing"]:  return "closing"
    return "botanical"

BASE_UNITS = {
    "cphy":"caput physis",
    "ydar":"vitalis dare aqua radialis",
    "ykor":"vitalis clavis oleum radialis",
    "ykal":"vitalis clavis aqua luna",
    "qokaiin":"quando oleum clavis aqua in",
    "qokain":"quando oleum clavis aqua in",
    "qokedy":"quando oleum clavis et dare vitalis",
    "qokeedy":"quando oleum clavis et dare vitalis",
    "qokar":"quando oleum clavis aqua radialis",
    "okain":"oleum clavis aqua in",
    "okar":"oleum clavis aqua radialis",
    "shedy":"sanguis humus et dare vitalis",
    "chedy":"caput humus et dare vitalis",
    "shor":"sanguis humus oleum radialis",
    "chol":"caput humus oleum luna",
    "daiin":"dare aqua in",
    "aiin":"aqua in",
    "ar":"aqua radialis",
    "paxar":"pax aqua radialis",
    "her":"herba",
    "dabas":"dare herba"
}
UNIT_ORDER = sorted(BASE_UNITS.keys(), key=len, reverse=True)

def make_char_map(ctx):
    k = "coquere" if ctx == "poultice" else "clavis"
    l = "lux" if ctx in {"astral"} else "luna"
    return {
        "q":"quando","o":"oleum","k":k,"e":"et","d":"dare","y":"vitalis","a":"aqua","i":"in",
        "s":"sanguis","h":"humus","c":"caput","t":"tempus","r":"radialis","l":l,"n":"",
        "f":"femina","p":"pax"
    }

def make_unit_map(ctx):
    k = "coquere" if ctx == "poultice" else "clavis"
    l = "lux" if ctx in {"astral"} else "luna"
    um = dict(BASE_UNITS)
    um["ykor"] = "vitalis %s oleum radialis" % k
    um["ykal"] = "vitalis %s aqua %s" % (k, l)
    um["qokaiin"] = "quando oleum %s aqua in" % k
    um["qokain"]  = "quando oleum %s aqua in" % k
    um["qokedy"]  = "quando oleum %s et dare vitalis" % k
    um["qokeedy"] = "quando oleum %s et dare vitalis" % k
    um["qokar"]   = "quando oleum %s aqua radialis" % k
    um["okain"]   = "oleum %s aqua in" % k
    um["okar"]    = "oleum %s aqua radialis" % k
    um["chol"]    = "caput humus oleum %s" % l
    return um

def write_mapping_json(out_dir, ctx):
    payload = {"context": ctx, "char_map": make_char_map(ctx), "unit_map": make_unit_map(ctx)}
    with open(os.path.join(out_dir, "mapping_%s.json" % ctx), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)

# ---------------- tokenization / clauses ----------------
STOP_ONCE = {"aqua","oleum","vitalis","radialis","caput","humus","sanguis"}

CLAUSE_ANCHOR = re.compile(r"\b(qok[a-z]+|daiin|aiin|shedy|chedy|chol|shor|okar|okain|ykal|ydar|ykor|dy|ar)\b")

def segment_clauses_eva(eva_text):
    eva_text = re.sub(r"\s+", " ", str(eva_text).strip())
    parts = CLAUSE_ANCHOR.split(eva_text)
    seps  = CLAUSE_ANCHOR.findall(eva_text)
    clauses = []
    for i, p in enumerate(parts):
        if not p.strip(): continue
        cue = seps[i] if i < len(seps) else ""
        clauses.append((cue, p.strip()))
    return clauses

def tokenize_word(word, unit_map, char_map):
    w = word.strip().lower()
    if not w: return []
    i, out = 0, []
    while i < len(w):
        matched = False
        for key in UNIT_ORDER:
            if w.startswith(key, i):
                out.extend(unit_map.get(key, key).split())
                i += len(key); matched = True; break
        if matched: continue
        ch = w[i]
        val = char_map.get(ch, "")
        if val: out.append(val)
        i += 1
    return out

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    return float(len(A & B)) / float(len(A | B))

def decode_eva_to_latin_clauses(eva_text, ctx):
    unit_map = make_unit_map(ctx)
    char_map = make_char_map(ctx)
    clauses = segment_clauses_eva(eva_text)
    out, seen = [], []
    for cue, span in clauses:
        toks = []
        for w in span.split():
            toks.extend(tokenize_word(w, unit_map, char_map))
        norm = []
        for t in toks:
            if norm and norm[-1] == t: continue
            norm.append(t)
        capped, counts = [], Counter()
        for t in norm:
            if t in STOP_ONCE:
                counts[t] += 1
                if counts[t] > 1: continue
            capped.append(t)
        if capped:
            if any(t in capped for t in ("quando","tempus","coquere","clavis")):
                capped.append("fac")
            else:
                capped.append("usa")
        if not capped: continue
        if seen:
            sim = max(jaccard(capped, s) for s in seen)
            if sim >= 0.88:
                continue
        out.append(capped); seen.append(capped)

    if not out and eva_text.strip():
        probe = []
        for w in eva_text.split()[:6]:
            probe.extend(tokenize_word(w, unit_map, char_map))
        if probe:
            if any(t in probe for t in ("quando","tempus","coquere","clavis")):
                probe.append("fac")
            else:
                probe.append("usa")
            out = [probe]
    return out

# ---------------- English rendering ----------------
PHRASE_GLOSS = {
    "quando":"when","tempus":"time","luna":"moon","lux":"light",
    "dare":"dose","usa":"Use","fac":"Do","aqua":"water","oleum":"oil",
    "sanguis":"blood","vitalis":"vital","caput":"head","humus":"earth",
    "radialis":"spread","clavis":"key","coquere":"cook","femina":"woman",
    "physis":"nature","pax":"peace","herba":"herb"
}

def render_clause(tokens):
    w = list(tokens)
    parts = []

    # condition
    if "quando" in w or "tempus" in w:
        cond = []
        if "aqua" in w: cond.append("water")
        if "oleum" in w: cond.append("oil")
        if "coquere" in w: cond.append("cook")
        if "clavis" in w: cond.append("key")
        if "luna" in w: cond.append("moon")
        if "lux" in w: cond.append("light")
        if cond:
            parts.append("When " + " ".join(cond) + ",")

    # action
    action = None
    if "dare" in w:
        objs = []
        if "vitalis" in w: objs.append("vital")
        if "aqua" in w:    objs.append("water")
        if "oleum" in w:   objs.append("oil")
        if "herba" in w:   objs.append("herb")
        if objs: action = "Dose " + " ".join(objs)
    if action is None:
        if "aqua" in w and "in" in w: action = "Apply water internally"
        elif "aqua" in w and "radialis" in w: action = "Spread water"
        elif "oleum" in w and "radialis" in w: action = "Spread oil"
        elif "oleum" in w and "in" in w: action = "Apply oil internally"
        elif "coquere" in w and "oleum" in w: action = "Cook oil"
        elif "aqua" in w: action = "Apply water"
        elif "oleum" in w: action = "Apply oil"
        else: action = "Procedure"
    parts.append(action)

    tail = len(parts) - 1
    if "luna" in w:   parts[tail] += " under moon"
    if "lux" in w:    parts[tail] += " by light"
    if "clavis" in w: parts[tail] += " with key"
    if "femina" in w and "caput" in w: parts[tail] += " on the woman head"
    if "humus" in w: parts[tail] += " with earth"
    if "sanguis" in w and "vitalis" in w: parts[tail] += " for vital blood"
    elif "sanguis" in w: parts[tail] += " for blood"
    if "physis" in w: parts[tail] += " for nature"

    sent = " ".join(parts).strip()
    if sent and not sent.endswith("."): sent += "."
    return sent[0].upper() + sent[1:] if sent else "Procedure."

def select_concise(clauses, k=12, diversity=0.7):
    cands = [render_clause(t) for t in clauses]
    if not cands: return []
    tokens_all = " ".join(cands).lower().split()
    freq = Counter(tokens_all)
    chosen, used_ngrams = [], set()

    def score(s):
        toks = s.lower().split()
        if not toks: return -1e9
        rarity = sum(1.0/(1+freq[t]) for t in toks) / float(len(toks))
        ngrams = {" ".join(toks[i:i+3]) for i in range(max(0, len(toks)-2))}
        overlap = len(ngrams & used_ngrams)
        return rarity - 0.15*overlap - 0.02*len(toks)

    pool = sorted(cands, key=score, reverse=True)
    for s in pool:
        if len(chosen) >= k: break
        toks = s.lower().split()
        ngrams = {" ".join(toks[i:i+3]) for i in range(max(0, len(toks)-2))}
        if len(ngrams) > 0:
            frac = float(len(ngrams & used_ngrams)) / float(len(ngrams))
            if frac > (1 - diversity):
                continue
        chosen.append(s)
        used_ngrams |= ngrams
    return chosen

# ---------------- metrics ----------------
MED_LEX = {"aqua","oleum","herba","sanguis","caput","luna","lux","humus",
           "tempus","radialis","femina","dare","quando","clavis","vitalis","in",
           "et","pax","coquere","physis"}

def entropy(counter):
    total = sum(counter.values())
    if total == 0: return 0.0
    p = np.array(list(counter.values()), dtype=float) / float(total)
    return float(-np.sum(p * np.log2(p + 1e-12)))

def bigrams(seq): return list(zip(seq, seq[1:]))

def top10_conc(bi):
    total = float(sum(bi.values()) or 1)
    return float(sum(c for _, c in Counter(bi).most_common(10))) / total

def shuffle_baseline(tokens, trials=200, seed=13):
    rnd = random.Random(seed)
    conc = []
    toks = list(tokens)
    for _ in range(trials):
        rnd.shuffle(toks)
        conc.append(top10_conc(Counter(bigrams(toks))))
    arr = np.array(conc, dtype=float)
    mu = float(arr.mean()); sd = float(arr.std())
    if sd <= 1e-9: sd = 1e-9
    return mu, sd

def compute_metrics(tokens, kept):
    uni = Counter(tokens); bi = Counter(bigrams(tokens))
    H1 = entropy(uni); H2 = entropy(bi)
    ttr = len(uni) / float(len(tokens) or 1)
    lex = sum(1 for t in tokens if t in MED_LEX) / float(len(tokens) or 1)
    obs = top10_conc(bi); mu, sd = shuffle_baseline(tokens)
    z = (obs - mu) / sd
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return {
        "unigram_entropy_bits": round(H1,4),
        "bigram_entropy_bits": round(H2,4),
        "type_token_ratio": round(ttr,4),
        "lexicon_alignment": round(lex,4),
        "top10_bigram_conc": round(obs,4),
        "baseline_mean_conc": round(mu,4),
        "baseline_std_conc": round(sd,6),
        "z_score": round(z,3),
        "p_value_one_tailed": round(p,6),
        "clauses_kept": len(kept)
    }

# ---------------- sensitivity test ----------------
def sensitivity_test(eva_text, folio_id, drop_ratio=0.10):
    ctx = context_for_folio(folio_id)
    clauses_ref = decode_eva_to_latin_clauses(eva_text, ctx)
    base_toks = [t for cl in clauses_ref for t in cl]
    base_H1 = entropy(Counter(base_toks))

    unit_map = make_unit_map(ctx)
    keys = list(unit_map.keys())
    if not keys or base_H1 == 0: return 0.0
    n_drop = max(1, int(len(keys)*drop_ratio))
    bad = random.sample(keys, n_drop)

    # permute selected units
    vals = [unit_map[k] for k in bad]
    random.shuffle(vals)
    for k, v in zip(bad, vals):
        unit_map[k] = v

    char_map = make_char_map(ctx)
    pert_toks = []
    for w in eva_text.split():
        # tokenization with perturbed map
        w = w.strip().lower()
        i = 0
        while i < len(w):
            matched = False
            for key in UNIT_ORDER:
                if w.startswith(key, i):
                    pert_toks.extend(unit_map.get(key, key).split())
                    i += len(key); matched = True; break
            if matched: continue
            ch = w[i]; val = char_map.get(ch, "")
            if val: pert_toks.append(val)
            i += 1
    pert_H1 = entropy(Counter(pert_toks))
    if base_H1 <= 0: return 0.0
    drop = (base_H1 - pert_H1) / base_H1
    return round(max(0.0, drop)*100.0, 2)

# ---------------- optional BERT perplexity ----------------
# ---- optional BERT perplexity (quiet + cached) ----
_BERT_CACHE = {"tok": None, "mdl": None}

def bert_perplexity_or_none(text, model_name="distilbert-base-uncased"):
    try:
        # Silence Transformers’ info/warning logs
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch

        # Load once, cache for all folios
        if _BERT_CACHE["tok"] is None or _BERT_CACHE["mdl"] is None:
            _BERT_CACHE["tok"] = AutoTokenizer.from_pretrained(model_name)
            _BERT_CACHE["mdl"] = AutoModelForMaskedLM.from_pretrained(
                model_name,
                # not strictly required, but keeps loaders quiet for edge cases
                ignore_mismatched_sizes=True
            )
            _BERT_CACHE["mdl"].eval()

        tok = _BERT_CACHE["tok"]
        mdl = _BERT_CACHE["mdl"]

        # Cap length to keep it snappy and avoid GPU/CPU memory spikes
        enc = tok(text[:4000], return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc["input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return float(torch.exp(loss))
    except Exception:
        # If transformers/torch is not available, just skip
        return None


# ---------------- histogram ----------------
def plot_entropy_histogram(met_df, out_dir):
    if met_df.empty or "z_score" not in met_df.columns: return
    bins = list(range(-5, 61, 5))
    z = np.clip(met_df["z_score"].values, -5, 60)
    plt.figure(figsize=(6,4))
    plt.hist(z, bins=bins, edgecolor="black", alpha=0.8)
    plt.axvline(20, color="orange", linestyle="--", label="10-20 range")
    plt.axvline(30, color="red", linestyle="--", label=">30 high coherence")
    plt.xlabel("Z-score (Top-10 Bigram Concentration)")
    plt.ylabel("Folios")
    plt.title("Entropy / Concentration Z-score Distribution")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "entropy_histogram.png")
    plt.savefig(path, dpi=200); plt.close()
    print("[OK] histogram -> %s" % path)

# ---------------- driver ----------------
def decode_one(eva_text, folio_id):
    ctx = context_for_folio(folio_id)
    clauses = decode_eva_to_latin_clauses(eva_text, ctx)
    latin_tokens = [t for cl in clauses for t in cl]
    english_sents = select_concise(clauses, k=12, diversity=0.7)
    metrics = compute_metrics(latin_tokens, clauses)
    metrics["sensitivity_drop_pct"] = sensitivity_test(eva_text, folio_id)
    # optional BERT
    eng_text = " ".join(english_sents)
    ppl = bert_perplexity_or_none(eng_text)
    if ppl is not None:
        metrics["bert_perplexity"] = ppl
    return ctx, " ".join(latin_tokens), eng_text, metrics

def run_pipeline(in_csv=None, out_dir=None, folio=None):
    in_csv  = in_csv  or os.path.join(INPUT_DIR, "voynich_full_transcription.csv")
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    if "folio" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must have columns: folio, text")

    df["folio"] = df["folio"].astype(str).str.strip().str.lower()
    df["text"]  = df["text"].apply(sanitize_eva_field)
    df = df[df["text"].str.len() > 0]
    if folio:
        df = df[df["folio"] == folio.lower()]
        if df.empty: raise ValueError("Folio %s not found" % folio)

    dec_rows, met_rows, seen_ctx = [], [], set()
    for _, row in df.iterrows():
        fid, eva = row["folio"], row["text"]
        ctx, latin, english, metrics = decode_one(eva, fid)
        dec_rows.append({"folio": fid, "context": ctx, "eva": eva, "latin": latin, "english": english})
        m = {"folio": fid, "context": ctx}; m.update(metrics); met_rows.append(m)

        glyphs = re.findall(r"[a-z]", eva)
        if glyphs:
            freq = pd.Series(glyphs).value_counts()
            ax = freq.plot(kind="bar", title="Glyph Frequency %s" % fid, figsize=(6,3))
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(out_dir, "freq_%s.png" % fid), dpi=150)
            ax.figure.clf()

        if ctx not in seen_ctx:
            write_mapping_json(out_dir, ctx); seen_ctx.add(ctx)

    out_dec = os.path.join(out_dir, "decoded_folios.csv")
    out_met = os.path.join(out_dir, "metrics.csv")
    dec_df = pd.DataFrame(dec_rows); met_df = pd.DataFrame(met_rows)

    dec_df = _folio_sort_cols(dec_df, "folio")
    met_df = _folio_sort_cols(met_df, "folio")

    dec_df.to_csv(out_dec, index=False); met_df.to_csv(out_met, index=False)
    print("[OK] decoded -> %s" % out_dec)
    print("[OK] metrics -> %s" % out_met)

    plot_entropy_histogram(met_df, out_dir)
    return out_dec, out_met

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voynich pipeline (CMM v6, sensitivity, histogram, optional BERT)")
    ap.add_argument("--in_csv", default=None, help="Input CSV path (default: input/voynich_full_transcription.csv)")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: ./output)")
    ap.add_argument("--folio", default=None, help="Decode only one folio, e.g., 1r")
    args = ap.parse_args()
    run_pipeline(args.in_csv, args.out_dir, args.folio)
