# fac/usa + 3-tier translations + metrics
import os, re, argparse, json, random
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR   = os.getcwd()
INPUT_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- EVA cleaning (strict) --------
SEED_PATTERNS = re.compile(
    r"(qokaiin|qokeedy|qokedy|qokar|ykal|ydar|ykor|daiin|aiin|shedy|chedy|shor|chol|okar|okain|\bdy\b|\bar\b)"
)
EVA_TOKEN = re.compile(r"^[acdefghiklmnopqrsty]+$")

BAD_WORDS = {
    "last","edited","identification","title","first","page","folio","author","beinecke",
    "yale","rene","stolfi","landini","annotation","notes","copyright","translation",
    "intro","index","contents","section","table","figure","updated","revision"
}
EVA_ANCHORS = {
    "daiin","aiin","qokaiin","qokain","qokeedy","qokedy","qokar","okar","okain","shedy",
    "chedy","shor","chol","ykal","dy","ar"
}

# -------- mapping.json for transparency --------
def write_mapping_json(out_dir, k_value="clavis"):
    char_map = {
        "q": "quando", "o": "oleum", "k": "clavis / coquere", "e": "et",
        "d": "dare", "y": "vitalis", "a": "aqua", "i": "in", "s": "sanguis",
        "h": "humus / silent", "c": "caput", "t": "tempus", "r": "radialis",
        "l": "luna / lux", "n": "silent extender", "f": "femina", "p": "pax"
    }
    unit_map = {
        "ydar": "vitalis dare aqua radialis",
        "ykor": "vitalis clavis oleum radialis"
    }
    mapping = {"char_map": char_map, "unit_map": unit_map, "k_value": k_value}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mapping.json"), "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=True, indent=2)
    print("[OK] mapping.json written to %s" % out_dir)

# -------- helpers: cleaning --------
def clean_text(line):
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"\[[^\]]+\]", " ", line)
    line = re.sub(r"[^a-zA-Z\s\.]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line.lower()

def looks_like_english(s):
    toks = set(s.split())
    return any(w in toks for w in BAD_WORDS)

def eva_purity_ratio(s):
    toks = [t for t in re.split(r"[.\s]+", s) if t]
    if not toks: return 0.0
    eva_like = sum(1 for t in toks if EVA_TOKEN.match(t))
    return eva_like / float(len(toks))

def has_eva_anchor(s):
    return any(a in s for a in EVA_ANCHORS)

def is_eva_line(raw_line):
    if not SEED_PATTERNS.search(raw_line): return False
    s = clean_text(raw_line)
    if not s or looks_like_english(s): return False
    if eva_purity_ratio(s) < 0.8: return False
    if not has_eva_anchor(s): return False
    return True

def sanitize_eva_field(text):
    lines = re.split(r"[\r\n]+", str(text))
    kept = [clean_text(l) for l in lines if is_eva_line(l)]
    if kept: return " ".join(kept)
    s = clean_text(str(text))
    return s if (not looks_like_english(s) and eva_purity_ratio(s) >= 0.8 and has_eva_anchor(s)) else ""

# -------- mapping (decoding) --------
def make_char_map(k_value="clavis"):
    return {
        "q":"quando","o":"oleum","k":k_value,"e":"et","d":"dare","y":"vitalis","a":"aqua","i":"in",
        "s":"sanguis","h":"humus","c":"caput","t":"tempus","r":"radialis","l":"luna","n":"",
        "f":"femina","p":"pax"
    }

def make_unit_map(k_value="clavis"):
    k = k_value
    return {
        "ydar":"vitalis dare aqua radialis",
        "ykor":"vitalis " + k + " oleum radialis",
        "qokaiin":"quando oleum " + k + " aqua in",
        "qokeedy":"quando oleum " + k + " et dare vitalis",
        "qokain":"quando oleum " + k + " aqua in",
        "qokedy":"quando oleum " + k + " et dare vitalis",
        "qokar":"quando oleum " + k + " aqua radialis",
        "okain":"oleum " + k + " aqua in",
        "okar":"oleum " + k + " aqua radialis",
        "shedy":"sanguis humus et dare vitalis",
        "chedy":"caput humus et dare vitalis",
        "shor":"sanguis humus oleum radialis",
        "chol":"caput humus oleum luna",
        "ykal":"vitalis " + k + " aqua luna",
        "daiin":"dare aqua in",
        "aiin":"aqua in",
        "dy":"dare vitalis",
        "ar":"aqua radialis",
        "her":"herba",
        "paxar":"pax aqua radialis"
    }

# validation lexicon
MED_LEXICON = {
    "aqua","oleum","herba","sanguis","caput","luna","humus","tempus","radialis","femina",
    "dare","quando","clavis","vitalis","in","et","pax","coquere"
}

# -------- tokenizer and clause decoding --------
def tokenizer(line, unit_map, char_map):
    tokens = []
    for word in str(line).split():
        w = word.strip().lower()
        if not w: continue
        if w in unit_map and unit_map[w] is not None:
            tokens.extend(unit_map[w].split()); continue
        i = 0
        while i < len(w):
            matched = False
            for L in range(min(7, len(w)-i), 1, -1):
                seg = w[i:i+L]
                if seg in unit_map and unit_map[seg] is not None:
                    tokens.extend(unit_map[seg].split()); i += L; matched = True; break
            if matched: continue
            ch = w[i]
            val = char_map.get(ch, "")
            if val != "": tokens.append(val)
            i += 1
    return tokens

CLAUSE_SPLIT = re.compile(r"\b(?:daiin|qok[a-z]+|chol|sh[a-z]+|okar|okain|dy|ar)\b")
STOP_LEMMAS = {"aqua","oleum","vitalis","radialis","caput","humus","sanguis"}
STOP_CAP = 1  # cap per clause

def segment_clauses(eva_text):
    eva_text = re.sub(r"\s+", " ", str(eva_text).strip())
    parts = CLAUSE_SPLIT.split(eva_text)
    seps  = CLAUSE_SPLIT.findall(eva_text)
    clauses = []
    for i, p in enumerate(parts):
        if not p.strip(): continue
        cue = seps[i] if i < len(seps) else ""
        clauses.append((cue.strip(), p.strip()))
    return clauses

def smooth_tokens(tokens, mode="none"):
    if mode == "none": return tokens
    out, last_fac, last_usa, span = [], False, False, 0
    for t in tokens:
        out.append(t); span += 1
        if mode == "light":
            if t == "dare" and not last_fac:
                out.append("fac"); last_fac = True; span = 0
            elif t in {"aqua","oleum","vitalis"} and not last_usa:
                out.append("usa"); last_usa = True; span = 0
            if span >= 8: last_fac = False; last_usa = False
        elif mode == "full":
            if t in {"quando","dare","aqua","oleum"}: out.append("fac")
            if t in {"vitalis","aqua","oleum"}: out.append("usa")
    return out

def decode_clause(eva_clause, unit_map, char_map, smoothing="none"):
    toks = tokenizer(eva_clause, unit_map, char_map)
    if smoothing != "none": toks = smooth_tokens(toks, mode=smoothing)
    # remove immediate repeats
    norm = []
    for t in toks:
        if norm and norm[-1] == t: continue
        norm.append(t)
    # cap over-used lemmas
    capped, counts = [], Counter()
    for t in norm:
        if t in STOP_LEMMAS:
            counts[t] += 1
            if counts[t] > STOP_CAP: continue
        capped.append(t)
    return capped

def decode_eva_to_latin_clauses(eva_text, k_value="clavis", smoothing="none", jaccard_drop=0.88):
    char_map = make_char_map(k_value)
    unit_map = make_unit_map(k_value)
    clauses = segment_clauses(eva_text)
    out, seen_sets = [], []
    for cue, span in clauses:
        toks = decode_clause(span, unit_map, char_map, smoothing=smoothing)
        sset = set(toks)
        if seen_sets:
            sim = max(len(sset & s) / float(len(sset | s)) for s in seen_sets) if sset else 0.0
            if sim >= jaccard_drop: continue
        if toks:
            out.append(toks)
            seen_sets.append(sset)
    return out  # list of token lists

# -------- Strange Latin (fac/usa), Full English, Concise English --------
def infer_implied_verb(tokens, context="botanical"):
    if any(t in tokens for t in ("quando","tempus","clavis","coquere")):
        return "fac"
    if any(t in tokens for t in ("aqua","oleum","vitalis","herba")):
        return "usa"
    return "fac"

def strange_latin_from_clauses(clauses_tokens, context="botanical"):
    sents = []
    for toks in clauses_tokens:
        verb = infer_implied_verb(toks, context=context)
        line = " ".join(toks + [verb]).strip()
        if line:
            s = line[0].upper() + line[1:]
            if not s.endswith("."): s += "."
            sents.append(s)
    return sents

def english_full_sentence(tokens):
    # condition
    cond_bits = []
    if "quando" in tokens: cond_bits.append("when")
    if "tempus" in tokens: cond_bits.append("at the right time")
    if "luna" in tokens:   cond_bits.append("under the moon")
    if "clavis" in tokens: cond_bits.append("with the key step")
    cond = ""
    if cond_bits:
        # Merge distinct condition phrases
        cond = " ".join(sorted(set(cond_bits)))
        cond = cond[0].upper() + cond[1:] + ", "

    # choose main action from content
    action = None
    if "dare" in tokens:
        objs = []
        if "vitalis" in tokens: objs.append("vital")
        if "aqua" in tokens:    objs.append("water")
        if "oleum" in tokens:   objs.append("oil")
        if "herba" in tokens:   objs.append("herb")
        if objs: action = "Dose " + " ".join(objs)
    if action is None:
        if "aqua" in tokens and "in" in tokens:
            action = "Apply water internally"
        elif "aqua" in tokens and "radialis" in tokens:
            action = "Spread water"
        elif "oleum" in tokens and "radialis" in tokens:
            action = "Spread oil"
        elif "oleum" in tokens:
            action = "Apply oil"
        elif "aqua" in tokens:
            action = "Apply water"
        else:
            action = "Perform the procedure"

    # modifiers (dedup + order)
    mods = []
    if "femina" in tokens and "caput" in tokens: mods.append("on the woman's head")
    if "humus" in tokens: mods.append("with earth")
    if "sanguis" in tokens and "vitalis" in tokens: mods.append("for vital blood")
    elif "sanguis" in tokens: mods.append("for blood")
    # if condition already carries luna/tempus, do not repeat
    if "luna" in tokens and "under the moon" not in cond.lower(): mods.append("under the moon")
    if "tempus" in tokens and "right time" not in cond.lower():   mods.append("at the right time")

    # compose
    sent = cond + action
    if mods:
        sent += " " + ", ".join(mods)
    if not sent.endswith("."): sent += "."
    return sent


def english_full_from_clauses(clauses_tokens):
    return [english_full_sentence(toks) for toks in clauses_tokens]

def english_concise_from_full(full_sents, k_max=12, min_diff=0.35):
    """
    Make a short, diverse, imperative list.
    k_max: cap how many lines we keep.
    min_diff: drop items too similar (Jaccard over unigrams).
    """
    def toks(s): return set(w for w in re.split(r"\W+", s.lower()) if w)

    curated = []
    for s in full_sents:
        # strip helpers
        s2 = re.sub(r"\b(Use|Apply|Spread|Dose|Perform|Do)\b\s*", "", s, flags=re.I)
        s2 = re.sub(r"\b(the|a|an)\b\s*", "", s2, flags=re.I).strip()
        if not s2.endswith("."): s2 += "."
        # similarity filter
        T = toks(s2)
        if curated and max(len(T & toks(x)) / float(len(T | toks(x))) for x in curated) > (1.0 - min_diff):
            continue
        curated.append(s2)
        if len(curated) >= k_max: break
    return curated


# -------- Metrics --------
def entropy(counter):
    total = sum(counter.values())
    if total == 0: return 0.0
    p = np.array(list(counter.values()), dtype=float) / float(total)
    return float(-np.sum(p * np.log2(p)))

def bigrams(seq):
    return list(zip(seq, seq[1:]))

def top10_concentration(bi_counter):
    total = sum(bi_counter.values())
    if total == 0: return 0.0
    top10 = sum(c for _, c in bi_counter.most_common(10))
    return float(top10) / float(total)

def shuffle_baseline(tokens, trials=200, seed=13):
    rnd = random.Random(seed)
    conc = []
    toks = list(tokens)
    for _ in range(trials):
        rnd.shuffle(toks)
        bi = Counter(bigrams(toks))
        conc.append(top10_concentration(bi))
    arr = np.array(conc, dtype=float)
    return float(arr.mean()), float(arr.std() if arr.std() > 1e-9 else 1e-9)

def compute_validation(tokens, clauses_tokens):
    toks = list(tokens)
    uni = Counter(toks)
    bi  = Counter(bigrams(toks))
    H1  = entropy(uni)
    H2  = entropy(bi)
    ttr = len(uni) / float(len(toks) or 1)
    lex_hits = sum(1 for t in toks if t in MED_LEXICON)
    lex_rate = lex_hits / float(len(toks) or 1)
    obs_conc = top10_concentration(bi)
    mu, sigma = shuffle_baseline(toks, trials=200, seed=42)
    z = (obs_conc - mu) / sigma
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return {
        "unigram_entropy_bits": round(H1, 4),
        "bigram_entropy_bits": round(H2, 4),
        "type_token_ratio": round(ttr, 4),
        "lexicon_alignment": round(lex_rate, 4),
        "top10_bigram_conc": round(obs_conc, 4),
        "baseline_mean_conc": round(mu, 4),
        "baseline_std_conc": round(sigma, 6),
        "z_score": round(z, 3),
        "p_value_one_tailed": round(p, 6),
        "clauses_kept": len(clauses_tokens)
    }

# -------- End-to-end decode for one EVA string --------
def decode_text_with_metrics(eva_text, k_value="clavis", smoothing="none", context="botanical"):
    clauses_tokens = decode_eva_to_latin_clauses(eva_text, k_value=k_value, smoothing=smoothing)
    latin_tokens = [t for clause in clauses_tokens for t in clause]

    # tier 1: strange latin
    strange_latin_sents = strange_latin_from_clauses(clauses_tokens, context=context)

    # tier 2: full english
    full_eng_sents = english_full_from_clauses(clauses_tokens)

    # tier 3: concise english
    concise_sents = english_concise_from_full(full_eng_sents)

    metrics = compute_validation(latin_tokens, clauses_tokens)

    latin_linear = " ".join(latin_tokens)
    strange_latin = " ".join(strange_latin_sents)
    english_full  = " ".join(full_eng_sents)
    english_short = " ".join(concise_sents)
    return latin_linear, strange_latin, english_full, english_short, metrics

# -------- Pipeline --------
def run_pipeline(in_csv=None, out_dir=None, folio=None, k_value="clavis", smoothing="none", context="botanical"):
    in_csv  = in_csv  or os.path.join(INPUT_DIR,  "voynich_full_transcription.csv")
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    write_mapping_json(out_dir, k_value=k_value)

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    df["folio"] = df["folio"].astype(str).str.lower()
    df["text"]  = df["text"].apply(sanitize_eva_field)
    df = df[df["text"].str.len() > 0]
    if folio:
        df = df[df["folio"] == folio.lower()]
        if df.empty: raise ValueError("Folio %s not found in %s" % (folio, in_csv))

    dec_rows, met_rows = [], []
    for _, row in df.iterrows():
        f, t = row["folio"], row["text"]
        latin, s_lat, eng_full, eng_short, metrics = decode_text_with_metrics(
            t, k_value=k_value, smoothing=smoothing, context=context
        )
        dec_rows.append({
            "folio": f,
            "eva": t,
            "latin_roots": latin,
            "latin_strange": s_lat,
            "english_full": eng_full,
            "english_concise": eng_short,
            "english": eng_short  # legacy column for older PDF scripts
        })
        m = {"folio": f}; m.update(metrics); met_rows.append(m)

        # glyph frequency figure
        glyphs = re.findall(r"[a-z]", t)
        if glyphs:
            freq = pd.Series(glyphs).value_counts()
            ax = freq.plot(kind="bar", title="Glyph Frequency %s" % f, figsize=(6,3))
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(out_dir, "freq_%s.png" % f), dpi=150)
            ax.figure.clf()

        # validation figure: observed vs baseline mean
        try:
            obs = metrics["top10_bigram_conc"]; mu = metrics["baseline_mean_conc"]
            fig_path = os.path.join(out_dir, "validation_%s.png" % f)
            plt.figure(figsize=(4,3))
            plt.bar(["observed","baseline"], [obs, mu])
            plt.title("Top-10 Bigram Concentration")
            plt.ylabel("fraction of bigrams")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
        except Exception:
            pass

    out_dec = os.path.join(out_dir, "decoded_folios.csv")
    out_met = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame(dec_rows).to_csv(out_dec, index=False)
    pd.DataFrame(met_rows).to_csv(out_met, index=False)
    print("[OK] decoded -> %s" % out_dec)
    print("[OK] metrics -> %s" % out_met)
    return out_dec, out_met

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voynich pipeline (strict EVA, 3-tier translation, metrics)")
    parser.add_argument("--in_csv", default=None, help="Input CSV (default: input/voynich_full_transcription.csv)")
    parser.add_argument("--out_dir", default=None, help="Output dir (default: ./output)")
    parser.add_argument("--folio", default=None, help="Single folio id (e.g. 1r)")
    parser.add_argument("--k", choices=["clavis","coquere"], default="clavis", help="k mapping variant")
    parser.add_argument("--smoothing", choices=["none","light","full"], default="none", help="smoothing intensity")
    parser.add_argument("--context", choices=["botanical","biological","astronomical","procedural"], default="botanical",
                        help="Context guides implied verb fac/usa")
    args = parser.parse_args()
    run_pipeline(args.in_csv, args.out_dir, args.folio, args.k, args.smoothing, args.context)
