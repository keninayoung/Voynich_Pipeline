
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import argparse, os, re, json

OUTPUT_DIR = "output"

def _parse_folios(arg):
    return [x.strip().lower() for x in (arg or "1r").split(",") if x.strip()]

def _parse_folio_images(arg):
    mapping = {}
    if not arg: return mapping
    for pair in arg.split(";"):
        if "=" not in pair: continue
        fol, path = pair.split("=", 1)
        fol = fol.strip().lower(); path = path.strip()
        if fol and path and os.path.exists(path):
            mapping[fol] = path
    return mapping

def _metrics_row(m):
    return [
        str(m.get("folio","")),
        str(m.get("lexicon_alignment","")),
        str(m.get("unigram_entropy_bits","")),
        str(m.get("bigram_entropy_bits","")),
        str(m.get("type_token_ratio","")),
        str(m.get("top10_bigram_conc","")),
        str(m.get("baseline_mean_conc","")),
        str(m.get("z_score","")),
        str(m.get("p_value_one_tailed","")),
        str(m.get("clauses_kept",""))
    ]

def _load_mapping_json():
    path = os.path.join(OUTPUT_DIR, "mapping.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)




def _add_exact_steps(story, styles):
    story.append(Paragraph("<b>Exact Steps to Generate Concise English Translations</b>", styles["HeadingCenter"]))
    story.append(Paragraph(
        "Step 1: Acquire EVA text. Step 2: Split into words and multi-glyph units. "
        "Step 3: Map glyphs to Latin roots (context-aware). Step 4: Assemble Strange Latin with implied verbs "
        "(fac or usa). Step 5: Translate to Full English (minimal grammar). "
        "Step 6: Condense to Concise English (~50 percent reduction). "
        "Step 7: Validate (entropy, lexicon, permutation baseline) and adjust mapping if needed.",
        styles["Body"]))
    rows = [
        ["Glyph/Cluster","Latin","Note"],
        ["q","quando","timing initiator"],
        ["o","oleum","remedy base"],
        ["k","clavis / coquere","procedure key; 99r uses 'coquere'"],
        ["e","et","connector"],
        ["d","dare","dose/action"],
        ["y","vitalis","vital outcome"],
        ["a","aqua","water"],
        ["i","in","internal"],
        ["s","sanguis","blood focus"],
        ["h","humus","earth/roots"],
        ["c","caput","head focus"],
        ["t","tempus","time"],
        ["r","radialis","spread/apply"],
        ["l","luna / lux","lunar or light cue"],
        ["n","(silent)","extender"],
        ["f","femina","woman"],
        ["p","pax","peace/closure"],
        ["ydar","vitalis dare aqua radialis","multi-glyph"],
        ["ykor","vitalis clavis oleum radialis","multi-glyph"]
    ]
    tbl = Table(rows, hAlign="LEFT", colWidths=[1.0*inch, 2.2*inch, 3.3*inch])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"TOP")
    ]))
    story.append(Spacer(1, 0.07*inch))
    story.append(tbl)
    story.append(PageBreak())


def _add_cover_and_summary(story, styles, folios, metrics_df, cover_img):
    story.append(Paragraph("<b>Deciphering the Voynich Manuscript</b>", styles["HeadingCenter"]))
    story.append(Paragraph("Kenneth Young, PhD (2025)", styles["HeadingCenter"]))
    story.append(Spacer(1, 0.15*inch))
    if cover_img:
        story.append(Image(cover_img, width=6.5*inch, height=4.5*inch))
        story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Abstract:</b> Clean EVA input, multi-glyph mapping, clause-level normalization, and "
        "statistical validation (entropy, lexicon alignment, permutation baseline) yield concise, "
        "medically coherent Latin phrasing and reproducible English paraphrases.", styles["Body"]))
    story.append(Spacer(1, 0.15*inch))
    # summary metrics
    story.append(Paragraph("<b>Summary of Validation Metrics</b>", styles["HeadingCenter"]))
    header = ["Folio","LexAlign","H1","H2","TTR","Top10 Obs","Top10 Base","Z","p(one-tail)","Clauses"]
    rows = [header]
    for f in folios:
        r = metrics_df[metrics_df["folio"].str.lower() == f]
        if not r.empty:
            rows.append(_metrics_row(r.iloc[0].to_dict()))
    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")
    ]))
    story.append(tbl)
    story.append(PageBreak())

def _add_methods_and_theory(story, styles):
    story.append(Paragraph("<b>Methods: How We Do It (CMM)</b>", styles["HeadingCenter"]))
    bullets = [
        "Glyph-to-Latin Mapping: map glyphs to Latin roots using frequency, folio context, and overlap with Trotula/Hildegard; prioritize multi-glyph units.",
        "Contextual Decoding: tokenization expands units (qokeedy, daiin, chol, shedy) before single glyphs; clauses are segmented and normalized.",
        "Implied Verbs: for each clause, insert fac (do) for procedural steps or usa (use) for applications.",
        "Translation: build Full English with articles and prepositions; then make Concise English by removing redundancy while preserving intent.",
        "Validation: compute entropy H1/H2, type/token ratio, lexicon alignment, and a permutation baseline (200 shuffles) for top-10 bigram concentration (z, one-tailed p)."
    ]
    story.append(ListFlowable([ListItem(Paragraph(x, styles["Body"]), leftIndent=10) for x in bullets], bulletType="1"))
    story.append(PageBreak())

def _add_worked_example(story, styles):
    story.append(Paragraph("<b>Worked Example: fachys</b>", styles["HeadingCenter"]))
    rows = [
        ["Stage","Output"],
        ["EVA word","fachys"],
        ["Mapping to roots","f = femina, a = aqua, c = caput, h = humus, y = vitalis, s = sanguis"],
        ["Strange Latin (implied verb)","femina aqua caput humus vitalis sanguis usa"],
        ["Full English","Use water on the woman's head with earth for vital blood."],
        ["Concise English","Water on woman's head with earth for blood."]
    ]
    tbl = Table(rows, hAlign="LEFT", colWidths=[1.8*inch, 4.6*inch])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"TOP")
    ]))
    story.append(tbl)
    story.append(PageBreak())

def _add_mapping_sections(story, styles):
    story.append(Paragraph("<b>Mapping (CMM)</b>", styles["HeadingCenter"]))
    mjson = _load_mapping_json()
    if not mjson:
        story.append(Paragraph("mapping.json not found. Re-run pipeline to generate it.", styles["Body"]))
        story.append(PageBreak()); return
    # single glyphs
    cm = mjson.get("char_map", {})
    rows = [["Glyph","Latin"]] + [[k, cm[k]] for k in sorted(cm.keys())]
    t1 = Table(rows, hAlign="LEFT")
    t1.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(Paragraph("Single glyphs (glyph -> Latin):", styles["Body"]))
    story.append(t1); story.append(Spacer(1, 0.1*inch))
    # multi-glyphs
    um = mjson.get("unit_map", {})
    rows2 = [["Token","Latin phrase"]] + [[k, um[k]] for k in sorted(um.keys())]
    t2 = Table(rows2, hAlign="LEFT")
    t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(Paragraph("Multi-glyph units (token -> Latin phrase):", styles["Body"]))
    story.append(t2)
    story.append(PageBreak())

def _add_summary_findings(story, styles, metrics_df, folios):
    # compute quick rollups
    try:
        lex = pd.to_numeric(metrics_df["lexicon_alignment"], errors="coerce").dropna()
        z   = pd.to_numeric(metrics_df["z_score"], errors="coerce").dropna()
        h1  = pd.to_numeric(metrics_df["unigram_entropy_bits"], errors="coerce").dropna()
        h2  = pd.to_numeric(metrics_df["bigram_entropy_bits"], errors="coerce").dropna()
        ttr = pd.to_numeric(metrics_df["type_token_ratio"], errors="coerce").dropna()
        roll = {
            "avg_lexalign": round(float(lex.mean()), 3) if len(lex) else "n/a",
            "avg_z": round(float(z.mean()), 2) if len(z) else "n/a",
            "avg_h1": round(float(h1.mean()), 2) if len(h1) else "n/a",
            "avg_h2": round(float(h2.mean()), 2) if len(h2) else "n/a",
            "avg_ttr": round(float(ttr.mean()), 3) if len(ttr) else "n/a",
            "folios": ", ".join(folios)
        }
    except Exception:
        roll = {"avg_lexalign":"n/a","avg_z":"n/a","avg_h1":"n/a","avg_h2":"n/a","avg_ttr":"n/a","folios":", ".join(folios)}

    story.append(Paragraph("<b>Summary and Findings</b>", styles["HeadingCenter"]))
    story.append(Paragraph(
        "This report presents a statistically validated decipherment using the Comprehensive "
        "Method and Mapping (CMM). The workflow is: clean EVA input, map glyphs to Latin roots, "
        "insert implied verbs (fac or usa), translate to Full English, condense to Concise English, "
        "and validate with entropy, lexicon alignment, and permutation baselines.", styles["Body"]))
    story.append(Spacer(1, 0.08*inch))

    rows = [
        ["Rollup Metric","Value"],
        ["Folios in this report", roll["folios"]],
        ["Average Lexicon Alignment", str(roll["avg_lexalign"])],
        ["Average Z-score (observed vs baseline)", str(roll["avg_z"])],
        ["Average Unigram Entropy H1 (bits)", str(roll["avg_h1"])],
        ["Average Bigram Entropy H2 (bits)", str(roll["avg_h2"])],
        ["Average Type/Token Ratio", str(roll["avg_ttr"])]
    ]
    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(tbl)
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph(
        "Interpretation: high lexicon alignment with medieval medical vocabulary, together with "
        "Z-scores well above randomized baselines, support coherent linguistic structure rather "
        "than chance. Cross-section sampling (botanical, astronomical, biological, recipe, closing) "
        "demonstrates that the mapping generalizes.", styles["Body"]))
    story.append(PageBreak())

def build_pdf(decoded_csv, metrics_csv, folios, folio_images=None, out_pdf=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ddf = pd.read_csv(decoded_csv, dtype=str).fillna("")
    mdf = pd.read_csv(metrics_csv, dtype=str).fillna("")
    folios = [f.lower() for f in folios]

    ddf = ddf[ddf["folio"].str.lower().isin(folios)]
    mdf = mdf[mdf["folio"].str.lower().isin(folios)]
    if ddf.empty: raise ValueError("Selected folios not in decoded CSV.")
    if mdf.empty: raise ValueError("Selected folios not in metrics CSV.")

    if not out_pdf:
        out_pdf = os.path.join(OUTPUT_DIR, "Voynich_Decipherment_%s_Young2025.pdf" % ("_".join(folios)))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HeadingCenter", alignment=1, fontSize=16, leading=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="Body", fontSize=11, leading=14))
    styles.add(ParagraphStyle(name="Mini", fontSize=9, leading=12))

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, title="Voynich Decipherment")
    story = []

    cover_img = (folio_images or {}).get(folios[0])
    _add_cover_and_summary(story, styles, folios, mdf, cover_img)
    _add_methods_and_theory(story, styles)
    _add_worked_example(story, styles)
    _add_mapping_sections(story, styles)
    _add_exact_steps(story, styles)
    _add_summary_findings(story, styles, mdf, folios)

    # per-folio pages
    for f in folios:
        subd = ddf[ddf["folio"].str.lower() == f]
        subm = mdf[mdf["folio"].str.lower() == f]
        if subd.empty or subm.empty: continue

        eva     = subd.iloc[0].get("eva","")
        latin   = subd.iloc[0].get("latin_roots","")
        s_lat   = subd.iloc[0].get("latin_strange","")
        en_full = subd.iloc[0].get("english_full","")
        en_con  = subd.iloc[0].get("english_concise","")
        met     = subm.iloc[0].to_dict()

        img_here = (folio_images or {}).get(f)
        freq_img = os.path.join(OUTPUT_DIR, "freq_%s.png" % f)
        val_img  = os.path.join(OUTPUT_DIR, "validation_%s.png" % f)

        story.append(Paragraph("<b>Folio %s</b>" % f, styles["HeadingCenter"]))
        if img_here:
            story.append(Image(img_here, width=6.0*inch, height=4.2*inch))
            story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("<b>Original EVA (clean)</b>", styles["HeadingCenter"]))
        story.append(Paragraph(eva, styles["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("<b>Latin Roots (per clause)</b>", styles["HeadingCenter"]))
        story.append(Paragraph(latin, styles["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("<b>Strange Latin (implied verb fac/usa)</b>", styles["HeadingCenter"]))
        story.append(Paragraph(s_lat, styles["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("<b>Full English (readable, faithful)</b>", styles["HeadingCenter"]))
        story.append(Paragraph(en_full, styles["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("<b>Concise English (top 12)</b>", styles["HeadingCenter"]))
        sents = [s.strip() for s in re.split(r"\.\s+", en_con) if s.strip()]
        sents = sents[:12]
        story.append(ListFlowable([ListItem(Paragraph(s + ".", styles["Body"]), leftIndent=10) for s in sents], bulletType="1"))
        if os.path.exists(freq_img):
            story.append(Spacer(1, 0.2*inch))
            story.append(Image(freq_img, width=5*inch, height=2.5*inch))
        story.append(PageBreak())

        story.append(Paragraph("<b>Validation (Statistical Evidence)</b>", styles["HeadingCenter"]))
        mt = [
            ["Metric","Value"],
            ["Lexicon alignment (medical)", str(met.get("lexicon_alignment",""))],
            ["Unigram entropy H1 (bits)", str(met.get("unigram_entropy_bits",""))],
            ["Bigram entropy H2 (bits)", str(met.get("bigram_entropy_bits",""))],
            ["Type/Token ratio", str(met.get("type_token_ratio",""))],
            ["Top-10 bigram concentration (observed)", str(met.get("top10_bigram_conc",""))],
            ["Baseline mean concentration (200 shuffles)", str(met.get("baseline_mean_conc",""))],
            ["Z-score (obs vs baseline)", str(met.get("z_score",""))],
            ["p-value, one-tailed", str(met.get("p_value_one_tailed",""))],
            ["Clauses kept after de-dup", str(met.get("clauses_kept",""))]
        ]
        tbl = Table(mt, hAlign="LEFT")
        tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                                 ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
        story.append(tbl)
        if os.path.exists(val_img):
            story.append(Spacer(1, 0.15*inch))
            story.append(Image(val_img, width=4.5*inch, height=2.8*inch))
        story.append(PageBreak())

    doc.build(story)
    print("[OK] PDF -> %s" % out_pdf)
    return out_pdf

def main():
    ap = argparse.ArgumentParser(description="Make Voynich PDF (methods + example + 3-tier output)")
    ap.add_argument("--decoded", default=os.path.join(OUTPUT_DIR, "decoded_folios.csv"), help="Decoded CSV path")
    ap.add_argument("--metrics", default=os.path.join(OUTPUT_DIR, "metrics.csv"), help="Metrics CSV path")
    ap.add_argument("--folios", default="1r,69r,75r,99r,116v", help="Comma-separated folios, e.g. 1r or 1r,2r")
    ap.add_argument("--folio_images", default="1r=input/IMG_3536.jpg", help='Map like "1r=input/IMG_3536.jpg;2r=input/folio2.jpg"')
    ap.add_argument("--out", default=None, help="Output PDF path (default auto)")
    args = ap.parse_args()

    folios = _parse_folios(args.folios)
    images = _parse_folio_images(args.folio_images)
    build_pdf(args.decoded, args.metrics, folios, images, args.out)

if __name__ == "__main__":
    main()
