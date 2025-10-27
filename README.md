# Voynich Manuscript Decipherment using the Comprehensive Method and Mapping (CMM)

Author: Kenneth Young, PhD  
Contact: ken.g.young@gmail.com

---

## Overview

This repository presents a reproducible, data-driven workflow for the computational decipherment of the Voynich Manuscript - a 15th-century text that has eluded linguists, historians, and cryptographers for centuries.

Our framework, called the Comprehensive Method and Mapping (CMM), systematically maps Voynich glyphs and multi-glyph tokens to compact Latin roots, applies contextual grammatical reconstruction, and validates results through statistical and linguistic measures.

The complete manuscript is available here: [Voynich_Decipherment.pdf](output/voynich_decipherment.pdf)   

---
 
 ![Voynich Image](input/voynich_pages.jpg)

---

## Key Features

- CMM Workflow: Converts Voynich glyphs to Latin roots to concise English procedural translations.  
- Quantitative Validation: Entropy, type-token ratio, Z-scores, lexicon alignment, and permutation baselines.  
- AI Extension Ready: Framework designed for masked-language model (MLM) perplexity and cross-text validation.  
- Publication-Ready PDF: Automatically generates a polished manuscript with embedded figures and tables.  

---

## Repository Structure

Voynich-Decipherment/
|
|-- archive/
|   |-- Arhive files              # Archive files
|
|-- input/
|   |-- IMG_3532.jpg              # Sample manuscript image
|   |-- other manuscript images   # (optional)
|
|-- output/
|   |-- Voynich_Decipherment.pdf  # Pre-print manuscript
|   |-- metrics.csv               # Statistical metrics per folio
|   |-- decoded_folios.csv        # Full per-folio translations
|   |-- figs/                     # Auto-generated plots (Z-score, alignments)
|
|-- make_pdf.py                   # PDF builder script
|-- voynich_pipeline.py           # Core pipeline for decoding and metrics
|-- README.md                     # This file

---

## Usage

1. Generate Metrics and Translations  
   ```bash
   python voynich_pipeline.py
   ```
   This produces:
   - output/metrics.csv
   - output/decoded_folios.csv

2. Build the PDF Manuscript  
   ```bash
   python make_pdf.py --decoded output/decoded_folios.csv --metrics output/metrics.csv --image input/IMG_3532.jpg --out output/Voynich_Decipherment.pdf
   ```

The script generates figures, inserts manuscript imagery, formats methods and discussion sections, and outputs a publication-ready PDF.

---

## Figures

- Figure 1: Z-score Distribution - entropy-based statistical validation
  ![Z-score Distribution](input/entropy_histogram.jpg)
- Figure 2: Lexicon Alignment by Folio - top and bottom performing folios
  ![Lexicon Alignment](input/fig_alignment.jpg)

Both figures are generated automatically into the output/figs/ directory and embedded into the PDF.

---

## Sorting Logic for Folios

The helper function _folio_sort_cols() ensures natural sorting (e.g., 9r, 9v, 10r, 10v, 11r, 11v, ...).  
It parses numeric and side indicators so that folio 111r does not appear directly after 10r.

---

## Requirements

- Python 3.8 or higher  
- Dependencies:  
  ```bash
  pip install pandas matplotlib reportlab
  ```

---

## References

A complete reference list is provided in the manuscript.  
Key supporting works include:

- Green, M. (2001). The Trotula.  
- Sweeting, O. (2025). Deciphering a Mysterious Manuscript. Yale News.  
- Dintino, T. C. (2024). Trotula is Not an Example of the Matilda Effect. Science Education (Wiley).  
- Anonymous. (2025). A Focus on Trotula de' Ruggiero: A Pioneer in Women and Children Health. ResearchGate.  
- Devender, R. (2025). Decoding Voynich: The Progress So Far. Medium.

---

## Citation

If you use or build upon this work, please cite:

Young, K. (2025). Deciphering the Voynich Manuscript using the Comprehensive Method and Mapping (CMM).  

---

## License

This project is released under the MIT License.  
You are free to use, modify, and distribute it with proper attribution.

---

## Acknowledgments

Special thanks to the interdisciplinary community of linguists, historians, and AI researchers who have contributed insights into the Voynich mystery.  
This analysis pays homage to Trotula de' Ruggiero - a 12th-century physician whose pioneering work in women's health continues to inspire scientific rediscovery.

---

"What was once thought impenetrable may only have been waiting for the right lens - a balance of history, data, and imagination."
