# Change 000: Abstract (A–F)

## Goal
Maintain the Abstract and IEEE Keywords of `paper/main.tex` following the A–F structure of the A-to-Z SOP. Updated 2026-11-26 to DESIGN.md v0.8 and the completed E1–E4 results; E5 (label sensitivity) added to the experiment sentence the same day.

## Affected Sections
- `paper/main.tex`: `\begin{abstract}...\end{abstract}` and `\begin{IEEEkeywords}...\end{IEEEkeywords}`

## A–Z Checklist Mapping
- **A (Attention Getter):** Concentration of long-term wealth (Bessembinder et al. 2023; Bessembinder 2026); 14.8M Taiwan brokerage accounts (CNA, Oct 2026).
- **B (But):** Low persistence of active funds (SPIVA), look-ahead and survivorship bias, LLM future-information leakage and arithmetic errors (FAITH).
- **C (Cure):** Multibagger Scout, a point-in-time research agent for TWSE common stocks.
- **D (Development):** PoT-style separation: fixed tools compute 10 prior-year features, LightGBM score, TreeSHAP, resolved kNN cases and 2 red-flag rules; the LLM writes the report; 3-round verifier with template fallback.
- **E (Experiments):** 6,193 firm-years (2013–2020, 1,171 firms incl. delisted), three walk-forward test years 2018–2020; E1 ranking vs 1,000 random rankings and factor rules, E2 top-50 vs 0050, E3 feature associations, E4 540 reports vs a direct LLM.
- **F (Findings):** LightGBM PR-AUC above the random 95th percentile in 2 of 3 rounds (0.082, 0.081), baselines never; report numerical accuracy 1.000 vs 0.623 (+0.377, 95% CI 0.359–0.397); 10.4% template fallback.
