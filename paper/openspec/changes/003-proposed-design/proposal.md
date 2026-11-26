# Change 003: Proposed Design (PA–PM)

## Goal
Maintain Section III (Proposed Design) of `paper/main.tex` following items PA–PM of the A-to-Z SOP. Rewritten 2026-11-26 to match DESIGN.md v0.8 (sections 3, 5, 6, 7, 8).

## Affected Sections
- `paper/main.tex`: `\section{Proposed Design: Multibagger Scout}`, Equations (1)–(6), Algorithms 1–2

## A–Z Checklist Mapping
- **PA–PC (Aim, Basis, Principle):** Rank TWSE common stocks by their potential to be fundamentally confirmed five-year winners; PoT principle with fixed tools instead of model-generated code.
- **PF, PM (Formulation, Math):** Universe (TWSE only, listed at least one year, emerging-board and bottom-10% liquidity excluded); wealth series with dividend, capital-reduction and split factors; 20-trading-day average end point; top ⌈0.05N⌉ label; revenue CAGR from year y−1 to y+4 ≥ 10%; UNKNOWN; point-in-time information constraint; resolved-label constraint (training year + 5 ≤ decision year).
- **PD, PE, PG (Design, Elements):** 8 modules; tools get_features, score_model, find_cases, check_rules, follow_up, verify_report (at most 12 calls); evidence records E001, E002, …; potential level by within-year rank (top 5% / 5–20% / rest); kNN with 3 successes + 3 failures on percentile ranks; 2 rules (accruals over average total assets; B/M or P/E deciles).
- **PH, PI (How, Implementation):** Algorithm 1 point-in-time panel construction; Algorithm 2 agent query with 3-round verification and template fallback.
- **PJ (Jump to an example):** Real cached run: 1310 at 2018-04-30, 45 evidence records; the first draft was rejected for 3 uncited numbers and the second draft passed.
- **PK, PL (Key points, Link):** All settings fixed before each experiment; E1–E3 answer RQ1–RQ2, E4 answers RQ3.
