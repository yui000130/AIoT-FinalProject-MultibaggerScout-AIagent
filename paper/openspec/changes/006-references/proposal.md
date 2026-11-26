# Change 006: References (IEEE Format)

## Goal
Maintain the IEEE bibliography (`\bibitem{b1}`–`\bibitem{b16}`) of `paper/main.tex`. Updated 2026-11-26.

## Affected Sections
- `paper/main.tex`: `\begin{thebibliography}{99}...\end{thebibliography}`

## Changes on 2026-11-26
- Added the core references of DESIGN.md v0.8: Bessembinder et al. 2023 (FAJ), Program of Thoughts (Chen et al., TMLR 2023), He et al. 2025, Gu, Kelly and Xiu 2020, Yartseva 2025 (original working paper, replacing the HKEJ news summary), Jensen, Kelly and Pedersen 2023, Bui et al. 2023, Sloan 1996, Lakonishok, Shleifer and Vishny 1994.
- Removed sources the paper no longer relies on: Lopez-Lira and Tang, the Alpha Arena blog, the Agentic RAG PoT preprint, CommonWealth, Focus Taiwan, the CNA 2025 market-close report, pitbacktest.
- Kept from the A–F evidence pack: Bessembinder 2026, SPIVA 2025, Gao, Jiang and Yan 2026, Didisheim et al. 2025, FAITH, TradingAgents, CNA account statistics.

## Verification Criteria
- Every `\cite{}` key resolves and every `\bibitem` is cited (checked by script: 0 undefined, 0 unused).
- Before submission, open each online source of [1], [3], [6], [8], [11] to confirm the cited statement.
