# Track 4 local numerical artifact provenance

Updated: 2026-09-23

This record covers the local numerical artifacts used by the v3-safe development candidate. Evaluation-time inputs and citations remain limited to the supplied task and frozen corpus. No public-unit realized outcome is stored in the runtime image.

## 1. Treasury intermeeting interval calibration

Runtime file: `safe_calibration.py`

Purpose: widen the 90% interval for `rate_curve_cross_section` while leaving the House point forecast and citations unchanged.

Source data: Federal Reserve Bank of St. Louis FRED, U.S. Treasury constant-maturity daily yield series `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`, and `DGS30`.

Local source cache:
`outputs/fomc-local-benchmark-v1/fred_rates.csv`

SHA-256:
`82ebe5418e1a9656abc3448ccde1f077602733a11974c7ab1f1fb770b36b0bff`

Fitting period: 2000-01-01 through 2016-12-31.

Validation/model-selection period: 2017-01-01 through 2021-12-31.

Forecast horizon used for calibration: 35 trading days.

Selection rule: among pre-specified empirical absolute-move quantiles, choose the quantile minimizing the mean absolute distance between six-tenor unit coverage and the Track 4 0.90 coverage target on the 2017-2021 validation origins.

Selected quantile: 0.995.

Selected half-widths in basis points:
- UST2Y: 112.915
- UST3Y: 110.000
- UST5Y: 116.915
- UST7Y: 120.915
- UST10Y: 122.000
- UST30Y: 110.915

Reproduction script:
`scripts/fomc_static_interval_safe_v3.py`

Reproduction report:
`outputs/fomc-static-interval-safe-v3/report.json`

Cutoff gate: the fitted artifact is used only for tasks with `cutoff_date >= 2022-01-01`. The 2022 and 2024 public-unit realized outcomes are not used for fitting or model selection.

## 2. EPS-growth persistence/shrinkage artifact

Runtime file: `eps_growth_family.py`

Purpose: forecast year-over-year diluted-EPS growth in percent. This fixes the v2 semantic error where a prior-year EPS dollar level could be submitted as a growth percentage.

Source data: SEC EDGAR Company Facts, `us-gaap:EarningsPerShareDiluted` / compatible diluted-EPS facts for the eight public-bank training histories. Only facts filed on or before 2024-10-10 are admitted by the reproduction script.

Local source caches and SHA-256:
- `companyfacts_0000019617.json`: `27a08b844ba5765ab1c7831e78516db0809c0cea515dca191862e127dda2ad43`
- `companyfacts_0000036104.json`: `6eaf2f7c8e0b8cf1abbbf3ef8631197c7783856fb46135c1aa4bb4e87b371702`
- `companyfacts_0000070858.json`: `60b2ee1923ffa60f610793da306fddf75affe54f1a6d0319bc0ea96fd1622379`
- `companyfacts_0000072971.json`: `7fa5cc41ab1a7446173325f3f5f762484ece18ab8294948b0ff179232aaa782a`
- `companyfacts_0000713676.json`: `50ff6b07336dac69157dbdd04342ba6e1d2907a1a23fc34e5605c717306a024e`
- `companyfacts_0000831001.json`: `907c94f8ce3750021bc977eede4959e983ad1aef67727d01699c1731c8ae4f8d`
- `companyfacts_0000886982.json`: `fafc87f7ce3da9ba43a0da02b5d48f2df54dfa5148d9d7fdffb85f17ee6f8629`
- `companyfacts_0000895421.json`: `7128320406778b1439f822bda3e27a84e682f59e8a2d8a9ef2ebe2121361d5c5`

Fitting/model-selection sample: 228 historical quarter-to-quarter transitions whose target-quarter diluted EPS was filed by the 2024-10-10 cutoff. The public 2024Q3 realized EPS values are held out and are used only in local diagnostic reporting after parameter selection.

Selected rule:
- previous-quarter YoY diluted-EPS growth is clipped to +/-50 percentage points;
- persistence coefficient = 0.75;
- 90% residual half-width = 77.52301640441917 percentage points.

Reproduction script:
`scripts/eps_growth_local_benchmark.py`

Reproduction report:
`outputs/eps-growth-local-benchmark-v1/report.json`

Cutoff gate: fitted parameters are used only for tasks with `cutoff_date >= 2024-10-10`. Earlier tasks fall back to a code-only semantic transformation and do not use the learned parameters.

## 3. Post-earnings interval safety rule

Runtime file: `safe_calibration.py`

No fitted artifact is used. The House label, House point forecast, and citations are left unchanged. The submitted 90% interval is widened to ten times the family class threshold on either side of the existing point forecast. The published family threshold is +/-1 percentage point, so the default half-width is 10 percentage points.

This is a deterministic task-semantics rule. It does not use historical earnings outcomes, the 2024-02-02 public realization, or any stored answer lookup.

## 4. Non-artifact code

BM25 retrieval, schema validation, cutoff filtering, output sanitation, and deterministic numerical transformations are code paths rather than learned models. They do not introduce external inference-time documents.

The House model remains the only language model used at evaluation time.
