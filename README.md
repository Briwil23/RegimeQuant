# RegimeQuant

RegimeQuant is a quantitative-finance research repository focused on disciplined, testable investigation of market-structure and statistical-regime hypotheses.

## Core Research Question

Can observable market structure and statistical regimes be identified using information available at the time, and can those regimes improve the robustness of systematic trading decisions out of sample?

This is a research question, not an assumption.
RegimeQuant does not assume regimes are predictive, does not assume statistical-arbitrage profitability, and does not claim alpha generation.

## Why This Project Exists

Many research pipelines overstate robustness by mixing in-sample optimization with insufficient out-of-sample controls.
RegimeQuant exists to enforce research discipline, reproducibility, and explicit boundary conditions before quantitative claims are made.

## Current Status

- Implemented milestone: M0 FOUNDATION
- M0 scope: repository architecture, research constitution, testing foundation, packaging policy, and data/governance discipline.
- Quantitative research methods are intentionally not implemented in M0.
- M1 market-data foundation is implemented locally but not yet certified or published.

## Architecture Overview

src-layout package:

- src/regimequant/data
- src/regimequant/features
- src/regimequant/regimes
- src/regimequant/statarb
- src/regimequant/signals
- src/regimequant/backtesting
- src/regimequant/validation
- src/regimequant/research

These are architectural placeholders for future milestones and currently contain no quantitative implementation logic.

The `src/regimequant/data` package now contains the M1 market-data foundation:

- semantic enums and provenance models
- canonical market-data validation and normalization
- deterministic fixture provider boundary
- observation-to-observation simple and log returns
- explicit three-clock semantics and availability metadata

## Research Principles

The research constitution is in docs/RESEARCH_CONSTITUTION.md and covers:

- causality
- look-ahead bias prevention
- survivorship-bias disclosure
- leakage controls
- out-of-sample discipline
- multiple-testing awareness
- transaction-cost realism
- reproducibility
- raw-data immutability
- failure-tolerant hypothesis handling
- claim discipline
- three-clock causality and information availability
- static-universe limitation
- adjusted-history vintage limitation

## Dashboard Boundary

A future research terminal may be introduced in a later milestone.
Quantitative/statistical logic must remain in src/regimequant and be consumed through public APIs.

## PaperTrail Boundary

RegimeQuant is a research system.
A separate future system (PaperTrail) may eventually consume certified signal records.
RegimeQuant does not implement broker integration, order execution, or paper trading.

Conceptual future signal contract (not implemented in M0):

- timestamp
- instrument
- signal
- target_position
- signal_strength
- regime
- model_version
- research_version

## Roadmap (Planning Labels)

- M0 — Repository & Research Constitution
- M1 — Market Data Foundation
- M2 — Statistical Feature Engine
- M3 — Regime Discovery
- M4 — Regime Stability
- M5 — Probabilistic Regimes
- M6 — Statistical-Arbitrage Universe
- M7 — Cointegration Research
- M8 — Signal Engine
- M9 — Backtesting Engine
- M10 — Walk-Forward Research
- M11 — Robustness & Statistical Validation
- M12 — Research Terminal
- M13 — Signal Contract
- M14 — Publication

These labels are planning references and may evolve through later audited decisions.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

For development testing tooling:

```bash
python -m pip install -e ".[dev]"
```

## Testing

Run foundational tests:

```bash
python -m pytest -q
```

Inspect collection:

```bash
python -m pytest --collect-only -q
```

## Data Hygiene Policy

Large downloaded market datasets should not be committed by default.
The repository ignores bulk files under:

- data/raw/
- data/interim/
- data/processed/

while preserving placeholder directories for structure.

## M1 Market-Data Foundation

M1 establishes a canonical long-form market-data contract with three distinct clocks:

- `timestamp`: the market observation time
- `available_at`: the earliest time the observation is considered available to research, or an explicit unknown/unverified state
- `retrieval_timestamp_utc`: when RegimeQuant retrieved the source data

Canonical prices must be finite and strictly greater than zero for the M1-supported equity-style close series.
Missing observations are represented by absent canonical rows, not NaN prices.
Returns are separate derived tables and do not mutate canonical observations.

Adjusted historical prices are supported as externally adjusted series, but M1 does not certify point-in-time vintage truth.

M1 uses a static universe only.
Point-in-time constituent ingestion remains future work and must not be inferred from the static-universe contract.

Deterministic fixture-based certification is preferred for tests.
No live market-data integration is part of M1.

### Canonical Semantics

Canonical observation columns are ordered and long-form:

1. `instrument`
2. `timestamp`
3. `available_at`
4. `availability_status`
5. `price`
6. `price_type`
7. `adjustment_state`
8. `currency`
9. `source_observation_id`

The supported enums are:

- `PriceType.CLOSE`
- `AdjustmentState.RAW`
- `AdjustmentState.ADJUSTED`
- `AvailabilityStatus.VERIFIED_TIMESTAMP`
- `AvailabilityStatus.UNKNOWN_UNVERIFIED`
- `AvailabilityStatus.UNDEFINED` (derived returns only)
- `UniverseType.STATIC`
- `ReturnType.SIMPLE`
- `ReturnType.LOG`

Availability policy is controlled by provenance `availability_policy_id` and must be one of:

- `fixture_declared_availability`
- `external_verified_availability`
- `unknown_unverified_availability`

`VERIFIED_TIMESTAMP` canonical observations require a compatible policy (`fixture_declared_availability` or `external_verified_availability`).
`AvailabilityStatus.UNDEFINED` is reserved for derived observations and is invalid for canonical price rows.

Availability distribution provenance is stage-specific:

- `source_availability_status_distribution` describes source/provider row counts before canonical duplicate collapse.
- `availability_status_distribution` describes canonical row counts after canonical validation and deduplication.

Canonicalization computes canonical counts deterministically as `(VERIFIED_TIMESTAMP, n_verified)` and `(UNKNOWN_UNVERIFIED, n_unknown)`.
If a non-empty declared canonical distribution conflicts with actual canonical rows, canonicalization raises a validation error.

`available_at` is distinct from retrieval time.
If historical availability cannot be established, the observation must carry `AvailabilityStatus.UNKNOWN_UNVERIFIED` and leave `available_at` null.

Provider adapters may localize naive source timestamps only when they declare an explicit source-timezone policy.
Canonicalization itself never guesses a timezone.

### Return Semantics

Returns are derived into a separate table.

- Simple return: `r_t = P_t / P_(t-1) - 1`
- Log return: `l_t = ln(P_t / P_(t-1))`

Returns are observation-to-observation rather than implicitly one-session or one-day returns.
The `previous_timestamp` and `interval_elapsed_seconds` fields expose irregular gaps without fabricating missing observations.

Return availability is derived from the inputs used to compute the return:

- first observation per instrument has no predecessor and is `UNDEFINED` with null `available_at`
- if both inputs are verified, `available_at` is the later of the two availability timestamps
- if either input is unknown, the derived return is `UNKNOWN_UNVERIFIED` and `available_at` is null

### Duplicate Identity Retention

Canonical uniqueness is keyed by:

- `instrument`
- `timestamp`
- `price_type`
- `adjustment_state`

Market-observation semantics are:

- `price`
- `available_at`
- `availability_status`
- `currency`

`source_observation_id` is treated as source-record identity metadata, not an economic observation value.
When duplicate rows share a canonical key and identical market semantics, canonicalization collapses rows and retains the lexicographically smallest normalized `source_observation_id` deterministically.
This deterministic collapse can lose per-record source identity.
It can also change canonical availability counts versus source-row counts, which is why source and canonical distributions are tracked separately.

### Provenance Durability Limitation

M1 stores provenance in `DataFrame.attrs` for in-memory workflows.
`DataFrame.attrs` is not a durable serialization contract and is not guaranteed to survive CSV round-trips, arbitrary DataFrame reconstruction, or all pandas operations.
Durable data-plus-metadata persistence is future work and requires an explicit artifact contract.

## Current Limitations (Intentional in M0)

RegimeQuant currently does not implement:

- market-data ingestion
- feature calculations
- PCA, clustering, GMM, HMM
- cointegration or ADF testing
- signal generation
- strategy logic
- portfolio optimization
- backtesting
- walk-forward analysis
- dashboard functionality
- paper trading
- broker integrations

## Claim Discipline

RegimeQuant does not claim:

- live trading capability
- alpha generation
- profitability
- real-time market-data operation
- globally arbitrage-free calibration
- production trading readiness

M1 does not claim:

- regime detection
- machine-learning market prediction
- statistical arbitrage
- backtesting
- paper trading
- live trading

## Testing

Run the M1-focused test suite:

```bash
python -m pytest -q
```

Inspect collection:

```bash
python -m pytest --collect-only -q
```
