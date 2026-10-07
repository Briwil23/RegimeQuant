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
