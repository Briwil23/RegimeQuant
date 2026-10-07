# RegimeQuant Research Constitution

RegimeQuant is a research system. It is not a live trading system and does not execute orders.

## Core Research Question

Can observable market structure and statistical regimes be identified using information available at the time, and can those regimes improve the robustness of systematic trading decisions out of sample?

This question is investigatory, not assumed true.

## A. Causality

Information dated after time t must not influence quantities represented as available at time t.
All future milestones must preserve strict causal information flow.

## B. Look-Ahead Bias

Look-ahead bias is prohibited.
Future transformations, fitting, parameter estimation, normalization, selection, and validation must respect historical information boundaries.

## C. Survivorship Bias

Present-day universe membership must not be silently treated as historical point-in-time membership.
If point-in-time constituent data is unavailable, that limitation must be explicit.

## D. Data Leakage

Training, validation, and test information must remain appropriately separated.
Any preprocessing that learns from data must eventually be fit only on information available within the relevant training window.

## E. Out-of-Sample Evaluation

In-sample evidence is not sufficient.
Future claims must clearly distinguish training, validation, testing, and walk-forward/out-of-sample evaluation.

## F. Multiple Testing

Testing many hypotheses increases false discoveries.
Future research must record the number and nature of tested hypotheses and must not present best historical outcomes without acknowledging the search process.

## G. Transaction Costs

Future strategy evaluation must not equate frictionless returns with realizable performance.
Relevant work should account for transaction costs, turnover, and spread/slippage assumptions where applicable.

## H. Reproducibility

Research outputs must be reproducible from explicit data provenance, universe definition, configuration, random seeds where relevant, model parameters, and code version.

## I. Immutability of Raw Data

Raw source data should not be silently overwritten by transformations.
Derived datasets must remain conceptually distinct from source inputs.

## J. Failure Is Allowed

A hypothesis that fails rigorous validation must not be manipulated to appear successful.
Negative results are valid research outcomes.

## K. Claim Discipline

RegimeQuant must distinguish observed historical relationships, statistical evidence, model estimates, out-of-sample results, simulated strategy results, and actual live trading results.
RegimeQuant does not conduct live trading.

## Boundaries

- Quantitative/statistical algorithms belong in src/regimequant.
- Any future dashboard must consume public research APIs rather than define canonical research logic.
- RegimeQuant may eventually emit certified signal records for a separate future system, PaperTrail.
- RegimeQuant will not include broker integration or order execution.

## Randomness Policy

Any future stochastic research procedure must support explicit deterministic random seeds where technically appropriate.
No research result should depend on undocumented random state.
