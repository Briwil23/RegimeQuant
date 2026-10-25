from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable

from regimequant.backtesting.m9_accounting import (
    create_ledger,
    equity,
    mark_unrealized_pnl,
    money,
    post_cash_dividend,
    post_atomic_fill,
    post_borrow_and_financing,
    post_split,
    reconciliation_residual,
    validate_ledger_residual,
)
from regimequant.backtesting.m9_contracts import AuthorizationInput, OrderIntent, OrderStatus, RiskCaps, SignalType, SourceMode
from regimequant.backtesting.m9_costs import borrow_accrual, commission_cost, financing_accrual, slippage_cost
from regimequant.backtesting.m9_execution import Stage1Context, Stage2RiskContext, authorize_order, execute_atomic_at_selected_close
from regimequant.backtesting.m9_identity import accounting_identity, event_stream_identity, fill_ledger_identity, policy_identity
from regimequant.backtesting.m9_reporting import build_research_report
from regimequant.backtesting.m9_temporal import deterministic_order_priority, deterministic_tiebreak_key, holding_exit_session
from regimequant.backtesting.m9_valuation import certification_status, delisting_valuation_status, valuation_from_stale_age
from regimequant.signals.m8_contracts import M8SignalEnum, M8SignalObservation


@dataclass(frozen=True, slots=True)
class SessionPrice:
    session_index: int
    close_y: Decimal | None
    close_x: Decimal | None
    available_at: datetime
    close_event_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class CorporateActionEvent:
    session_index: int
    security_id: str
    action_type: str
    value: Decimal
    evidence_complete: bool


@dataclass(frozen=True, slots=True)
class PairSignalInput:
    signal_observation: M8SignalObservation
    beta: Decimal
    y_security_id: str
    x_security_id: str
    pair_id: str
    authorization_time: datetime
    submission_time: datetime
    signal_identity: str
    decision_session_index: int | None = None
    decision_timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class SecurityObservation:
    session_index: int
    security_id: str
    close_price: Decimal | None
    close_event_time: datetime
    available_at: datetime
    evidence_id: str


@dataclass(frozen=True, slots=True)
class ResearchInput:
    signal_observation: M8SignalObservation
    beta: Decimal
    y_security_id: str
    x_security_id: str
    pair_id: str
    sessions: list[SessionPrice]
    authorization_time: datetime
    submission_time: datetime
    initial_equity: Decimal = Decimal("100000000")
    max_signal_delay_sessions: int = 2
    pair_signals: list[PairSignalInput] | None = None
    security_observations: list[SecurityObservation] | None = None
    corporate_actions: list[CorporateActionEvent] | None = None
    forced_exit_retry_limit: int = 3
    session_calendar_identity: str = "default-session-calendar"


@dataclass(frozen=True, slots=True)
class EngineResult:
    events: list[dict[str, object]]
    fills: list[dict[str, object]]
    report: dict[str, object]


def _to_signal_type(signal: M8SignalEnum) -> SignalType | None:
    if signal == M8SignalEnum.LONG_SPREAD_CANDIDATE:
        return SignalType.LONG_SPREAD
    if signal == M8SignalEnum.SHORT_SPREAD_CANDIDATE:
        return SignalType.SHORT_SPREAD
    return None


def _security_exposure_from_positions(positions: dict[str, Decimal], prices: dict[str, Decimal]) -> dict[str, Decimal]:
    exposure: dict[str, Decimal] = {}
    for security_id, qty in positions.items():
        if qty != Decimal("0"):
            exposure[security_id] = qty * prices[security_id]
    return exposure


def _sorted_sessions(sessions: list[SessionPrice]) -> list[SessionPrice]:
    return sorted(sessions, key=lambda item: item.session_index)


def _normalize_pair_inputs(input_data: ResearchInput) -> list[PairSignalInput]:
    def _sort_key(item: PairSignalInput) -> tuple[str, str, str]:
        return (item.pair_id, item.authorization_time.isoformat(), item.signal_identity)

    if input_data.pair_signals:
        return sorted(list(input_data.pair_signals), key=_sort_key)
    return sorted([
        PairSignalInput(
            signal_observation=input_data.signal_observation,
            beta=input_data.beta,
            y_security_id=input_data.y_security_id,
            x_security_id=input_data.x_security_id,
            pair_id=input_data.pair_id,
            authorization_time=input_data.authorization_time,
            submission_time=input_data.submission_time,
            signal_identity=input_data.signal_observation.scientific_identity,
        )
    ], key=_sort_key)


def _session_price_map(sessions: Iterable[SessionPrice]) -> dict[int, SessionPrice]:
    return {session.session_index: session for session in sessions}


def _session_security_observations(input_data: ResearchInput, pair_inputs: list[PairSignalInput]) -> dict[int, dict[str, SecurityObservation]]:
    by_session: dict[int, dict[str, SecurityObservation]] = {}
    if input_data.security_observations:
        for obs in input_data.security_observations:
            session_bucket = by_session.setdefault(obs.session_index, {})
            existing = session_bucket.get(obs.security_id)
            if existing is None:
                session_bucket[obs.security_id] = obs
                continue
            if (
                existing.close_price != obs.close_price
                or existing.close_event_time != obs.close_event_time
                or existing.available_at != obs.available_at
                or existing.evidence_id != obs.evidence_id
            ):
                raise ValueError("CONFLICTING_SECURITY_OBSERVATION")
        return by_session

    for session in _sorted_sessions(input_data.sessions):
        mapped: dict[str, SecurityObservation] = {}
        close_event_time = session.close_event_time or session.available_at
        for pair in pair_inputs:
            mapped[pair.y_security_id] = SecurityObservation(
                session_index=session.session_index,
                security_id=pair.y_security_id,
                close_price=session.close_y,
                close_event_time=close_event_time,
                available_at=session.available_at,
                evidence_id=f"{pair.pair_id}:{session.session_index}:Y",
            )
            mapped[pair.x_security_id] = SecurityObservation(
                session_index=session.session_index,
                security_id=pair.x_security_id,
                close_price=session.close_x,
                close_event_time=close_event_time,
                available_at=session.available_at,
                evidence_id=f"{pair.pair_id}:{session.session_index}:X",
            )
        by_session[session.session_index] = mapped
    return by_session


def _pair_gross_notional(open_positions: dict[str, dict[str, Decimal]], latest_prices: dict[str, Decimal]) -> Decimal:
    gross = Decimal("0")
    for pair_state in open_positions.values():
        y_id = pair_state["y_id"]
        x_id = pair_state["x_id"]
        qy = pair_state["qy"]
        qx = pair_state["qx"]
        if y_id in latest_prices:
            gross += qy.copy_abs() * latest_prices[y_id]
        if x_id in latest_prices:
            gross += qx.copy_abs() * latest_prices[x_id]
    return gross


def _security_exposure_from_ledger(ledger_positions: dict[str, Decimal], latest_prices: dict[str, Decimal]) -> dict[str, Decimal]:
    exposure: dict[str, Decimal] = {}
    for sec, qty in ledger_positions.items():
        if sec in latest_prices and qty != Decimal("0"):
            exposure[sec] = qty * latest_prices[sec]
    return exposure


def _event_priority(event_type: str) -> int:
    if event_type == "forced_exit":
        return deterministic_order_priority(OrderIntent.FORCED_EXIT)
    if event_type == "scheduled_exit":
        return deterministic_order_priority(OrderIntent.SCHEDULED_EXIT)
    if event_type == "opposing_close":
        return deterministic_order_priority(OrderIntent.OPPOSING_CLOSE)
    return deterministic_order_priority(OrderIntent.NEW_ENTRY)


def run_research_backtest(input_data: ResearchInput) -> EngineResult:
    events: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []

    pair_inputs = _normalize_pair_inputs(input_data)

    ordered_sessions = _sorted_sessions(input_data.sessions)
    default_decision_session = int(ordered_sessions[0].session_index) if ordered_sessions else 0
    decision_provenance_payload: list[dict[str, object]] = []
    for pair in pair_inputs:
        decision_provenance_payload.append(
            {
                "pair_id": pair.pair_id,
                "canonical_pair_id": pair.signal_observation.canonical_pair_id,
                "m8_signal_identity": pair.signal_observation.scientific_identity,
                "decision_session_index": pair.decision_session_index if pair.decision_session_index is not None else default_decision_session,
                "decision_timestamp": (pair.decision_timestamp or pair.authorization_time).isoformat(),
                "observation_timestamp": pair.signal_observation.eval_observation_time.isoformat(),
                "availability_timestamp": pair.signal_observation.eval_available_at.isoformat(),
            }
        )

    decision_provenance_identity = event_stream_identity(sorted(decision_provenance_payload, key=lambda item: (str(item["pair_id"]), str(item["m8_signal_identity"]))))

    if not ordered_sessions:
        report = build_research_report(
            equity_start=input_data.initial_equity,
            equity_end=input_data.initial_equity,
            uncertifiable=True,
            uncertifiable_reason="NO_SESSIONS",
            assumptions={"mode": "research-only"},
            identities={"decision_provenance_identity": decision_provenance_identity},
        )
        return EngineResult(events=events, fills=fills, report=report)

    missing_decision_evidence = next(
        (
            pair
            for pair in pair_inputs
            if not pair.signal_observation.canonical_pair_id
            or not pair.signal_observation.scientific_identity
            or not pair.signal_identity
        ),
        None,
    )
    if missing_decision_evidence is not None:
        events.append({"event": "authorize_reject", "pair_id": missing_decision_evidence.pair_id, "reason": "MISSING_DECISION_PROVENANCE"})
        report = build_research_report(
            equity_start=input_data.initial_equity,
            equity_end=input_data.initial_equity,
            uncertifiable=True,
            uncertifiable_reason="MISSING_DECISION_PROVENANCE",
            assumptions={"mode": "research-only"},
            identities={
                "decision_provenance_identity": decision_provenance_identity,
                "events_identity": event_stream_identity(events),
                "fills_identity": fill_ledger_identity(fills),
            },
        )
        return EngineResult(events=events, fills=fills, report=report)

    if len(pair_inputs) > 1 and not input_data.security_observations:
        events.append({"event": "authorize_reject", "pair_id": "*", "reason": "MISSING_CANONICAL_SECURITY_OBSERVATIONS"})
        report = build_research_report(
            equity_start=input_data.initial_equity,
            equity_end=input_data.initial_equity,
            uncertifiable=True,
            uncertifiable_reason="MISSING_CANONICAL_SECURITY_OBSERVATIONS",
            assumptions={"mode": "research-only"},
            identities={
                "decision_provenance_identity": decision_provenance_identity,
                "events_identity": event_stream_identity(events),
                "fills_identity": fill_ledger_identity(fills),
            },
        )
        return EngineResult(events=events, fills=fills, report=report)

    ledger = create_ledger(input_data.initial_equity)
    caps = RiskCaps()

    session_prices = _session_price_map(ordered_sessions)
    try:
        session_observations = _session_security_observations(input_data, pair_inputs)
    except ValueError:
        events.append({"event": "authorize_reject", "pair_id": "*", "reason": "CONFLICTING_SECURITY_OBSERVATION"})
        report = build_research_report(
            equity_start=input_data.initial_equity,
            equity_end=input_data.initial_equity,
            uncertifiable=True,
            uncertifiable_reason="CONFLICTING_SECURITY_OBSERVATION",
            assumptions={"mode": "research-only"},
            identities={
                "decision_provenance_identity": decision_provenance_identity,
                "events_identity": event_stream_identity(events),
                "fills_identity": fill_ledger_identity(fills),
            },
        )
        return EngineResult(events=events, fills=fills, report=report)
    open_positions: dict[str, dict[str, Decimal | int | str]] = {}
    pending_forced_exits: dict[str, dict[str, Decimal | int | str]] = {}
    reservations: dict[str, Decimal] = {}
    pending_entries: dict[int, list[dict[str, object]]] = {}
    any_authorized = False
    any_auth_rejected = False

    latest_prices: dict[str, Decimal] = {}
    if pair_inputs:
        for item in pair_inputs:
            latest_prices.setdefault(item.y_security_id, Decimal("0"))
            latest_prices.setdefault(item.x_security_id, Decimal("0"))

    for pair in pair_inputs:
        if pair.signal_observation.source_mode != "replay":
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "PRODUCTION_SCOPE_FORBIDDEN"})
            any_auth_rejected = True
            continue

        signal_type = _to_signal_type(pair.signal_observation.signal)
        if signal_type is None:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "NON_ACTIONABLE_SIGNAL"})
            any_auth_rejected = True
            continue

        if pair.signal_observation.eval_available_at > pair.authorization_time:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "LATE_EVIDENCE"})
            any_auth_rejected = True
            continue

        decision_session = int(pair.decision_session_index) if pair.decision_session_index is not None else int(ordered_sessions[0].session_index)
        decision_ts = pair.decision_timestamp or pair.authorization_time
        selected_entry = next((s for s in ordered_sessions if s.session_index > decision_session), None)
        if selected_entry is None:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "NO_NEXT_ELIGIBLE_SESSION"})
            any_auth_rejected = True
            continue

        if selected_entry.session_index > decision_session + input_data.max_signal_delay_sessions:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "SIGNAL_EXPIRED"})
            any_auth_rejected = True
            continue

        selected_obs = session_observations.get(selected_entry.session_index, {})
        y_obs = selected_obs.get(pair.y_security_id)
        x_obs = selected_obs.get(pair.x_security_id)
        if y_obs is None or x_obs is None:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "MISSING_SELECTED_CLOSE"})
            any_auth_rejected = True
            continue
        if y_obs.available_at > y_obs.close_event_time or x_obs.available_at > x_obs.close_event_time:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "LATE_CLOSE_AVAILABILITY"})
            any_auth_rejected = True
            continue
        selected_close_event_time = max(y_obs.close_event_time, x_obs.close_event_time)
        if decision_ts > selected_close_event_time:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "CHRONOLOGY_INVALID"})
            any_auth_rejected = True
            continue
        if pair.submission_time > selected_close_event_time:
            events.append({"event": "authorize_reject", "pair_id": pair.pair_id, "reason": "SUBMISSION_AFTER_SELECTED_CLOSE"})
            any_auth_rejected = True
            continue

        auth_input = AuthorizationInput(
            pair_id=pair.pair_id,
            y_security_id=pair.y_security_id,
            x_security_id=pair.x_security_id,
            signal=signal_type,
            source_mode=SourceMode.RESEARCH,
            beta=pair.beta,
            authorization_time=pair.authorization_time,
            decision_session_index=decision_session,
            evidence_available=True,
            provenance_complete=bool(pair.signal_observation.evidence_digest),
            initial_equity=input_data.initial_equity,
            current_equity=input_data.initial_equity,
            free_cash=ledger.free_cash,
        )

        current_reserved = sum(reservations.values(), Decimal("0"))
        stage1 = Stage1Context(
            evidence_ready_at=pair.signal_observation.eval_available_at.to_pydatetime(),
            preliminary_risk_ok=True,
            current_reserved_capital=current_reserved,
        )
        auth = authorize_order(auth_input, stage1, caps)
        events.append({"event": "authorize", "pair_id": pair.pair_id, "status": auth.status.value, "reason": auth.reason.value if auth.reason else None})
        if auth.status != OrderStatus.AUTHORIZED or auth.policy is None:
            any_auth_rejected = True
            continue
        any_authorized = True

        reservations[pair.pair_id] = auth.policy.gross_budget
        pending_entries.setdefault(selected_entry.session_index, []).append(
            {
                "pair_id": pair.pair_id,
                "policy": auth.policy,
                "auth_time": pair.authorization_time,
                "signal_identity": pair.signal_identity,
                "y_id": pair.y_security_id,
                "x_id": pair.x_security_id,
                "decision_time": decision_ts,
            }
        )

    uncertifiable = False
    uncertifiable_reason: str | None = None
    last_mark_session = ordered_sessions[0].session_index
    prior_prices = dict(latest_prices)
    total_delta_unrealized = Decimal("0")
    required_securities = {item.y_security_id for item in pair_inputs} | {item.x_security_id for item in pair_inputs}

    action_by_session: dict[int, list[CorporateActionEvent]] = {}
    for action in input_data.corporate_actions or []:
        action_by_session.setdefault(action.session_index, []).append(action)

    for session in ordered_sessions:
        session_idx = session.session_index

        current_observations = session_observations.get(session_idx, {})
        has_complete_prices = True
        for required_security in required_securities:
            obs = current_observations.get(required_security)
            if obs is None or obs.close_price is None:
                has_complete_prices = False
                break

        for sec_id, obs in current_observations.items():
            if obs.available_at > obs.close_event_time:
                uncertifiable = True
                uncertifiable_reason = uncertifiable_reason or "LATE_CLOSE_AVAILABILITY"
            if obs.close_price is not None:
                latest_prices[sec_id] = obs.close_price

        if has_complete_prices:
            last_mark_session = session_idx

        # Corporate actions are evidence-gated and never fabricated.
        for action in action_by_session.get(session_idx, []):
            if action.action_type == "split":
                ledger, ok = post_split(
                    ledger,
                    security_id=action.security_id,
                    split_ratio=action.value,
                    evidence_complete=action.evidence_complete,
                )
                events.append({"event": "corp_action_split", "security": action.security_id, "ok": ok})
                if not ok:
                    uncertifiable = True
                    uncertifiable_reason = "MISSING_CORPORATE_ACTION_EVIDENCE"
            elif action.action_type == "cash_dividend":
                ledger, ok = post_cash_dividend(
                    ledger,
                    security_id=action.security_id,
                    dividend_per_share=action.value,
                    evidence_complete=action.evidence_complete,
                )
                events.append({"event": "corp_action_dividend", "security": action.security_id, "ok": ok})
                if not ok:
                    uncertifiable = True
                    uncertifiable_reason = "MISSING_CORPORATE_ACTION_EVIDENCE"
            else:
                uncertifiable = True
                uncertifiable_reason = "UNSUPPORTED_CORPORATE_ACTION"
                events.append({"event": "corp_action_unsupported", "security": action.security_id, "type": action.action_type})

        # Build the session candidate queue by frozen priority and deterministic tie-break.
        candidates: list[dict[str, object]] = []
        for pair_id, state in pending_forced_exits.items():
            candidates.append(
                {
                    "type": "forced_exit",
                    "pair_id": pair_id,
                    "auth_time": state["auth_time"],
                    "signal_identity": state["signal_identity"],
                }
            )
        for pair_id, state in open_positions.items():
            if int(state["exit_session"]) == session_idx:
                candidates.append(
                    {
                        "type": "scheduled_exit",
                        "pair_id": pair_id,
                        "auth_time": state["auth_time"],
                        "signal_identity": state["signal_identity"],
                    }
                )
        for entry in pending_entries.get(session_idx, []):
            existing = open_positions.get(str(entry["pair_id"]))
            if existing and existing["direction"] != entry["policy"].direction:
                candidates.append(
                    {
                        "type": "opposing_close",
                        "pair_id": entry["pair_id"],
                        "auth_time": entry["auth_time"],
                        "signal_identity": entry["signal_identity"],
                        "entry": entry,
                    }
                )
            elif not existing:
                candidates.append(
                    {
                        "type": "new_entry",
                        "pair_id": entry["pair_id"],
                        "auth_time": entry["auth_time"],
                        "signal_identity": entry["signal_identity"],
                        "entry": entry,
                    }
                )

        candidates.sort(
            key=lambda item: (
                _event_priority(str(item["type"])),
                str(item["pair_id"]),
                deterministic_tiebreak_key(
                    pair_id=str(item["pair_id"]),
                    authorization_timestamp=item["auth_time"],
                    signal_identity=str(item["signal_identity"]),
                ),
            )
        )

        for candidate in candidates:
            candidate_type = str(candidate["type"])
            pair_id = str(candidate["pair_id"])
            session_obs = session_observations.get(session_idx, {})

            if candidate_type in {"scheduled_exit", "forced_exit", "opposing_close"}:
                state = open_positions.get(pair_id)
                if state is None:
                    continue
                y_id = str(state["y_id"])
                x_id = str(state["x_id"])
                y_obs = session_obs.get(y_id)
                x_obs = session_obs.get(x_id)
                py = y_obs.close_price if y_obs is not None else None
                px = x_obs.close_price if x_obs is not None else None
                if py is None or px is None:
                    if candidate_type == "forced_exit":
                        retries = int(state.get("forced_retry", 0)) + 1
                        state["forced_retry"] = retries
                        events.append({"event": "forced_exit_retry", "pair_id": pair_id, "retry": retries})
                        if retries >= input_data.forced_exit_retry_limit:
                            uncertifiable = True
                            uncertifiable_reason = "FORCED_EXIT_UNRESOLVED"
                            events.append({"event": "forced_exit_exhausted", "pair_id": pair_id})
                    else:
                        pending_forced_exits[pair_id] = dict(state)
                        pending_forced_exits[pair_id]["auth_time"] = candidate["auth_time"]
                        pending_forced_exits[pair_id]["signal_identity"] = candidate["signal_identity"]
                        events.append({"event": "scheduled_exit_missing_close", "pair_id": pair_id})
                    continue

                close_legs = (
                    type(state["fill_legs"][0])(
                        security_id=y_id,
                        quantity=-state["fill_legs"][0].quantity,
                        close_price=py,
                    ),
                    type(state["fill_legs"][1])(
                        security_id=x_id,
                        quantity=-state["fill_legs"][1].quantity,
                        close_price=px,
                    ),
                )
                notional_y_exit = close_legs[0].quantity.copy_abs() * close_legs[0].close_price
                notional_x_exit = close_legs[1].quantity.copy_abs() * close_legs[1].close_price
                commissions_exit = (commission_cost(notional_y_exit), commission_cost(notional_x_exit))
                slippages_exit = (slippage_cost(notional_y_exit), slippage_cost(notional_x_exit))
                ledger = post_atomic_fill(ledger, close_legs, commissions=commissions_exit, slippages=slippages_exit)
                fills.append(
                    {
                        "pair_id": pair_id,
                        "session": session_idx,
                        "legs": [
                            {"security": leg.security_id, "qty": str(leg.quantity), "price": str(leg.close_price)}
                            for leg in close_legs
                        ],
                    }
                )
                open_positions.pop(pair_id, None)
                pending_forced_exits.pop(pair_id, None)
                reservations.pop(pair_id, None)
                events.append({"event": "exit_fill", "pair_id": pair_id, "session": session_idx, "status": "FILLED"})
                if candidate_type == "opposing_close":
                    # Preserve no same-close reversal; the opposite entry can happen only on later session.
                    events.append({"event": "opposing_close_done", "pair_id": pair_id, "session": session_idx})
                continue

            entry = candidate.get("entry")
            if entry is None:
                continue
            policy = entry["policy"]
            y_id = str(entry["y_id"])
            x_id = str(entry["x_id"])
            y_obs = session_obs.get(y_id)
            x_obs = session_obs.get(x_id)
            py = y_obs.close_price if y_obs is not None else None
            px = x_obs.close_price if x_obs is not None else None

            stage2 = Stage2RiskContext(
                current_security_exposure=_security_exposure_from_ledger(ledger.positions, latest_prices),
                free_cash_before_fill=ledger.free_cash,
                reservation_released=reservations.get(pair_id, Decimal("0")),
                current_pair_gross=_pair_gross_notional(open_positions, latest_prices),
            )

            fill = execute_atomic_at_selected_close(
                policy=policy,
                selected_session_index=session_idx,
                close_price_y=py,
                close_price_x=px,
                stage2_risk=stage2,
                caps=caps,
                current_equity=equity(ledger, latest_prices),
            )
            events.append({"event": "entry_fill", "pair_id": pair_id, "status": fill.status.value, "reason": fill.reason.value if fill.reason else None})
            if fill.status != OrderStatus.FILLED:
                if fill.reason and fill.reason.value == "MISSING_SELECTED_CLOSE":
                    uncertifiable = True
                    uncertifiable_reason = uncertifiable_reason or "MISSING_REQUIRED_PRICES"
                reservations.pop(pair_id, None)
                continue

            notional_y = fill.legs[0].quantity.copy_abs() * fill.legs[0].close_price
            notional_x = fill.legs[1].quantity.copy_abs() * fill.legs[1].close_price
            commissions = (commission_cost(notional_y), commission_cost(notional_x))
            slippages = (slippage_cost(notional_y), slippage_cost(notional_x))
            ledger = post_atomic_fill(ledger, fill.legs, commissions=commissions, slippages=slippages)
            fills.append(
                {
                    "pair_id": pair_id,
                    "session": session_idx,
                    "legs": [
                        {"security": leg.security_id, "qty": str(leg.quantity), "price": str(leg.close_price)}
                        for leg in fill.legs
                    ],
                }
            )

            open_positions[pair_id] = {
                "fill_legs": fill.legs,
                "y_id": y_id,
                "x_id": x_id,
                "qy": fill.legs[0].quantity,
                "qx": fill.legs[1].quantity,
                "direction": policy.direction,
                "entry_session": session_idx,
                "exit_session": holding_exit_session(entry_session_index=session_idx, completed_holding_sessions=5),
                "auth_time": entry["auth_time"],
                "signal_identity": entry["signal_identity"],
                "forced_retry": 0,
            }

        # Daily MTM and carry costs across all open positions.
        ledger, period_unrealized = mark_unrealized_pnl(ledger, prices=latest_prices, prior_prices=prior_prices)
        total_delta_unrealized = money(total_delta_unrealized + period_unrealized)
        prior_prices = dict(latest_prices)

        sessions_since_mark = session_idx - last_mark_session
        stale = valuation_from_stale_age(sessions_since_mark=sessions_since_mark, max_stale_sessions=3)
        if stale.uncertifiable:
            uncertifiable = True
            uncertifiable_reason = stale.reason

        short_value = Decimal("0")
        for sec, qty in ledger.positions.items():
            if qty < 0 and sec in latest_prices:
                short_value += qty.copy_abs() * latest_prices[sec]
        borrow = borrow_accrual(short_value, annual_rate=Decimal("0.03"), days=1)
        financing = financing_accrual(ledger.free_cash, annual_rate=Decimal("0.04"), days=1)
        ledger = post_borrow_and_financing(ledger, borrow_cost=borrow, financing_cost=financing)

    eq_start = input_data.initial_equity
    eq_end = equity(ledger, latest_prices)

    if not any_authorized:
        uncertifiable = True
        if any_auth_rejected:
            first_reject = next((event for event in events if event.get("event") == "authorize_reject"), None)
            uncertifiable_reason = str(first_reject.get("reason")) if first_reject is not None else "NO_AUTHORIZED_CANDIDATES"
        else:
            uncertifiable_reason = "NO_AUTHORIZED_CANDIDATES"

    if open_positions:
        delist = delisting_valuation_status(has_terminal_executable_price=False)
        uncertifiable = True
        uncertifiable_reason = uncertifiable_reason or delist.reason

    certified, cert_reason = certification_status(
        prices_complete=not uncertifiable,
        provenance_complete=bool(input_data.signal_observation.evidence_digest),
        corporate_action_evidence_complete=True,
    )
    if not certified:
        uncertifiable = True
        if uncertifiable_reason is None:
            uncertifiable_reason = cert_reason

    residual = reconciliation_residual(
        equity_prev=eq_start,
        equity_now=eq_end,
        realized=ledger.realized_pnl,
        delta_unrealized=total_delta_unrealized,
        costs=ledger.fee_expense + ledger.borrow_expense + ledger.financing_expense,
        corp_action_cash=Decimal("0"),
        rounding_adjustment=ledger.rounding_expense,
    )
    try:
        validate_ledger_residual(residual)
    except ValueError:
        uncertifiable = True
        uncertifiable_reason = "INVALID_LEDGER"

    policy_id = policy_identity(
        {
            "pair_budget_fraction": str(caps.pair_budget_fraction),
            "gross_cap_fraction": str(caps.gross_cap_fraction),
            "net_cap_fraction": str(caps.net_cap_fraction),
            "security_cap_fraction": str(caps.security_cap_fraction),
        }
    )
    events_id = event_stream_identity(events)
    fills_id = fill_ledger_identity(fills)
    accounting_id = accounting_identity(
        {
            "free_cash": str(ledger.free_cash),
            "restricted_cash": str(ledger.restricted_cash),
            "positions": {k: str(v) for k, v in sorted(ledger.positions.items())},
            "basis_by_security": {k: str(v) for k, v in sorted(ledger.basis_by_security.items())},
            "fee_expense": str(ledger.fee_expense),
            "borrow_expense": str(ledger.borrow_expense),
            "financing_expense": str(ledger.financing_expense),
            "unrealized_pnl": str(ledger.unrealized_pnl),
        }
    )

    price_evidence_payload = []
    for session_idx in sorted(session_observations):
        for sec_id in sorted(session_observations[session_idx]):
            obs = session_observations[session_idx][sec_id]
            price_evidence_payload.append(
                {
                    "session": session_idx,
                    "security": sec_id,
                    "close": None if obs.close_price is None else str(obs.close_price),
                    "available_at": obs.available_at.isoformat(),
                    "close_event_time": obs.close_event_time.isoformat(),
                    "evidence_id": obs.evidence_id,
                }
            )

    identities = {
        "m8_signal_identity": event_stream_identity([{"pair": p.pair_id, "sig": p.signal_identity} for p in pair_inputs]),
        "decision_provenance_identity": decision_provenance_identity,
        "policy_identity": policy_id,
        "events_identity": events_id,
        "fills_identity": fills_id,
        "accounting_identity": accounting_id,
        "price_evidence_identity": event_stream_identity(price_evidence_payload),
        "calendar_identity": event_stream_identity([{"calendar": input_data.session_calendar_identity}]),
        "corporate_action_identity": event_stream_identity(
            [
                {
                    "session": item.session_index,
                    "security": item.security_id,
                    "type": item.action_type,
                    "value": str(item.value),
                    "evidence_complete": item.evidence_complete,
                }
                for item in (input_data.corporate_actions or [])
            ]
        ),
        "certification_identity": event_stream_identity(
            [
                {
                    "certified": not uncertifiable,
                    "reason": uncertifiable_reason,
                }
            ]
        ),
    }

    report = build_research_report(
        equity_start=eq_start,
        equity_end=eq_end,
        uncertifiable=uncertifiable,
        uncertifiable_reason=uncertifiable_reason,
        assumptions={
            "execution": "next-eligible-session-close-proxy",
            "cost_mode": "synthetic",
            "research_scope": "true",
        },
        identities=identities,
    )
    return EngineResult(events=events, fills=fills, report=report)
