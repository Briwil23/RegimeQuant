from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from regimequant.universe.contracts import CorporateActionContinuity, SecurityAliasRecord, SecurityIdentityEvidence


class IdentityAdapterError(ValueError):
    """Raised when the M1/M6 identity adapter cannot prove admissible identity."""


def _normalize_alias_records(records: Any) -> tuple[SecurityAliasRecord, ...]:
    normalized: list[SecurityAliasRecord] = []
    for record in records or ():
        if isinstance(record, SecurityAliasRecord):
            normalized.append(record)
        elif isinstance(record, dict):
            normalized.append(
                SecurityAliasRecord(
                    canonical_security_id=record.get("canonical_security_id", ""),
                    symbol=record.get("symbol", ""),
                    listing_venue=record.get("listing_venue", ""),
                    valid_from=record.get("valid_from"),
                    valid_to=record.get("valid_to"),
                    verified_at=record.get("verified_at"),
                    evidence_provenance=record.get("evidence_provenance", ""),
                    source_authority=record.get("source_authority", "exchange"),
                    evidence_digest=record.get("evidence_digest", ""),
                )
            )
        else:
            raise IdentityAdapterError("alias records must be SecurityAliasRecord or dict")
    return tuple(normalized)


def _normalize_continuity_records(records: Any) -> tuple[CorporateActionContinuity, ...]:
    normalized: list[CorporateActionContinuity] = []
    for record in records or ():
        if isinstance(record, CorporateActionContinuity):
            normalized.append(record)
        elif isinstance(record, dict):
            normalized.append(
                CorporateActionContinuity(
                    predecessor_security_id=record.get("predecessor_security_id", ""),
                    successor_security_id=record.get("successor_security_id", ""),
                    event_kind=record.get("event_kind", ""),
                    effective_at=record.get("effective_at"),
                    verified_at=record.get("verified_at"),
                    source_authority=record.get("source_authority", "exchange"),
                    evidence_provenance=record.get("evidence_provenance", ""),
                    evidence_digest=record.get("evidence_digest", ""),
                )
            )
        else:
            raise IdentityAdapterError("continuity records must be CorporateActionContinuity or dict")
    return tuple(normalized)


def _normalize_evidence_records(records: Any) -> tuple[SecurityIdentityEvidence, ...]:
    normalized: list[SecurityIdentityEvidence] = []
    for record in records or ():
        if isinstance(record, SecurityIdentityEvidence):
            normalized.append(record)
        elif isinstance(record, dict):
            normalized.append(
                SecurityIdentityEvidence(
                    security_id=record.get("security_id", ""),
                    ticker=record.get("ticker", ""),
                    aliases=record.get("aliases", ()),
                    authority=record.get("authority", "exchange"),
                    verified_at=record.get("verified_at"),
                    source=record.get("source", "exchange"),
                    evidence_digest=record.get("evidence_digest", ""),
                    status=record.get("status", "verified"),
                    ambiguous=bool(record.get("ambiguous", False)),
                    issuer_id=record.get("issuer_id"),
                )
            )
        else:
            raise IdentityAdapterError("evidence records must be SecurityIdentityEvidence or dict")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class VerifiedIdentityAdapter:
    security_id: str
    ticker: str
    issuer_id: str | None
    alias_records: tuple[SecurityAliasRecord, ...]
    continuity_records: tuple[CorporateActionContinuity, ...]
    evidence_records: tuple[SecurityIdentityEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "security_id", str(self.security_id).strip())
        object.__setattr__(self, "ticker", str(self.ticker).strip())
        object.__setattr__(self, "issuer_id", None if self.issuer_id is None else str(self.issuer_id).strip())
        object.__setattr__(self, "alias_records", _normalize_alias_records(self.alias_records))
        object.__setattr__(self, "continuity_records", _normalize_continuity_records(self.continuity_records))
        object.__setattr__(self, "evidence_records", _normalize_evidence_records(self.evidence_records))
        if not self.security_id:
            raise IdentityAdapterError("security_id is required")
        if not self.ticker:
            raise IdentityAdapterError("ticker is required")
        if not self.evidence_records:
            raise IdentityAdapterError("verified identity evidence is required")

        by_security = {record.security_id: record for record in self.evidence_records}
        if self.security_id not in by_security:
            raise IdentityAdapterError("security_id has no verified evidence")

        evidence = by_security[self.security_id]
        if evidence.status != "verified":
            raise IdentityAdapterError("identity evidence must be verified")
        if evidence.ambiguous:
            raise IdentityAdapterError("verified identity evidence cannot be ambiguous")
        if evidence.ticker != self.ticker:
            raise IdentityAdapterError("ticker must match verified evidence")
        if self.issuer_id is not None and evidence.issuer_id is not None and evidence.issuer_id != self.issuer_id:
            raise IdentityAdapterError("issuer_id must match verified evidence")

        seen_tickers: dict[str, str | None] = {}
        for record in self.evidence_records:
            if record.status != "verified":
                raise IdentityAdapterError("only verified evidence is admissible")
            if record.ambiguous:
                raise IdentityAdapterError("ambiguous evidence is not admissible")
            issuer = record.issuer_id or self.issuer_id
            prior = seen_tickers.get(record.ticker)
            if prior is not None and prior != issuer:
                raise IdentityAdapterError("same ticker cannot be reused across different issuers")
            seen_tickers[record.ticker] = issuer

        if self.alias_records and not any(record.canonical_security_id == self.security_id for record in self.alias_records):
            raise IdentityAdapterError("alias records must include the canonical security_id")

    def is_admissible_at(self, as_of: pd.Timestamp) -> bool:
        as_of_utc = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo is not None else pd.Timestamp(as_of, tz="UTC")
        for alias in self.alias_records:
            valid_from = pd.Timestamp(alias.valid_from).tz_convert("UTC")
            valid_to = pd.Timestamp(alias.valid_to).tz_convert("UTC") if alias.valid_to is not None else None
            if alias.canonical_security_id == self.security_id and valid_from <= as_of_utc and (valid_to is None or as_of_utc <= valid_to):
                return True
        return False

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "security_id": self.security_id,
            "ticker": self.ticker,
            "issuer_id": self.issuer_id,
            "evidence_count": len(self.evidence_records),
            "alias_count": len(self.alias_records),
            "continuity_count": len(self.continuity_records),
        }
