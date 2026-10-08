"""In-process evidence registry shared by search and targeted reads."""

from __future__ import annotations

from dataclasses import dataclass, field

from data_management_agent.tools.schemas import ContentLocation
from data_management_agent.workspace import WorkspacePath


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    resource_id: str
    path: WorkspacePath
    location: ContentLocation


@dataclass(slots=True)
class EvidenceStore:
    _records: dict[str, EvidenceRecord] = field(default_factory=dict)

    def put(self, record: EvidenceRecord) -> None:
        self._records[record.evidence_id] = record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)
