"""Small in-memory BM25 baseline for ``search_workspace``."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath

from data_management_agent.tools import Tool, ToolContext, ToolExecutionError
from data_management_agent.tools.contracts import SEARCH_WORKSPACE
from data_management_agent.tools.schemas import (
    ContentLocation,
    SearchEvidence,
    SearchWorkspaceInput,
    SearchWorkspaceOutput,
)
from data_management_agent.workspace import (
    ResourceKind,
    WorkspaceEntry,
    WorkspaceQuery,
    WorkspaceRef,
)

from ._files import decode_text, ensure_supported, normalize_extensions
from .evidence import EvidenceRecord, EvidenceStore

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+.+?$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class _Chunk:
    entry: WorkspaceEntry
    text: str
    location: ContentLocation


class SearchWorkspaceTool(Tool[SearchWorkspaceInput, SearchWorkspaceOutput]):
    contract = SEARCH_WORKSPACE

    def __init__(self, evidence_store: EvidenceStore | None = None) -> None:
        self._evidence_store = evidence_store or EvidenceStore()

    async def execute(
        self,
        arguments: SearchWorkspaceInput,
        *,
        context: ToolContext,
    ) -> SearchWorkspaceOutput:
        query_tokens = _tokenize(arguments.query)
        if not query_tokens:
            raise ToolExecutionError("search query must contain at least one searchable token")

        extensions, media_types = normalize_extensions(arguments.file_types)
        entries = await _list_all_files(
            context,
            WorkspaceQuery(
                path=arguments.scope,
                recursive=True,
                kinds=frozenset({ResourceKind.FILE}),
                extensions=extensions,
                media_types=media_types,
                metadata=arguments.metadata,
            ),
        )
        chunks: list[_Chunk] = []
        for entry in entries:
            ensure_supported(entry)
            content = await context.workspace.read_bytes(
                WorkspaceRef(resource_id=entry.resource_id)
            )
            chunks.extend(_chunks_for(entry, decode_text(content, path=str(entry.path))))

        scored = _bm25_rank(chunks, query_tokens)
        evidence: list[SearchEvidence] = []
        for score, chunk in scored[: arguments.top_k]:
            evidence_id = _evidence_id(chunk)
            self._evidence_store.put(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    resource_id=chunk.entry.resource_id,
                    path=chunk.entry.path,
                    location=chunk.location,
                )
            )
            evidence.append(
                SearchEvidence(
                    evidence_id=evidence_id,
                    resource_id=chunk.entry.resource_id,
                    path=chunk.entry.path,
                    snippet=_snippet(chunk.text, query_tokens),
                    location=chunk.location,
                    score=round(score, 6),
                )
            )
        return SearchWorkspaceOutput(evidence=tuple(evidence))


async def _list_all_files(context: ToolContext, query: WorkspaceQuery) -> list[WorkspaceEntry]:
    entries: list[WorkspaceEntry] = []
    cursor: str | None = None
    while True:
        page = await context.workspace.list_entries(query, cursor=cursor, limit=500)
        entries.extend(page.entries)
        if page.next_cursor is None:
            return entries
        cursor = page.next_cursor


def _chunks_for(entry: WorkspaceEntry, text: str) -> list[_Chunk]:
    extension = PurePosixPath(str(entry.path)).suffix.lower()
    if extension == ".md":
        return _markdown_chunks(entry, text)
    if extension == ".json":
        return _json_chunks(entry, text)
    return _csv_chunks(entry, text)


def _markdown_chunks(entry: WorkspaceEntry, text: str) -> list[_Chunk]:
    headings = list(_HEADING_PATTERN.finditer(text))
    if not headings:
        return _text_windows(entry, text)
    chunks: list[_Chunk] = []
    if headings[0].start() > 0:
        chunks.extend(_text_windows(entry, text[: headings[0].start()], offset=0))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.start() : end]
        chunks.extend(_text_windows(entry, section, offset=heading.start()))
    return chunks


def _text_windows(
    entry: WorkspaceEntry,
    text: str,
    *,
    offset: int = 0,
    window: int = 1_500,
) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    for start in range(0, len(text), window):
        end = min(start + window, len(text))
        if text[start:end].strip():
            chunks.append(
                _Chunk(
                    entry=entry,
                    text=text[start:end],
                    location=ContentLocation(
                        char_start=offset + start,
                        char_end=offset + end,
                    ),
                )
            )
    return chunks


def _json_chunks(entry: WorkspaceEntry, text: str) -> list[_Chunk]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"invalid JSON in {entry.path}: {exc}") from exc
    if not isinstance(value, list):
        return _text_windows(entry, text)
    return [
        _Chunk(
            entry=entry,
            text=json.dumps(item, ensure_ascii=False),
            location=ContentLocation(row_start=index, row_end=index + 1),
        )
        for index, item in enumerate(value)
    ]


def _csv_chunks(entry: WorkspaceEntry, text: str, *, rows_per_chunk: int = 50) -> list[_Chunk]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return []
    chunks: list[_Chunk] = []
    batch: list[list[str]] = []
    start = 0
    for row_index, row in enumerate(reader):
        batch.append(row)
        if len(batch) == rows_per_chunk:
            chunks.append(_csv_chunk(entry, header, batch, start, row_index + 1))
            batch = []
            start = row_index + 1
    if batch:
        chunks.append(_csv_chunk(entry, header, batch, start, start + len(batch)))
    return chunks


def _csv_chunk(
    entry: WorkspaceEntry,
    header: list[str],
    rows: list[list[str]],
    start: int,
    end: int,
) -> _Chunk:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return _Chunk(
        entry=entry,
        text=output.getvalue(),
        location=ContentLocation(row_start=start, row_end=end),
    )


def _bm25_rank(chunks: list[_Chunk], query_tokens: list[str]) -> list[tuple[float, _Chunk]]:
    if not chunks:
        return []
    documents = [_tokenize(f"{chunk.entry.path} {chunk.text}") for chunk in chunks]
    average_length = sum(map(len, documents)) / len(documents)
    document_frequency = Counter(
        token for document in documents for token in set(document) if token in query_tokens
    )
    query_counts = Counter(query_tokens)
    ranked: list[tuple[float, _Chunk]] = []
    for chunk, document in zip(chunks, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for token, query_frequency in query_counts.items():
            if frequencies[token] == 0:
                continue
            frequency = frequencies[token]
            inverse_frequency = math.log(
                1
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / average_length)
            score += query_frequency * inverse_frequency * (frequency * 2.5 / denominator)
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], str(item[1].entry.path)))
    return ranked


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def _snippet(text: str, query_tokens: list[str], *, limit: int = 500) -> str:
    lowered = text.casefold()
    positions = [lowered.find(token) for token in query_tokens]
    first_match = min((position for position in positions if position >= 0), default=0)
    start = max(0, first_match - limit // 4)
    end = min(len(text), start + limit)
    return text[start:end].strip()


def _evidence_id(chunk: _Chunk) -> str:
    identity = f"{chunk.entry.resource_id}:{chunk.location.model_dump_json()}:{chunk.text}"
    return f"ev-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
