"""Download the shared DABstep workspace files from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from huggingface_hub import snapshot_download

REPO_ID = "adyen/DABstep"
REVISION = "main"
CONTEXT_PREFIX = "data/context"
CONTEXT_FILES = (
    "payments.csv",
    "payments-readme.md",
    "fees.json",
    "merchant_data.json",
    "merchant_category_codes.csv",
    "acquirer_countries.csv",
    "manual.md",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "external" / "dabstep"

SnapshotDownload = Callable[..., str]


class DownloadError(RuntimeError):
    """Raised when the requested DABstep context cannot be downloaded safely."""


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    name: str
    size_bytes: int
    status: Literal["downloaded", "skipped"]


def download_context(
    destination: Path = DEFAULT_DESTINATION,
    *,
    force: bool = False,
    snapshot_download_fn: SnapshotDownload = snapshot_download,
) -> tuple[DownloadedFile, ...]:
    """Download only DABstep's shared context files into ``destination``."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    files_to_download: list[str] = []
    for name in CONTEXT_FILES:
        target = destination / name
        if target.exists() and not target.is_file():
            raise DownloadError(f"expected a file but found a non-file path: {target}")
        if force or not target.is_file():
            files_to_download.append(name)

    if files_to_download:
        try:
            with TemporaryDirectory(prefix=".dabstep-", dir=destination.parent) as staging_dir:
                snapshot_root = Path(
                    snapshot_download_fn(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        revision=REVISION,
                        allow_patterns=[
                            f"{CONTEXT_PREFIX}/{name}" for name in files_to_download
                        ],
                        local_dir=staging_dir,
                        force_download=force,
                    )
                )
                staged_files = {
                    name: snapshot_root / CONTEXT_PREFIX / name for name in files_to_download
                }
                missing = [name for name, path in staged_files.items() if not path.is_file()]
                if missing:
                    missing_list = ", ".join(missing)
                    raise DownloadError(
                        f"Hugging Face snapshot did not contain required files: {missing_list}"
                    )

                destination.mkdir(parents=True, exist_ok=True)
                for name, source in staged_files.items():
                    target = destination / name
                    if target.is_file() and not force:
                        continue
                    os.replace(source, target)
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(
                f"failed to download {REPO_ID}/{CONTEXT_PREFIX}: {exc}"
            ) from exc

    downloaded_names = set(files_to_download)
    results: list[DownloadedFile] = []
    for name in CONTEXT_FILES:
        target = destination / name
        if not target.is_file():
            raise DownloadError(f"required file is missing after download: {target}")
        results.append(
            DownloadedFile(
                name=name,
                size_bytes=target.stat().st_size,
                status="downloaded" if name in downloaded_names else "skipped",
            )
        )
    return tuple(results)


def format_size(size_bytes: int) -> str:
    """Return a compact binary file size for command-line output."""

    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the shared data/context files from adyen/DABstep."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download and replace files even when they already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        files = download_context(force=args.force)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"DABstep workspace files: {DEFAULT_DESTINATION}")
    for file in files:
        print(f"- {file.name}: {format_size(file.size_bytes)} ({file.status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
