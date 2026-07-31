from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import gzip
import hashlib
import json
import os
import pickle
import re
import tempfile


CACHE_DATASET_INDEX_VERSION = 1
_CACHE_DATASET_INDEX_FORMAT = "grouped-pickle-v1"


class CacheDatasetIndexError(RuntimeError):
    """Raised when a serialized cache dataset index is invalid or incompatible."""


@dataclass(frozen=True)
class CacheSampleKey:
    """Unique key for one cached sample across multiple cache roots."""

    cache_index: int
    log_name: str
    token: str


@dataclass(frozen=True)
class CacheDatasetIndexSpec:
    """Inputs that determine which cache samples belong to a dataset."""

    cache_paths: Tuple[str, ...]
    required_file_names: Tuple[str, ...]
    log_names: Optional[Tuple[str, ...]]
    tokens: Optional[Tuple[str, ...]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cache_paths": list(self.cache_paths),
            "required_file_names": list(self.required_file_names),
            "log_names": None if self.log_names is None else list(self.log_names),
            "tokens": None if self.tokens is None else list(self.tokens),
        }

    @property
    def signature(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class CacheDatasetIndex:
    """An ordered, complete cache sample index shared by every training rank."""

    spec: CacheDatasetIndexSpec
    samples: List[CacheSampleKey]


def make_cache_dataset_index_spec(
    cache_paths: Sequence[str],
    required_file_names: Sequence[str],
    log_names: Optional[Sequence[str]] = None,
    tokens: Optional[Sequence[str]] = None,
) -> CacheDatasetIndexSpec:
    """Create a canonical index specification from dataset inputs."""
    normalized_cache_paths = tuple(
        os.path.abspath(os.path.expanduser(str(cache_path))) for cache_path in cache_paths
    )
    normalized_log_names = None if log_names is None else tuple(str(name) for name in log_names)
    normalized_tokens = None if tokens is None else tuple(sorted({str(token) for token in tokens}))
    return CacheDatasetIndexSpec(
        cache_paths=normalized_cache_paths,
        required_file_names=tuple(str(name) for name in required_file_names),
        log_names=normalized_log_names,
        tokens=normalized_tokens,
    )


def get_cache_dataset_index_path(
    index_dir: Path,
    split_name: str,
    spec: CacheDatasetIndexSpec,
) -> Path:
    """Return a deterministic path for one split and index specification."""
    safe_split_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", split_name).strip("._")
    if not safe_split_name:
        raise ValueError("split_name must contain at least one filename-safe character")
    filename = (
        f"{safe_split_name}.cache-index-v{CACHE_DATASET_INDEX_VERSION}."
        f"{spec.signature[:20]}.pkl.gz"
    )
    return Path(index_dir) / filename


def _validate_path_component(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheDatasetIndexError(f"Cache index {field_name} must be a non-empty string")
    if Path(value).name != value or value in {".", ".."}:
        raise CacheDatasetIndexError(
            f"Cache index {field_name} must be a single relative path component: {value!r}"
        )
    return value


def _iter_sample_groups(
    samples: Sequence[CacheSampleKey],
) -> Iterable[Tuple[int, str, List[str]]]:
    current_group: Optional[Tuple[int, str]] = None
    current_tokens: List[str] = []

    for sample in samples:
        group = (sample.cache_index, sample.log_name)
        if current_group is not None and group != current_group:
            yield current_group[0], current_group[1], current_tokens
            current_tokens = []
        current_group = group
        current_tokens.append(sample.token)

    if current_group is not None:
        yield current_group[0], current_group[1], current_tokens


def write_cache_dataset_index(path: Path, index: CacheDatasetIndex) -> None:
    """Atomically serialize a cache index without duplicating all sample records in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    header = {
        "version": CACHE_DATASET_INDEX_VERSION,
        "format": _CACHE_DATASET_INDEX_FORMAT,
        "signature": index.spec.signature,
        "spec": index.spec.to_payload(),
        "sample_count": len(index.samples),
    }

    try:
        with os.fdopen(file_descriptor, "wb") as raw_file:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=1,
                fileobj=raw_file,
                mtime=0,
            ) as compressed_file:
                pickler = pickle.Pickler(compressed_file, protocol=pickle.HIGHEST_PROTOCOL)
                pickler.dump(header)
                pickler.clear_memo()
                for group in _iter_sample_groups(index.samples):
                    pickler.dump(group)
                    pickler.clear_memo()
                pickler.dump(None)
            raw_file.flush()
            os.fsync(raw_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_header(
    header: Any,
    path: Path,
    expected_spec: CacheDatasetIndexSpec,
) -> int:
    if not isinstance(header, dict):
        raise CacheDatasetIndexError(f"Cache index header is not a dictionary: {path}")
    if header.get("version") != CACHE_DATASET_INDEX_VERSION:
        raise CacheDatasetIndexError(
            f"Unsupported cache index version in {path}: {header.get('version')!r}"
        )
    if header.get("format") != _CACHE_DATASET_INDEX_FORMAT:
        raise CacheDatasetIndexError(
            f"Unsupported cache index format in {path}: {header.get('format')!r}"
        )
    if header.get("signature") != expected_spec.signature:
        raise CacheDatasetIndexError(
            f"Cache index signature mismatch for {path}; expected {expected_spec.signature}, "
            f"found {header.get('signature')!r}"
        )
    if header.get("spec") != expected_spec.to_payload():
        raise CacheDatasetIndexError(f"Cache index specification mismatch for {path}")

    sample_count = header.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
        raise CacheDatasetIndexError(f"Invalid cache index sample count in {path}: {sample_count!r}")
    return sample_count


def load_cache_dataset_index(
    path: Path,
    expected_spec: CacheDatasetIndexSpec,
) -> CacheDatasetIndex:
    """Load and validate a cache index for an exact dataset specification."""
    path = Path(path)
    samples: List[CacheSampleKey] = []

    try:
        with gzip.open(path, "rb") as compressed_file:
            unpickler = pickle.Unpickler(compressed_file)
            sample_count = _validate_header(unpickler.load(), path, expected_spec)

            while True:
                group = unpickler.load()
                if group is None:
                    break
                if not isinstance(group, tuple) or len(group) != 3:
                    raise CacheDatasetIndexError(f"Invalid sample group in cache index {path}")

                cache_index, log_name, group_tokens = group
                if (
                    not isinstance(cache_index, int)
                    or isinstance(cache_index, bool)
                    or cache_index < 0
                    or cache_index >= len(expected_spec.cache_paths)
                ):
                    raise CacheDatasetIndexError(
                        f"Invalid cache root index in {path}: {cache_index!r}"
                    )
                log_name = _validate_path_component(log_name, "log_name")
                if not isinstance(group_tokens, list):
                    raise CacheDatasetIndexError(f"Invalid token group in cache index {path}")

                for token in group_tokens:
                    samples.append(
                        CacheSampleKey(
                            cache_index=cache_index,
                            log_name=log_name,
                            token=_validate_path_component(token, "token"),
                        )
                    )

        if len(samples) != sample_count:
            raise CacheDatasetIndexError(
                f"Cache index sample count mismatch for {path}: "
                f"expected {sample_count}, loaded {len(samples)}"
            )
    except CacheDatasetIndexError:
        raise
    except (EOFError, OSError, pickle.PickleError, TypeError, ValueError) as error:
        raise CacheDatasetIndexError(f"Failed to load cache dataset index {path}: {error}") from error

    return CacheDatasetIndex(spec=expected_spec, samples=samples)
