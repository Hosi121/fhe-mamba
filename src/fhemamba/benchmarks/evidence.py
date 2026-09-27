"""Publish a separate evidence tree without disclosing local machine identities.

Measurements remain JSON/CSV files. Logs and measured source snapshots provide
provenance. Disposable orchestration scripts are excluded, not archived.
Original and published hashes are distinct and neither is silently rewritten.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from .io import read_object, sha256, write_json

_HOME = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+")
_PRIVATE_IP = re.compile(
    r"(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}"
    r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\w.])"
)
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_INSTANCE = re.compile(r"\bi-[0-9a-f]{8,17}\b")
_IDENTITY_KEYS = {"hostname", "host_name", "username", "machine_id", "ssm_session_id"}
_PRIVATE_KEYS = {"user_authorization", "user_steering"}
_DIRECT_SUFFIXES = {".json", ".csv", ".npz", ".npy"}
_ARCHIVED_JSON = {
    "evidence-manifest.json",  # Describes original bytes, not the public copy.
    "container-state.json",
    "compile_commands.json",
    "environment.json",
    "hardware.json",
    "cleanup.json",
    "local-cleanup.json",
    "preflight.json",
    "postflight.json",
    "notification.json",
}
_MANIFEST = "publication.json"
_BUNDLE = "provenance.tar.gz"
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def disposable_script(name: str, *, source_archive: bool = False) -> bool:
    """Keep measured package/build sources, not a study's one-off controllers."""
    path = PurePosixPath(name)
    is_script = path.suffix in {".py", ".pyc", ".sh", ".bash", ".sbatch"}
    is_script |= path.name == "profile-packed"
    if not is_script:
        return False
    if not source_archive:
        return True
    # These are versioned source snapshots that identify the measured code.
    # Loose job.py/campaign.py/cleanup.py copies do not qualify as source trees.
    source_directories = {"src", "native", "tests", "experiments", "deps"}
    return not any(part in source_directories for part in path.parts[:-1])


class PrivateArtifactError(ValueError):
    """Opaque profiler output remains local; only its identity is published."""


class Redactor:
    """Generic identities plus explicit, local-only literal replacements.

    This is a publication transform, not a secret detector. Review the allowlist
    of files and supply site-specific names through an ignored policy file.
    """

    def __init__(
        self,
        replacements: dict[str, str] | None = None,
        *,
        binary_suffixes: list[str] | None = None,
    ):
        self.replacements = dict(replacements or {})
        self.omissions: list[dict[str, str]] = []
        self.excluded_scripts = 0
        self.binary_suffixes = {".npz", ".npy", *(binary_suffixes or [])}
        if any(not suffix.startswith(".") for suffix in self.binary_suffixes):
            raise ValueError("binary_suffixes must contain extensions starting with a dot")
        if any(not key or not isinstance(value, str) for key, value in self.replacements.items()):
            raise ValueError("replacements must map non-empty strings to strings")

    def text(self, value: str) -> str:
        for old in sorted(self.replacements, key=len, reverse=True):
            value = value.replace(old, self.replacements[old])
        value = _HOME.sub("${HOME}", value)
        value = _PRIVATE_IP.sub("<private-ip>", value)
        value = _GPU_UUID.sub("<gpu-uuid>", value)
        return _INSTANCE.sub("<instance-id>", value)

    def json_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.json_value(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                public_key = self.text(key)
                if public_key in result:
                    raise ValueError("redaction would merge distinct JSON keys")
                if key.lower() in _IDENTITY_KEYS | _PRIVATE_KEYS:
                    result[public_key] = "<private>"
                else:
                    result[public_key] = self.json_value(item)
            return result
        return value

    def content(self, name: str, data: bytes, *, depth: int = 0) -> bytes:
        if depth > 4:
            raise ValueError(f"nested archive limit exceeded: {name}")
        if name.endswith((".nsys-rep", ".ncu-rep", ".sqlite")):
            raise PrivateArtifactError("opaque profiler data requires a reviewed text export")
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            members = []
            expanded = 0
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                for item in archive:
                    _safe_relative(item.name)
                    if item.isdir():
                        continue
                    if not item.isfile():
                        raise ValueError(f"only regular archive files are supported: {item.name}")
                    if disposable_script(item.name, source_archive=True):
                        self.excluded_scripts += 1
                        continue
                    expanded += item.size
                    if expanded > _MAX_ARCHIVE_BYTES:
                        raise ValueError(f"archive size limit exceeded: {name}")
                    stream = archive.extractfile(item)
                    assert stream is not None
                    public_name = self.text(item.name)
                    member_data = stream.read()
                    if "__pycache__" in PurePosixPath(item.name).parts or item.name.endswith(
                        ".pyc"
                    ):
                        self.omissions.append(
                            {
                                "member": public_name,
                                "sha256": sha256(member_data),
                                "reason": "cache",
                            }
                        )
                        continue
                    try:
                        public_data = self.content(public_name, member_data, depth=depth + 1)
                    except PrivateArtifactError as exc:
                        self.omissions.append(
                            {
                                "member": public_name,
                                "sha256": sha256(member_data),
                                "reason": str(exc),
                            }
                        )
                        continue
                    members.append((public_name, public_data))
            return _tar_bytes(members, compressed=not name.endswith(".tar"))
        if name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                expanded_data = stream.read(_MAX_ARCHIVE_BYTES + 1)
            if len(expanded_data) > _MAX_ARCHIVE_BYTES:
                raise ValueError(f"gzip size limit exceeded: {name}")
            return gzip.compress(self.content(name[:-3], expanded_data, depth=depth + 1), mtime=0)
        if Path(name).suffix in self.binary_suffixes:
            return data  # Reviewed numerical arrays; never rewritten as text.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"unreviewed binary format: {name}") from exc
        if name.endswith(".json"):
            try:
                value = json.loads(text)
            except ValueError:
                # Failed/truncated output remains failed/truncated evidence.
                return self.text(text).encode("utf-8")
            public_value = self.json_value(value)
            if public_value == value:
                return data
            # Preserve legacy NaN/Infinity literals in failed raw records.
            return (json.dumps(public_value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        return self.text(text).encode("utf-8")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")
    return path


def _tar_bytes(members: list[tuple[str, bytes]], *, compressed: bool = True) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        names = set()
        for name, data in sorted(members):
            _safe_relative(name)
            if name in names:
                raise ValueError(f"duplicate archive member: {name}")
            names.add(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644  # Archival scripts are not runnable entry points.
            archive.addfile(info, io.BytesIO(data))
    data = stream.getvalue()
    return gzip.compress(data, mtime=0) if compressed else data


def publish(source: Path, destination: Path, redactor: Redactor | None = None) -> dict[str, Any]:
    """Create a public copy. Never edit the source or overwrite an export."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir() or source == destination or source in destination.parents:
        raise ValueError("source must be a directory separate from destination")
    if destination.exists():
        raise ValueError("destination already exists; publish to a new directory")
    redactor = redactor or Redactor()
    direct = []
    bundled = []
    records = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not a publishable artifact: {path}")
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        name = path.relative_to(source).as_posix()
        if disposable_script(name):
            redactor.excluded_scripts += 1
            continue
        if name in {_MANIFEST, _BUNDLE}:
            raise ValueError("source is already a publication; use its original input")
        public_name = redactor.text(name)
        _safe_relative(public_name)
        data = path.read_bytes()
        try:
            public_data = redactor.content(name, data)
        except PrivateArtifactError as exc:
            records.append(
                {
                    "path": public_name,
                    "location": "private",
                    "original_sha256": sha256(data),
                    "original_bytes": len(data),
                    "reason": str(exc),
                }
            )
            continue
        is_direct = path.suffix in _DIRECT_SUFFIXES and path.name not in _ARCHIVED_JSON
        (direct if is_direct else bundled).append((public_name, public_data))
        records.append(
            {
                "path": public_name,
                "location": "file" if is_direct else _BUNDLE,
                "original_sha256": sha256(data),
                "original_bytes": len(data),
                "published_sha256": sha256(public_data),
                "published_bytes": len(public_data),
                "transformed": data != public_data,
            }
        )
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("redaction would merge distinct artifact paths")
    archive = _tar_bytes(bundled)
    manifest = {
        "schema_version": 1,
        "kind": "fhemamba-public-evidence",
        "description": "Public derivative; original hashes refer to unpublished raw bytes.",
        "binary_suffixes_preserved_without_text_redaction": sorted(redactor.binary_suffixes),
        "omitted_archive_members": redactor.omissions,
        "excluded_disposable_scripts": redactor.excluded_scripts,
        "bundle": {"path": _BUNDLE, "sha256": sha256(archive), "bytes": len(archive)},
        "files": records,
    }
    destination.mkdir(parents=True)
    for name, data in direct:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / _BUNDLE).write_bytes(archive)
    write_json(destination / _MANIFEST, manifest)
    return manifest


def verify(root: Path) -> dict[str, Any]:
    """Verify public file/member hashes, not historical performance assertions."""
    manifest = read_object(root / _MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "fhemamba-public-evidence":
        raise ValueError("unsupported evidence manifest")
    bundle = manifest["bundle"]
    bundle_path = root / _safe_relative(bundle["path"])
    data = bundle_path.read_bytes()
    if len(data) != bundle["bytes"] or sha256(data) != bundle["sha256"]:
        raise ValueError("provenance bundle hash mismatch")
    archived = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for item in archive:
            _safe_relative(item.name)
            if not item.isfile() or item.name in archived:
                raise ValueError("invalid or duplicate provenance member")
            stream = archive.extractfile(item)
            assert stream is not None
            archived[item.name] = stream.read()
    names = set()
    expected_archived = set()
    for record in manifest["files"]:
        name = record["path"]
        _safe_relative(name)
        if name in names:
            raise ValueError("duplicate manifest entry")
        names.add(name)
        if record["location"] == "file":
            path = root / name
            if path.is_symlink() or root.resolve() not in path.resolve().parents:
                raise ValueError(f"artifact escapes publication: {name}")
            data = path.read_bytes()
        elif record["location"] == _BUNDLE:
            expected_archived.add(name)
            data = archived[name]
        elif record["location"] == "private":
            continue
        else:
            raise ValueError("unknown artifact location")
        if len(data) != record["published_bytes"] or sha256(data) != record["published_sha256"]:
            raise ValueError(f"published artifact hash mismatch: {name}")
    if set(archived) != expected_archived:
        raise ValueError("unlisted provenance members")
    return {"passed": True, "files": len(names), "archived_files": len(archived)}


def extract_provenance(root: Path, destination: Path) -> None:
    """Read-only inspection material, extracted safely to a new local directory."""
    verify(root)
    if destination.exists():
        raise ValueError("destination already exists")
    destination.mkdir(parents=True)
    with tarfile.open(root / _BUNDLE, mode="r:gz") as archive:
        for item in archive:
            target = destination / _safe_relative(item.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(item)
            assert stream is not None
            target.write_bytes(stream.read())
