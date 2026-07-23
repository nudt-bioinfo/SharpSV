import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


_BUNDLED_MODELS_DIR = Path(__file__).resolve().parent / "_bundle" / "models"
_BUNDLED_MANIFEST_PATH = _BUNDLED_MODELS_DIR / "manifest.json"
_DEFAULT_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/sharpsv-cache"))
_BUNDLED_CACHE_DIR = Path(
    os.environ.get("SHARPSV_MODEL_CACHE_DIR", str(_DEFAULT_CACHE_ROOT / "bundled-models"))
).expanduser()
_CACHE_SENTINEL_SUFFIX = ".sha256.ok"
_DOWNLOAD_PART_SUFFIX = ".part"
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_DOWNLOAD_REPORT_STEP_PCT = 10
_DEFAULT_DOWNLOAD_RETRIES = max(int(os.environ.get("SHARPSV_BUNDLE_DOWNLOAD_RETRIES", "8")), 1)
_DEFAULT_RETRY_DELAY_SECONDS = max(
    float(os.environ.get("SHARPSV_BUNDLE_DOWNLOAD_RETRY_DELAY_SEC", "5")),
    0.0,
)


with open(_BUNDLED_MANIFEST_PATH, "r", encoding="utf-8") as manifest_handle:
    _BUNDLED_MANIFEST = json.load(manifest_handle)

_BUNDLED_MODEL_LAYOUT = _BUNDLED_MANIFEST["models"]
_BUNDLED_LABEL_TO_STAGE = {
    model_info["label"]: stage_label for stage_label, model_info in _BUNDLED_MODEL_LAYOUT.items()
}


def is_bundled_model_ref(checkpoint_ref):
    return isinstance(checkpoint_ref, str) and checkpoint_ref in _BUNDLED_LABEL_TO_STAGE


def resolve_bundled_model_ref(stage_label):
    validate_bundled_model(stage_label)
    return _BUNDLED_MODEL_LAYOUT[stage_label]["label"]


def validate_bundled_model(stage_label):
    model_info = _BUNDLED_MODEL_LAYOUT.get(stage_label)
    if not model_info:
        raise KeyError(f"Unknown bundled model stage: {stage_label}")

    missing = [
        field
        for field in ("label", "filename", "sha256", "size_bytes", "override_flag")
        if field not in model_info
    ]
    if missing:
        raise ValueError(
            f"Bundled model manifest entry for {stage_label} is missing required fields: {missing}"
        )

    return _cache_path_for(model_info["filename"]), model_info


def resolve_runtime_checkpoint_path(checkpoint_ref):
    if not is_bundled_model_ref(checkpoint_ref):
        return str(Path(checkpoint_ref).expanduser().resolve())

    stage_label = _BUNDLED_LABEL_TO_STAGE[checkpoint_ref]
    return str(_ensure_bundled_model(stage_label))


def _ensure_bundled_model(stage_label):
    model_info = _BUNDLED_MODEL_LAYOUT[stage_label]
    cache_path = _cache_path_for(model_info["filename"])
    _BUNDLED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _is_valid_cached_model(cache_path, model_info):
        return cache_path

    packaged_path = (_BUNDLED_MODELS_DIR / model_info["filename"]).resolve()
    if packaged_path.exists():
        _emit_model_event(stage_label, f"using packaged fallback asset {packaged_path.name}")
        _copy_and_verify_local_asset(packaged_path, cache_path, model_info)
        return cache_path

    download_url = _download_url_for(model_info)
    _emit_model_event(
        stage_label,
        f"bundled checkpoint cache miss; downloading {model_info['filename']} "
        f"from {download_url}",
    )
    _download_and_verify(download_url, cache_path, model_info, stage_label)
    return cache_path


def _cache_path_for(filename):
    return (_BUNDLED_CACHE_DIR / filename).resolve()


def _cache_sentinel_path(cache_path):
    return cache_path.with_name(cache_path.name + _CACHE_SENTINEL_SUFFIX)


def _download_part_path(cache_path):
    return cache_path.with_name(cache_path.name + _DOWNLOAD_PART_SUFFIX)


def _is_valid_cached_model(cache_path, model_info):
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size != int(model_info["size_bytes"]):
        return False

    sentinel_path = _cache_sentinel_path(cache_path)
    if sentinel_path.exists():
        try:
            sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sentinel = None
        if sentinel and sentinel.get("sha256") == model_info["sha256"] and sentinel.get(
            "size_bytes"
        ) == int(model_info["size_bytes"]):
            return True

    if _sha256_of_file(cache_path) != model_info["sha256"]:
        return False

    _write_cache_sentinel(cache_path, model_info)
    return True


def _copy_and_verify_local_asset(source_path, cache_path, model_info):
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        hasher = hashlib.sha256()
        byte_count = 0
        with open(source_path, "rb") as source_handle, open(temp_path, "wb") as target_handle:
            while True:
                chunk = source_handle.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                target_handle.write(chunk)
                hasher.update(chunk)
                byte_count += len(chunk)
        _finalize_verified_file(temp_path, cache_path, byte_count, hasher.hexdigest(), model_info)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _download_and_verify(download_url, cache_path, model_info, stage_label):
    temp_path = _download_part_path(cache_path)
    started_at = time.time()
    expected_size = int(model_info["size_bytes"])
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    max_attempts = _DEFAULT_DOWNLOAD_RETRIES
    retry_delay = _DEFAULT_RETRY_DELAY_SECONDS
    bytes_written = _prepare_partial_download(temp_path, expected_size)
    next_report_pct = _next_report_threshold(bytes_written, expected_size)

    if bytes_written > 0:
        _emit_model_event(
            stage_label,
            f"resuming partial download at {bytes_written / 1024 / 1024:.1f}/"
            f"{expected_size / 1024 / 1024:.1f} MB",
        )

    for attempt in range(1, max_attempts + 1):
        if bytes_written >= expected_size:
            break

        try:
            bytes_written, next_report_pct = _download_attempt(
                download_url=download_url,
                temp_path=temp_path,
                token=token,
                bytes_written=bytes_written,
                expected_size=expected_size,
                next_report_pct=next_report_pct,
                stage_label=stage_label,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Failed to download bundled model asset from {download_url}: {exc}. "
                    f"Partial file kept at {temp_path}; rerun SharpSV to resume the download."
                ) from exc

            _emit_retry_event(
                stage_label=stage_label,
                attempt=attempt,
                max_attempts=max_attempts,
                bytes_written=bytes_written,
                expected_size=expected_size,
                reason=str(exc),
                retry_delay=retry_delay,
            )
            time.sleep(retry_delay)
            continue

        if bytes_written < expected_size:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Bundled model download for {model_info['filename']} stopped early at "
                    f"{bytes_written} of {expected_size} bytes after {max_attempts} attempts. "
                    f"Partial file kept at {temp_path}; rerun SharpSV to resume the download."
                )

            _emit_retry_event(
                stage_label=stage_label,
                attempt=attempt,
                max_attempts=max_attempts,
                bytes_written=bytes_written,
                expected_size=expected_size,
                reason="connection closed before the full asset arrived",
                retry_delay=retry_delay,
            )
            time.sleep(retry_delay)

    if bytes_written != expected_size:
        raise RuntimeError(
            f"Bundled model download for {model_info['filename']} did not complete: "
            f"have {bytes_written} of {expected_size} bytes cached at {temp_path}. "
            "Rerun SharpSV to continue the download."
        )

    _finalize_verified_file(temp_path, cache_path, bytes_written, _sha256_of_file(temp_path), model_info)
    _emit_model_event(
        stage_label,
        f"bundled checkpoint cached at {cache_path} in {time.time() - started_at:.1f}s",
    )


def _finalize_verified_file(temp_path, cache_path, bytes_written, file_sha256, model_info):
    expected_size = int(model_info["size_bytes"])
    if bytes_written != expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Bundled model size mismatch for {model_info['filename']}: "
            f"expected {expected_size} bytes, got {bytes_written}."
        )
    if file_sha256 != model_info["sha256"]:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Bundled model checksum mismatch for {model_info['filename']}: "
            f"expected {model_info['sha256']}, got {file_sha256}."
        )

    os.replace(temp_path, cache_path)
    _write_cache_sentinel(cache_path, model_info)


def _write_cache_sentinel(cache_path, model_info):
    sentinel_path = _cache_sentinel_path(cache_path)
    sentinel_path.write_text(
        json.dumps(
            {
                "filename": model_info["filename"],
                "sha256": model_info["sha256"],
                "size_bytes": int(model_info["size_bytes"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_of_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _prepare_partial_download(temp_path, expected_size):
    if not temp_path.exists():
        return 0

    current_size = temp_path.stat().st_size
    if current_size > expected_size:
        temp_path.unlink(missing_ok=True)
        return 0
    return current_size


def _next_report_threshold(bytes_written, expected_size):
    if expected_size <= 0:
        return _DOWNLOAD_REPORT_STEP_PCT

    pct = int((bytes_written * 100) / expected_size)
    return max(((pct // _DOWNLOAD_REPORT_STEP_PCT) + 1) * _DOWNLOAD_REPORT_STEP_PCT, _DOWNLOAD_REPORT_STEP_PCT)


def _build_download_request(download_url, token=None, range_start=None):
    request = urllib.request.Request(
        download_url,
        headers={
            "User-Agent": "SharpSV/0.1.0",
            "Accept": "application/octet-stream",
        },
    )
    if token and "github.com/" in download_url:
        request.add_header("Authorization", f"Bearer {token}")
    if range_start is not None and range_start > 0:
        request.add_header("Range", f"bytes={range_start}-")
    return request


def _download_attempt(
    download_url,
    temp_path,
    token,
    bytes_written,
    expected_size,
    next_report_pct,
    stage_label,
):
    range_start = bytes_written if bytes_written > 0 else None
    request = _build_download_request(download_url, token=token, range_start=range_start)
    file_mode = "ab" if range_start else "wb"

    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        if range_start and status != 206:
            _emit_model_event(
                stage_label,
                "download origin ignored the resume request; restarting from 0 MB",
            )
            temp_path.unlink(missing_ok=True)
            return 0, _next_report_threshold(0, expected_size)

        if range_start:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {range_start}-"
            if content_range and not content_range.startswith(expected_prefix):
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Bundled model server returned an unexpected resume offset: "
                    f"expected prefix {expected_prefix!r}, got {content_range!r}."
                )

        with open(temp_path, file_mode) as output_handle:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output_handle.write(chunk)
                bytes_written += len(chunk)

                if expected_size > 0:
                    pct = int((bytes_written * 100) / expected_size)
                    if pct >= next_report_pct:
                        _emit_model_event(
                            stage_label,
                            f"download progress {min(pct, 100)}% "
                            f"({bytes_written / 1024 / 1024:.1f}/{expected_size / 1024 / 1024:.1f} MB)",
                        )
                        next_report_pct += _DOWNLOAD_REPORT_STEP_PCT

    return bytes_written, next_report_pct


def _emit_retry_event(stage_label, attempt, max_attempts, bytes_written, expected_size, reason, retry_delay):
    _emit_model_event(
        stage_label,
        f"download interrupted at {bytes_written / 1024 / 1024:.1f}/{expected_size / 1024 / 1024:.1f} MB "
        f"({reason}); retrying {attempt + 1}/{max_attempts} in {retry_delay:.0f}s",
    )


def _download_url_for(model_info):
    base_url = os.environ.get("SHARPSV_BUNDLE_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/{model_info['filename']}"

    release_info = _BUNDLED_MANIFEST.get("release", {})
    if release_info.get("base_url"):
        return f"{release_info['base_url'].rstrip('/')}/{model_info['filename']}"

    repo = release_info.get("repo")
    tag = release_info.get("tag")
    if not repo or not tag:
        raise RuntimeError(
            "Bundled model manifest does not define a usable release location. "
            "Set SHARPSV_BUNDLE_BASE_URL or update sharpsv/_bundle/models/manifest.json."
        )
    return f"https://github.com/{repo}/releases/download/{tag}/{model_info['filename']}"


def _emit_model_event(stage_label, message):
    try:
        from .utils.console import emit

        emit(stage_label, message)
    except Exception:
        print(f"[SharpSV {stage_label}] {message}")
