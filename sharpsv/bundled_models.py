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
                chunk = source_handle.read(1024 * 1024)
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
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    request = urllib.request.Request(
        download_url,
        headers={
            "User-Agent": "SharpSV/0.1.0",
            "Accept": "application/octet-stream",
        },
    )

    started_at = time.time()
    hasher = hashlib.sha256()
    bytes_written = 0
    report_step = 10
    next_report_pct = report_step
    expected_size = int(model_info["size_bytes"])

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and "github.com/" in download_url:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request) as response, open(temp_path, "wb") as output_handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_handle.write(chunk)
                hasher.update(chunk)
                bytes_written += len(chunk)

                if expected_size > 0:
                    pct = int((bytes_written * 100) / expected_size)
                    if pct >= next_report_pct:
                        _emit_model_event(
                            stage_label,
                            f"download progress {min(pct, 100)}% "
                            f"({bytes_written / 1024 / 1024:.1f}/{expected_size / 1024 / 1024:.1f} MB)",
                        )
                        next_report_pct += report_step
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download bundled model asset from {download_url}: {exc}. "
            "Upload the release asset to GitHub Releases or pass a local override checkpoint."
        ) from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    _finalize_verified_file(temp_path, cache_path, bytes_written, hasher.hexdigest(), model_info)
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
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


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
