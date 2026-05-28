import os
from pathlib import Path


_BUNDLED_MODELS_DIR = Path(__file__).resolve().parent / "_bundle" / "models"
_BUNDLED_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/sharpsv-cache")) / "bundled-models"

_BUNDLED_MODEL_LAYOUT = {
    "stage-1": {
        "kind": "file",
        "filename": "stage1.model.bin",
        "label": "bundled://stage1",
        "override_flag": "--stage1-model",
    },
    "stage-2": {
        "kind": "chunked",
        "part_prefix": "stage2.model.bin.part-",
        "materialized_name": "stage2.model.bin",
        "label": "bundled://stage2",
        "override_flag": "--stage2-model",
    },
}

_BUNDLED_LABEL_TO_STAGE = {
    model_info["label"]: stage_label for stage_label, model_info in _BUNDLED_MODEL_LAYOUT.items()
}


def is_bundled_model_ref(checkpoint_ref):
    return isinstance(checkpoint_ref, str) and checkpoint_ref in _BUNDLED_LABEL_TO_STAGE


def resolve_bundled_model_ref(stage_label):
    validate_bundled_model(stage_label)
    return _BUNDLED_MODEL_LAYOUT[stage_label]["label"]


def validate_bundled_model(stage_label):
    model_info = _BUNDLED_MODEL_LAYOUT[stage_label]
    if model_info["kind"] == "file":
        model_path = (_BUNDLED_MODELS_DIR / model_info["filename"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return model_path, model_info

    part_paths = _chunk_part_paths(stage_label)
    if not part_paths:
        raise FileNotFoundError(
            f"No bundled parts found under {_BUNDLED_MODELS_DIR} for {model_info['label']}"
        )
    return part_paths[0], model_info


def resolve_runtime_checkpoint_path(checkpoint_ref):
    if not is_bundled_model_ref(checkpoint_ref):
        return str(Path(checkpoint_ref).expanduser().resolve())

    stage_label = _BUNDLED_LABEL_TO_STAGE[checkpoint_ref]
    model_info = _BUNDLED_MODEL_LAYOUT[stage_label]
    if model_info["kind"] == "file":
        model_path = (_BUNDLED_MODELS_DIR / model_info["filename"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return str(model_path)

    return str(_materialize_chunked_model(stage_label))


def _chunk_part_paths(stage_label):
    model_info = _BUNDLED_MODEL_LAYOUT[stage_label]
    return sorted(_BUNDLED_MODELS_DIR.glob(f"{model_info['part_prefix']}*"))


def _materialize_chunked_model(stage_label):
    model_info = _BUNDLED_MODEL_LAYOUT[stage_label]
    part_paths = _chunk_part_paths(stage_label)
    if not part_paths:
        raise FileNotFoundError(
            f"No bundled parts found under {_BUNDLED_MODELS_DIR} for {model_info['label']}"
        )

    expected_size = sum(part_path.stat().st_size for part_path in part_paths)
    _BUNDLED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _BUNDLED_CACHE_DIR / model_info["materialized_name"]
    if output_path.exists() and output_path.stat().st_size == expected_size:
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "wb") as target_handle:
        for part_path in part_paths:
            with open(part_path, "rb") as source_handle:
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    target_handle.write(chunk)
    os.replace(temp_path, output_path)
    return output_path
