"""Checkpoint metadata utilities — prediction type detection, safetensors header reading.

Supports safetensors single-file checkpoints from any source (CivitAI, HuggingFace,
kohya-ss, ComfyUI, etc.).  All detection runs on the JSON header only (~100-400 KB) —
the multi-GB tensor data is never loaded.

Detection methods are modeled after Forge/ComfyUI's huggingface_guess library and
A1111's sd_models_config.py, covering all known conventions:

  State dict tensor keys (Forge/ComfyUI):
    - "v_pred"              → v_prediction   (training tools embed a sentinel tensor)
    - "edm_vpred.sigma_max" → v_prediction   (EDM v-prediction variant)
    - "edm_mean" + "edm_std"→ edm            (Playground V2.5 etc.)
    - "ztsnr"               → zero terminal SNR flag (not a prediction type itself)

  Safetensors __metadata__ fields:
    - modelspec.predict_key          = "v" | "epsilon"   (SAI ModelSpec standard)
    - modelspec.prediction_type      = "v" | "epsilon"   (alternative ModelSpec key)
    - ss_v_parameterization          = "True"            (kohya-ss convention)

  Companion YAML sidecar (A1111/Forge):
    - model.params.parameterization  = "v"               (LDM format)
    - model.params.denoiser_config.params.scaling_config.target  endswith ".VScaling"
                                                          (SGM format)
"""

from __future__ import annotations

import json
import os
import struct

import log


def read_safetensors_header(path: str) -> dict | None:
    """Read just the JSON header from a safetensors file.

    Returns the parsed header dict (tensor names → metadata, plus __metadata__),
    or None if the file is not a valid safetensors file.
    """
    if not path.endswith('.safetensors'):
        return None
    try:
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if header_len > 100 * 1024 * 1024:  # sanity limit: 100 MB
                return None
            return json.loads(f.read(header_len))
    except Exception:
        return None


def read_safetensors_metadata(path: str) -> dict[str, str]:
    """Read the __metadata__ dict from a safetensors file header.

    Returns an empty dict if the file has no metadata or isn't a safetensors file.
    """
    header = read_safetensors_header(path)
    if header is None:
        return {}
    return header.get('__metadata__', {})


def detect_prediction_type(checkpoint_path: str) -> str | None:
    """Detect prediction type from a safetensors checkpoint without loading weights.

    Checks all known conventions used by Forge, ComfyUI, A1111, kohya, and
    the SAI ModelSpec standard.  Returns "v_prediction", "epsilon", or None
    if detection is inconclusive.

    Priority order:
    1. Tensor marker keys in state dict header (Forge/ComfyUI convention)
    2. Safetensors __metadata__ fields (ModelSpec, kohya)
    3. Companion YAML sidecar file (A1111/Forge)
    """
    header = read_safetensors_header(checkpoint_path)
    if header is not None:
        result = _detect_from_header(header)
        if result is not None:
            return result

    # Companion YAML sidecar — checked last as it's a user-created file
    result = _detect_from_yaml_sidecar(checkpoint_path)
    if result is not None:
        return result

    return None


def has_ztsnr_marker(checkpoint_path: str) -> bool:
    """Check if the checkpoint has a zero terminal SNR marker tensor.

    Some v-prediction models embed a "ztsnr" sentinel tensor to signal that
    the beta schedule should be rescaled to have zero terminal signal-to-noise
    ratio.  This is separate from the prediction type itself.
    """
    header = read_safetensors_header(checkpoint_path)
    if header is None:
        return False
    return 'ztsnr' in header


def _detect_from_header(header: dict) -> str | None:
    """Detect prediction type from a parsed safetensors header dict."""
    # --- Tensor marker keys (Forge/ComfyUI convention) ---
    # Training tools embed small sentinel tensors with these names.
    if 'v_pred' in header:
        return 'v_prediction'
    if 'edm_vpred.sigma_max' in header:
        return 'v_prediction'
    # EDM models (Playground V2.5 etc.) — different from both eps and v_pred,
    # but we map to None and let the caller handle it, since our scheduler
    # doesn't natively support EDM sigma scaling.
    if 'edm_mean' in header and 'edm_std' in header:
        log.warning("  Checkpoint: EDM model detected (edm_mean/edm_std markers). "
                    "EDM sigma scaling is not currently supported; "
                    "falling back to epsilon prediction.")
        return 'epsilon'

    # --- Safetensors __metadata__ fields ---
    metadata = header.get('__metadata__', {})

    # SAI ModelSpec standard: modelspec.predict_key
    predict_key = metadata.get('modelspec.predict_key', '')
    if predict_key == 'v':
        return 'v_prediction'
    elif predict_key == 'epsilon':
        return 'epsilon'

    # Alternative ModelSpec key: modelspec.prediction_type
    pred_type = metadata.get('modelspec.prediction_type', '')
    if pred_type in ('v', 'v_prediction'):
        return 'v_prediction'
    elif pred_type == 'epsilon':
        return 'epsilon'

    # kohya-ss convention: ss_v_parameterization
    v_param = metadata.get('ss_v_parameterization', '')
    if isinstance(v_param, str) and v_param.lower() == 'true':
        return 'v_prediction'

    return None


def _detect_from_yaml_sidecar(checkpoint_path: str) -> str | None:
    """Check for a companion YAML file alongside the checkpoint.

    A1111 and Forge both support placing a .yaml file with the same base name
    as the checkpoint (e.g., mymodel.yaml next to mymodel.safetensors).
    Two YAML structures are recognized:

    LDM format (Stability AI original):
        model.params.parameterization: "v"

    SGM format (generative-models):
        model.params.denoiser_config.params.scaling_config.target: "...VScaling"
    """
    yaml_path = os.path.splitext(checkpoint_path)[0] + '.yaml'
    if not os.path.isfile(yaml_path):
        return None

    try:
        import yaml
    except ImportError:
        return None

    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            return None

        params = config.get('model', {}).get('params', {})

        # LDM format: model.params.parameterization
        param = params.get('parameterization', '')
        if param == 'v':
            return 'v_prediction'
        elif param in ('eps', 'epsilon'):
            return 'epsilon'

        # SGM format: model.params.denoiser_config.params.scaling_config.target
        target = (params.get('denoiser_config', {})
                  .get('params', {})
                  .get('scaling_config', {})
                  .get('target', ''))
        if target.endswith('.VScaling'):
            return 'v_prediction'
        elif target.endswith('.EpsScaling'):
            return 'epsilon'

    except Exception:
        pass

    return None
