"""Validate and normalize public manifests on the client and in the Colab worker."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

if __package__:
    from .job_paths import JOB_ID, confined_path, relative_parts
    from .models import DEFAULT_MODEL, SUPPORTED_MODELS, qwen_image_21 as Q
else:
    from job_paths import JOB_ID, confined_path, relative_parts
    from models import DEFAULT_MODEL, SUPPORTED_MODELS, qwen_image_21 as Q


MAX_REFERENCES = len(Q.BATCH_IMAGE_SLOTS)
MAX_SEED = 2**64 - 1
MAX_JOB_TIMEOUT = 2**31 - 1
MODES = ("t2i", "edit")
RESOLUTIONS = {"1K": "1K", "1024": "1K", "1MP": "1K", "2K": "2K", "2048": "2K", "4MP": "2K"}
CACHE_DEVICES = ("auto", "gpu", "cpu", "off")
CACHE_DTYPES = ("default", "int8", "int4")
VARIANTS = ("int8", "w4a8")

# Pinned ComfyUI comfy/samplers.py, used by KSamplerSelect.
SAMPLERS = (
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
)
SIGMA_TOKEN = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
SIGMA_SEPARATOR = r"(?:\s*,\s*|\s+)"


def parse_manifest(data: str | bytes):
    def fields(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"JSON 欄位重複：{key!r}")
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise ValueError(f"JSON 數值必須有限：{value}")

    return json.loads(data, object_pairs_hook=fields, parse_constant=invalid_constant)


def _text(value):
    if not isinstance(value, str):
        raise ValueError("必須是字串")
    return value


def _nonblank(value):
    if not _text(value).strip():
        raise ValueError("不可為空白")
    return value


def _job_id(value):
    if not JOB_ID.fullmatch(_text(value)):
        raise ValueError("只接受 1～60 個英數、- 或 _")
    relative_parts(f"{value}.png")
    return value


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("必須是 JSON 布林值 true 或 false")
    return value


def _range(minimum, maximum, *, integer=True):
    types = (int,) if integer else (int, float)

    def validate(value):
        # A range comparison also rejects NaN, infinity and huge integers without converting them.
        if type(value) not in types or not minimum <= value <= maximum:
            kind = "整數" if integer else "有限數字"
            raise ValueError(f"必須是 {minimum}～{maximum} 的{kind}")
        return value

    return validate


def _choice(choices, normalize=str.lower):
    def validate(value):
        effective = normalize(_text(value).strip())
        if effective not in choices:
            raise ValueError(f"可用值為 {', '.join(choices)}")
        return effective

    return validate


_resolution_option = _choice(RESOLUTIONS, str.upper)


def normalize_resolution(value: str) -> str:
    return RESOLUTIONS[_resolution_option(value)]


def _sigmas(value):
    schedule = _text(value).strip()
    if schedule.startswith("[") and schedule.endswith("]"):
        schedule = schedule[1:-1].strip()
    tokens = re.split(SIGMA_SEPARATOR, schedule)
    if len(tokens) < 2 or any(not re.fullmatch(SIGMA_TOKEN, token) for token in tokens):
        raise ValueError("必須包含至少兩個小數或整數，以逗號或空白分隔")
    try:
        values = [struct.unpack("!f", struct.pack("!f", float(token)))[0] for token in tokens]
    except (OverflowError, struct.error):
        raise ValueError("超出 ComfyUI 的浮點數範圍") from None
    if values[0] == float("inf"):
        raise ValueError("超出 ComfyUI 的浮點數範圍")
    if values[-1] < 0 or not all(left > right for left, right in zip(values, values[1:])):
        raise ValueError("必須是非負且嚴格遞減的有限數值")
    return value


def _inputs(value):
    if not isinstance(value, list) or len(value) > MAX_REFERENCES:
        raise ValueError(f"必須是最多 {MAX_REFERENCES} 項的路徑陣列")
    for name in value:
        relative_parts(name)
    return value


_quant = _choice(Q.VALID_QUANTS, str.upper)
_variant = _choice(VARIANTS)
COMMON_FIELDS = {
    "quant": _quant, "refine_prompt": _boolean, "thinking": _boolean,
    "pe_max_length": _range(1, 32768), "pe_variant": _variant, "text_encoder_variant": _variant,
    "aspect_ratio": _choice(Q.ASPECT_RATIOS_2K), "resolution": normalize_resolution,
    "cache_device": _choice(CACHE_DEVICES), "cache_dtype": _choice(CACHE_DTYPES),
}
MANIFEST_FIELDS = {**COMMON_FIELDS, "model": _choice(SUPPORTED_MODELS), "$schema": _nonblank, "_gpu": _nonblank}
JOB_FIELDS = {
    **COMMON_FIELDS, "id": _job_id, "prompt": _nonblank, "negative_prompt": _text,
    "mode": _choice(MODES), "inputs": _inputs, "transparent": _boolean, "custom_size": _boolean,
    "width": _range(32, 4096), "height": _range(32, 4096), "edit_resolution": _range(0, 4096),
    "seed": _range(0, MAX_SEED), "cfg": _range(0, 100, integer=False), "timeout": _range(1, MAX_JOB_TIMEOUT),
    "sampler": _choice(SAMPLERS, lambda value: value), "sigmas": _sigmas,
}


def normalize_quant(value: str | None) -> str | None:
    return None if value is None else _quant(value)


def _normalize_fields(obj, rules, label, problems):
    result = {}
    for key, value in obj.items():
        if key not in rules:
            problems.append(f"{label}.{key}：不支援的欄位")
            continue
        try:
            result[key] = rules[key](value)
        except ValueError as exc:
            problems.append(f"{label}.{key}：{exc}")
    return result


def normalize_manifest(data, base: Path | None, cli_quant: str | None = None) -> tuple[dict, list[str]]:
    """Return normalized fields and validation problems without mutating the input."""
    if not isinstance(data, dict):
        return {}, ["manifest 頂層必須是 JSON object"]
    problems = []
    try:
        cli_quant = normalize_quant(cli_quant)
    except ValueError as exc:
        problems.append(f"--quant：{exc}")
    manifest = _normalize_fields({key: value for key, value in data.items() if key != "jobs"},
                                 MANIFEST_FIELDS, "manifest", problems)
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return manifest, [*problems, "manifest.jobs 必須是非空陣列"]
    plugin = SUPPORTED_MODELS[manifest.get("model", DEFAULT_MODEL)]
    seen = set()
    manifest["jobs"] = []
    for index, raw in enumerate(jobs, 1):
        label = f"jobs[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{label} 必須是 JSON object")
            continue
        before = len(problems)
        job = _normalize_fields(raw, JOB_FIELDS, label, problems)
        for name in ("id", "prompt"):
            if name not in raw:
                problems.append(f"{label}.{name}：必填")
        jid = job.get("id")
        if jid is not None:
            if jid in seen:
                problems.append(f"{label}.id：重複的任務 ID {jid}")
            seen.add(jid)
        if len(problems) != before:
            continue

        mode, refs = job.get("mode", "t2i"), job.get("inputs", [])
        if mode == "edit" and not refs:
            problems.append(f"{label}.inputs：Edit 需要 1～{MAX_REFERENCES} 張參考圖")
        if mode == "t2i" and refs:
            problems.append(f"{label}.inputs：T2I 不接受參考圖")
        if mode == "t2i" and ("edit_resolution" in job or "custom_size" in job):
            problems.append(f"{label}：edit_resolution 與 custom_size 適用於 Edit")
        if ("width" in job) != ("height" in job):
            problems.append(f"{label}：width 與 height 必須成對指定")
        if job.get("custom_size") is False and {"width", "height", "aspect_ratio"} & job.keys():
            problems.append(f"{label}：指定輸出尺寸時，custom_size 不可為 false")
        if job.get("sampler") == "ddim" and job.get("seed") == MAX_SEED:
            problems.append(f"{label}.seed：DDIM 會將種子加一，必須小於 {MAX_SEED}")
        if base is not None:
            for ref in refs:
                try:
                    confined_path(base, ref)
                except FileNotFoundError:
                    problems.append(f"{label}.inputs：找不到輸入圖片 {ref!r}")
                except (OSError, ValueError) as exc:
                    problems.append(f"{label}.inputs：無法使用輸入圖片 {ref!r}（{exc}）")
        if len(problems) == before:
            try:
                plugin.build_qwen21_graph(job, manifest=manifest, cli_quant=cli_quant)
            except ValueError as exc:
                problems.append(f"{label}：{exc}")
            else:
                manifest["jobs"].append(job)
    return manifest, problems
