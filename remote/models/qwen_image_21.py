# -*- coding: utf-8 -*-
"""Qwen-Image-2.1-Turbo-GGUF model plugin for colab-studio.

Builds flat ComfyUI API prompt graphs for:
- Text-to-Image (mode="t2i", including transparent RGBA generation)
- Image Editing / Multi-Reference (mode="edit", 1..10 reference images)
- Optional Qwen 3.5 9B Prompt Enhancer (refine_prompt=True) for both T2I and Edit
"""
from __future__ import annotations

import json
import re
from typing import Any

from .qwen_prompts import (
    QWEN21_EDIT_SYSTEM_PROMPT,
    QWEN21_T2I_SYSTEM_PROMPT,
    QWEN21_TRANSPARENT_T2I_SYSTEM_PROMPT,
)

GRAPH_VERSION = "qwen21_turbo_gguf_v1"

DEFAULT_QUANT = "INT8"
GGUF_QUANTS = (
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_XL",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q6_K_XL",
    "Q8_0",
    "F16",
)
NATIVE_QUANTS = {
    "INT8": ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", "diffusion_models"),
    "BF16": ("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors", "diffusion_models"),
}
VALID_QUANTS = GGUF_QUANTS + tuple(NATIVE_QUANTS.keys())

TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"

# Official Qwen-Image-2.1 2K (~4MP) aspect ratio presets; 1K (~1MP) scales dimensions by 0.5 (all multiples of 32).
ASPECT_RATIOS_2K: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

ASPECT_RATIOS_1K: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "4:3": (1216, 896),
    "3:4": (896, 1216),
    "3:2": (1280, 832),
    "2:3": (832, 1280),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
}

CORE_TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
ALT_TEXT_ENCODER = "qwen3vl_8b_w4a8.safetensors"
CORE_VAE = "qwen_image_2.1_vae_bf16.safetensors"

PE_MODELS = {
    ("t2i", "int8"): "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
    ("t2i", "w4a8"): "qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors",
    ("edit", "int8"): "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
    ("edit", "w4a8"): "qwen3.5_9b_qwen_image_2.1_pe_i2i_w4a8.safetensors",
}

# BatchImagesNode input slot order from ComfyUI workflow template (up to 10 images)
BATCH_IMAGE_SLOTS = (
    "images.image0",
    "images.image1",
    "images.image5",
    "images.image6",
    "images.image7",
    "images.image8",
    "images.image9",
    "images.image10",
    "images.image11",
    "images.image12",
)


def gguf_filename(quant: str | None = None) -> str:
    q = (quant or DEFAULT_QUANT).strip().upper()
    if q not in VALID_QUANTS:
        raise ValueError(f"不支援的量化等級 {quant!r}（可用：{', '.join(VALID_QUANTS)}）")
    if q in NATIVE_QUANTS:
        return NATIVE_QUANTS[q][1].rsplit("/", 1)[-1]
    return f"qwen-image-2.1-turbo-{q}.gguf"


def is_native_quant(quant: str | None = None) -> bool:
    return (quant or DEFAULT_QUANT).strip().upper() in NATIVE_QUANTS


def resolve_pe_filename(mode: str, pe_variant: str = "int8") -> str:
    m = "edit" if mode == "edit" else "t2i"
    v = (pe_variant or "int8").strip().lower()
    if (m, v) not in PE_MODELS:
        raise ValueError(f"不支援的 Prompt Enhancer 設定 mode={mode!r}, pe_variant={pe_variant!r}（可用：int8, w4a8）")
    return PE_MODELS[(m, v)]


def resolve_dimensions(job: dict[str, Any], manifest: dict[str, Any] | None = None) -> tuple[int, int]:
    """Return (width, height) rounded to multiples of 32."""
    manifest = manifest or {}
    if "width" in job and "height" in job:
        w = max(32, int(round(int(job["width"]) / 32) * 32))
        h = max(32, int(round(int(job["height"]) / 32) * 32))
        return w, h

    ar = str(job.get("aspect_ratio") or manifest.get("aspect_ratio") or "1:1").strip()
    res = str(job.get("resolution") or manifest.get("resolution") or "1K").strip().upper()
    if ar not in ASPECT_RATIOS_2K:
        raise ValueError(f"不支援的比例 {ar!r}（可用：{', '.join(ASPECT_RATIOS_2K)}，或直接指定 width 與 height）")
    if res in ("2K", "2048", "4MP"):
        return ASPECT_RATIOS_2K[ar]
    if res in ("1K", "1024", "1MP"):
        return ASPECT_RATIOS_1K[ar]
    raise ValueError(f"不支援的解析度等級 {res!r}（可用：1K, 2K，或直接指定 width 與 height）")


_RGBA_PREFIX = "This is an RGBA format image with transparency."
_RGBA_SUFFIX = "The image has an alpha channel and a transparent background."


def wrap_transparent_prompt(prompt: str) -> str:
    """Official Qwen-Image-2.1 RGBA transparent prompt wrapper."""
    p = prompt.strip()
    if p.startswith(_RGBA_PREFIX):
        return p
    return f"{_RGBA_PREFIX} {p}. {_RGBA_SUFFIX}"


def required_models_for_batch(manifest: dict[str, Any], cli_quant: str | None = None) -> list[tuple[str, str, str]]:
    """Return ordered list of (repo, filename_in_repo, dest_subfolder) required by a manifest batch."""
    top_quant = cli_quant or manifest.get("quant") or DEFAULT_QUANT
    top_pe_var = manifest.get("pe_variant") or "int8"
    top_te_var = manifest.get("text_encoder_variant") or "int8"

    quants_needed: list[str] = []
    te_needed: list[str] = []
    pe_needed: list[tuple[str, str]] = []

    for job in manifest.get("jobs", []):
        q = (cli_quant or job.get("quant") or top_quant).strip().upper()
        if q not in quants_needed:
            quants_needed.append(q)
        tv = str(job.get("text_encoder_variant") or top_te_var).strip().lower()
        if tv not in te_needed:
            te_needed.append(tv)
        if bool(job.get("refine_prompt", manifest.get("refine_prompt", False))):
            mode_norm = str(job.get("mode", "t2i")).strip().lower()
            mode = "edit" if mode_norm == "edit" else "t2i"
            pv = str(job.get("pe_variant") or top_pe_var).strip().lower()
            if (mode, pv) not in pe_needed:
                pe_needed.append((mode, pv))

    if not quants_needed:
        quants_needed.append(str(top_quant).strip().upper())
    if not te_needed:
        te_needed.append(str(top_te_var).strip().lower())

    files: list[tuple[str, str, str]] = []
    for q in quants_needed:
        if q in NATIVE_QUANTS:
            files.append(NATIVE_QUANTS[q])
        else:
            files.append(("unsloth/Qwen-Image-2.1-Turbo-GGUF", gguf_filename(q), "unet"))

    for tv in te_needed:
        te_file = ALT_TEXT_ENCODER if tv == "w4a8" else CORE_TEXT_ENCODER
        files.append(("Comfy-Org/Qwen-Image-2.1", f"text_encoders/{te_file}", "text_encoders"))
    files.append(("Comfy-Org/Qwen-Image-2.1", f"vae/{CORE_VAE}", "vae"))

    for mode, pv in pe_needed:
        pe_fn = resolve_pe_filename(mode, pv)
        files.append(("Comfy-Org/Qwen-Image-2.1", f"text_encoders/{pe_fn}", "text_encoders"))

    return files


_IMAGE_TAG = re.compile(r"<image[0-9]+>")
_BACKDROP_FILL = r"(?:white|checkerboard|transparency grid)"
_BACKDROP_MODIFIERS = r"(?:a|an|bright|clean|flat|nearly|plain|pure|solid|uniform)"
_VISIBLE_BACKDROP = re.compile(
    rf"\b{_BACKDROP_FILL}(?:\s+(?:studio|tiled|patterned|transparency-style))?\s+(?:background|backdrop)\b"
    rf"|\b(?:background|backdrop)\s+(?:is|appears|remains)\s+(?:{_BACKDROP_MODIFIERS},?\s+)*{_BACKDROP_FILL}\b"
    r"|\b(?:open|blank)\s+white space\b",
    re.IGNORECASE,
)
_SUBJECT_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])|\n+")
_SUBJECT_SENTENCE_LIMIT = 220


def uses_separate_edit_pe(job: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return (str(job.get("mode", "t2i")).strip().lower() == "edit" and len(job.get("inputs", [])) > 1
            and bool(job.get("refine_prompt", manifest.get("refine_prompt", False))))


def uses_separate_transparent_t2i_pe(job: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return (str(job.get("mode", "t2i")).strip().lower() == "t2i" and bool(job.get("transparent"))
            and bool(job.get("refine_prompt", manifest.get("refine_prompt", False))))


def build_pe_graph(full_graph: dict, prefix: str) -> dict:
    """Select only the ancestor nodes of the raw Prompt Enhancer preview."""
    selected = {}
    pending = ["33"]
    while pending:
        node_id = pending.pop()
        if node_id in selected:
            continue
        node = full_graph[node_id]
        selected[node_id] = node
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                pending.append(value[0])
    selected["33"] = {**selected["33"], "_meta": {"title": prefix}}
    return selected


# Keep the Edit helper available to existing history readers.
build_edit_pe_graph = build_pe_graph


def _pe_answer_text(raw: str | None) -> str:
    if not raw or not raw.strip():
        raise ValueError("Prompt Enhancer 輸出為空")
    answer = raw.strip()
    if answer.startswith(("{", "[")):
        try:
            response = json.loads(answer)
        except ValueError:
            raise ValueError("Prompt Enhancer 回覆的 JSON 格式無效") from None
        fields = response if isinstance(response, dict) else {}
        field = "output" if "rewritten_prompt" not in fields and "output" in fields else "rewritten_prompt"
        answer = fields.get(field)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Prompt Enhancer 的 {field} 為空或格式不符")
        answer = answer.strip()
    return answer


def normalize_edit_pe_answer(raw: str | None, requested: str, reference_count: int) -> str:
    """Use the answer text and validate the reference numbers it carries."""
    answer = _pe_answer_text(raw)
    expected = set(_IMAGE_TAG.findall(requested))
    actual = set(_IMAGE_TAG.findall(answer))
    allowed = {f"<image{n}>" for n in range(1, reference_count + 1)}
    if actual - allowed:
        raise ValueError("Prompt Enhancer 回覆含未提供的參考圖標記")
    if (expected & allowed) - actual:
        raise ValueError("Prompt Enhancer 回覆缺少請求指定的參考圖標記")
    return answer


def _unwrap_transparent_prompt(text: str) -> str:
    subject = text.strip()
    if subject.startswith(_RGBA_PREFIX):
        subject = subject.removeprefix(_RGBA_PREFIX).strip()
        if subject.endswith(_RGBA_SUFFIX):
            subject = subject.removesuffix(_RGBA_SUFFIX).rstrip().removesuffix(".").rstrip()
    return subject


def normalize_transparent_t2i_pe_answer(raw: str | None, requested: str) -> str:
    """Keep a concise subject description and exclude newly introduced visible backdrops."""
    answer = _unwrap_transparent_prompt(_pe_answer_text(raw))
    if not answer.strip(".").strip():
        raise ValueError("Prompt Enhancer 的透明主體描述為空")
    subject = _unwrap_transparent_prompt(requested)
    original = " ".join(subject.casefold().split())
    sentences = [sentence.strip() for sentence in _SUBJECT_SENTENCE_BREAK.split(answer) if sentence.strip()]
    clean = [
        sentence for sentence in sentences
        if all(" ".join(match.group().casefold().split()) in original
               for match in _VISIBLE_BACKDROP.finditer(sentence))
    ]
    if clean == sentences and len(answer) <= _SUBJECT_SENTENCE_LIMIT:
        return answer
    if not clean:
        raise ValueError("Prompt Enhancer 在透明 T2I 擴寫中加入可見背景")

    detail = next((sentence for sentence in clean
                   if len(sentence) <= _SUBJECT_SENTENCE_LIMIT and any(char.isalnum() for char in sentence)), None)
    if detail is None:
        raise ValueError("Prompt Enhancer 未提供可用的簡短透明主體描述")
    # Keep the brief when dropping text, including subjects the retained sentence might not name.
    separator = " " if subject.endswith((".", "!", "?", "。", "！", "？")) else ". "
    return f"{subject}{separator}{detail}"


def build_qwen21_graph(
    job: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    cli_quant: str | None = None,
    uploaded_inputs: list[str] | None = None,
) -> dict[str, Any]:
    """Build flat ComfyUI API prompt dictionary for a single Qwen-Image-2.1-Turbo job.

    Node IDs are deterministic strings:
      "10": UnetLoaderGGUF
      "11": QwenImage21Cache
      "12": CLIPLoader (qwen3vl_8b)
      "13": VAELoader (qwen_image_2.1_vae_bf16)
      "20".."29": LoadImage (for edit mode inputs 1..10)
      "30": CLIPLoader (Prompt Enhancer qwen3.5_9b_pe, only when refine_prompt=True)
      "31": BatchImagesNode (only when mode="edit" and refine_prompt=True)
      "32": TextGenerate (only when refine_prompt=True)
      "33": PreviewAny (only when refine_prompt=True; captures enhanced prompt in ComfyUI history)
      "40": TextEncodeQwenImage21
      "41": EmptyLatentImage (for t2i, or edit with custom_size=True)
      "50": ManualSigmas
      "51": KSamplerSelect
      "52": RandomNoise
      "53": CFGGuider
      "54": SamplerCustomAdvanced
      "60": VAEDecode
      "61": SaveImage
      "70".."79": JoinImageWithAlpha (Edit references for TextEncodeQwenImage21)
      "80".."89": ImageCompositeMasked (Edit references for Prompt Enhancer)
      "90": EmptyImage (white background for Edit Prompt Enhancer)
    """
    manifest = manifest or {}
    mode = str(job.get("mode", "t2i")).strip().lower()
    if mode not in ("t2i", "edit"):
        raise ValueError(f"job {job.get('id')!r}: mode 必須是 't2i' 或 'edit'（收到 {mode!r}）")

    inputs = uploaded_inputs if uploaded_inputs is not None else list(job.get("inputs", []))
    if mode == "edit" and not (1 <= len(inputs) <= 10):
        raise ValueError(f"job {job.get('id')!r}: edit 模式需要 1～10 張輸入圖片（收到 {len(inputs)} 張）")
    if mode == "t2i" and inputs:
        raise ValueError(f"job {job.get('id')!r}: t2i 模式不接受 inputs（若要改圖或參考圖請設 mode='edit'）")

    quant = (cli_quant or job.get("quant") or manifest.get("quant") or DEFAULT_QUANT).strip().upper()
    unet_file = gguf_filename(quant)

    te_var = str(job.get("text_encoder_variant") or manifest.get("text_encoder_variant") or "int8").strip().lower()
    clip_file = ALT_TEXT_ENCODER if te_var == "w4a8" else CORE_TEXT_ENCODER

    refine = bool(job.get("refine_prompt", manifest.get("refine_prompt", False)))
    thinking = bool(job.get("thinking", manifest.get("thinking", True)))
    max_length = int(job.get("pe_max_length", manifest.get("pe_max_length", 4096)))
    pe_var = str(job.get("pe_variant") or manifest.get("pe_variant") or "int8").strip().lower()

    raw_prompt = str(job.get("prompt", "")).strip()
    if not raw_prompt:
        raise ValueError(f"job {job.get('id')!r}: prompt 不能為空")

    transparent = bool(job.get("transparent", False))
    prompt_text = wrap_transparent_prompt(raw_prompt) if transparent else raw_prompt
    negative_prompt = str(job.get("negative_prompt", ""))
    seed = int(job.get("seed", 42))
    cfg = float(job.get("cfg", 1.0))
    sampler_name = str(job.get("sampler", "euler"))
    cache_device = str(job.get("cache_device", manifest.get("cache_device", "gpu")))
    cache_dtype = str(job.get("cache_dtype", manifest.get("cache_dtype", "default")))

    width, height = resolve_dimensions(job, manifest)

    if is_native_quant(quant):
        loader_node = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_file,
                "weight_dtype": "default",
            },
        }
    else:
        loader_node = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": unet_file,
            },
        }

    graph: dict[str, Any] = {
        "10": loader_node,
        "11": {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "model": ["10", 0],
                "device": cache_device,
                "dtype": cache_dtype,
            },
        },
        "12": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_file,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": CORE_VAE,
            },
        },
    }

    reference_nodes: list[tuple[str, str]] = []
    if mode == "edit":
        for idx, img_name in enumerate(inputs):
            load_id, rgba_id = str(20 + idx), str(70 + idx)
            reference_nodes.append((load_id, rgba_id))
            graph[load_id] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": img_name,
                },
            }
            # LoadImage returns 1 - alpha as its mask. JoinImageWithAlpha inverts it.
            graph[rgba_id] = {
                "class_type": "JoinImageWithAlpha",
                "inputs": {"image": [load_id, 0], "alpha": [load_id, 1]},
            }

    # Prompt Enhancer chain (optional)
    prompt_source: Any = prompt_text
    if refine:
        pe_file = resolve_pe_filename(mode, pe_var)
        pe_clip_type = "qwen_image" if mode == "edit" else "stable_diffusion"
        pe_prompt = prompt_text
        sys_prompt = QWEN21_EDIT_SYSTEM_PROMPT if mode == "edit" else QWEN21_T2I_SYSTEM_PROMPT
        if transparent and mode == "t2i":
            pe_prompt = _unwrap_transparent_prompt(raw_prompt)
            if not pe_prompt:
                raise ValueError(f"job {job.get('id')!r}: 透明主體描述不能為空")
            sys_prompt = QWEN21_TRANSPARENT_T2I_SYSTEM_PROMPT
        pres_pen = 0.0 if mode == "edit" else 1.5

        graph["30"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": pe_file,
                "type": pe_clip_type,
                "device": "default",
            },
        }

        tg_inputs: dict[str, Any] = {
            "clip": ["30", 0],
            "prompt": pe_prompt,
            "max_length": max_length,
            # ComfyUI expands DynamicCombo options from dotted API input names.
            "sampling_mode": "on",
            "sampling_mode.temperature": 1.0,
            "sampling_mode.top_k": 20,
            "sampling_mode.top_p": 0.95,
            "sampling_mode.min_p": 0.0,
            "sampling_mode.repetition_penalty": 1.0,
            "sampling_mode.presence_penalty": pres_pen,
            "sampling_mode.seed": seed,
            "thinking": thinking,
            "use_default_template": True,
            "mtp": "auto",
            "system_prompt": sys_prompt,
        }

        if mode == "edit":
            # PE consumes RGB. Use the transparency mask to blend white into each reference.
            graph["90"] = {
                "class_type": "EmptyImage",
                "inputs": {"width": 1, "height": 1, "batch_size": 1, "color": 0xFFFFFF},
            }
            batch_inputs: dict[str, Any] = {}
            for idx, (load_id, _) in enumerate(reference_nodes):
                rgb_id = str(80 + idx)
                graph[rgb_id] = {
                    "class_type": "ImageCompositeMasked",
                    "inputs": {
                        "destination": [load_id, 0], "source": ["90", 0], "mask": [load_id, 1],
                        "x": 0, "y": 0, "resize_source": True,
                    },
                }
                batch_inputs[BATCH_IMAGE_SLOTS[idx]] = [rgb_id, 0]
            graph["31"] = {
                "class_type": "BatchImagesNode",
                "inputs": batch_inputs,
            }
            tg_inputs["image"] = ["31", 0]

        graph["32"] = {
            "class_type": "TextGenerate",
            "inputs": tg_inputs,
        }
        graph["33"] = {
            "class_type": "PreviewAny",
            "inputs": {
                "source": ["32", 0],
            },
        }
        prompt_source = ["33", 0]

    # TextEncodeQwenImage21
    # In edit mode:
    # - edit_resolution controls how reference images are resized inside TextEncodeQwenImage21
    #   (default 1024 -> ~1MP preserving aspect ratio; 2048 -> ~4MP; 0 -> original size rounded to 32).
    edit_res = int(job.get("edit_resolution", 2048 if str(job.get("resolution", "")).upper() == "2K" else 1024))
    encode_inputs: dict[str, Any] = {
        "clip": ["12", 0],
        "prompt": prompt_source,
        "negative_prompt": negative_prompt,
        "resolution": edit_res if mode == "edit" else 1024,
    }
    if mode == "edit":
        encode_inputs["vae"] = ["13", 0]
        for idx, (_, rgba_id) in enumerate(reference_nodes):
            encode_inputs[f"images.image_{idx + 1}"] = [rgba_id, 0]

    graph["40"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": encode_inputs,
    }

    # Latent selection:
    # For T2I, or Edit with custom_size=True (or explicit width/height/aspect_ratio): use EmptyLatentImage.
    # For Edit with custom_size=False (default when user didn't force custom_size): use slot 2 of TextEncodeQwenImage21.
    custom_size = bool(job.get("custom_size", mode == "t2i" or "width" in job or "height" in job or "aspect_ratio" in job))
    if mode == "t2i" or custom_size:
        graph["41"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }
        latent_link = ["41", 0]
    else:
        latent_link = ["40", 2]

    graph["50"] = {
        "class_type": "ManualSigmas",
        "inputs": {
            "sigmas": str(job.get("sigmas", TURBO_SIGMAS)),
        },
    }
    graph["51"] = {
        "class_type": "KSamplerSelect",
        "inputs": {
            "sampler_name": sampler_name,
        },
    }
    graph["52"] = {
        "class_type": "RandomNoise",
        "inputs": {
            "noise_seed": seed,
        },
    }
    graph["53"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["11", 0],
            "positive": ["40", 0],
            "negative": ["40", 1],
            "cfg": cfg,
        },
    }
    graph["54"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["52", 0],
            "guider": ["53", 0],
            "sampler": ["51", 0],
            "sigmas": ["50", 0],
            "latent_image": latent_link,
        },
    }
    graph["60"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["54", 0],
            "vae": ["13", 0],
        },
    }
    graph["61"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["60", 0],
            "filename_prefix": f"cstudio_{job['id']}",
        },
    }

    return graph
