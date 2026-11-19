# -*- coding: utf-8 -*-
"""Qwen-Image-2.1-Turbo-GGUF model plugin for colab-studio.

Builds flat ComfyUI API prompt graphs for:
- Text-to-Image (mode="t2i", including transparent RGBA generation)
- Image Editing / Multi-Reference (mode="edit", 1..10 reference images)
- Optional Qwen 3.5 9B Prompt Enhancer (refine_prompt=True) for both T2I and Edit
"""
from __future__ import annotations

import math
from typing import Any

from .qwen_prompts import QWEN21_EDIT_SYSTEM_PROMPT, QWEN21_T2I_SYSTEM_PROMPT

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


def wrap_transparent_prompt(prompt: str) -> str:
    """Official Qwen-Image-2.1 RGBA transparent prompt wrapper."""
    p = prompt.strip()
    prefix = "This is an RGBA format image with transparency."
    if p.startswith(prefix):
        return p
    return f"This is an RGBA format image with transparency. {p}. The image has an alpha channel and a transparent background."


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

    # Load input images for edit mode
    load_image_nodes: list[str] = []
    if mode == "edit":
        for idx, img_name in enumerate(inputs):
            nid = str(20 + idx)
            load_image_nodes.append(nid)
            graph[nid] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": img_name,
                },
            }

    # Prompt Enhancer chain (optional)
    prompt_source: Any = prompt_text
    if refine:
        pe_file = resolve_pe_filename(mode, pe_var)
        pe_clip_type = "qwen_image" if mode == "edit" else "stable_diffusion"
        sys_prompt = QWEN21_EDIT_SYSTEM_PROMPT if mode == "edit" else QWEN21_T2I_SYSTEM_PROMPT
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
            "prompt": prompt_text,
            "max_length": max_length,
            "sampling_mode": {
                "sampling_mode": "on",
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": pres_pen,
                "seed": seed,
            },
            "thinking": thinking,
            "use_default_template": True,
            "mtp": "auto",
            "system_prompt": sys_prompt,
        }

        if mode == "edit":
            batch_inputs: dict[str, Any] = {}
            for idx, nid in enumerate(load_image_nodes):
                batch_inputs[BATCH_IMAGE_SLOTS[idx]] = [nid, 0]
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
        for idx, nid in enumerate(load_image_nodes):
            encode_inputs[f"images.image_{idx + 1}"] = [nid, 0]

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
