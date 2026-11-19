# -*- coding: utf-8 -*-
"""Model registry for colab-studio."""
from __future__ import annotations

from . import qwen_image_21

SUPPORTED_MODELS = {
    "qwen-image-2.1-turbo": qwen_image_21,
    "qwen-image-2.1-turbo-gguf": qwen_image_21,
    "qwen21": qwen_image_21,
}
DEFAULT_MODEL = "qwen-image-2.1-turbo"


def get_model_plugin(name: str | None = None):
    key = (name or DEFAULT_MODEL).strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"不支援的模型 {name!r}（可用：{', '.join(SUPPORTED_MODELS)}）")
    return SUPPORTED_MODELS[key]
