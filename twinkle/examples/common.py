"""Shared runtime configuration; importing this module has no environment side effects."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_BASE_MODEL = 'Qwen/Qwen3.6-27B'
BASE_URL = 'https://www.modelscope.cn/twinkle'


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    base_model: str
    model_path: Optional[str]

    @property
    def template_model_id(self):
        return f'ms://{self.base_model}'


def load_config(*, mode):
    """Load only twinkle/.env at runtime, without overriding shell variables."""
    if mode not in ('base', 'lora'):
        raise ValueError('mode must be base or lora')

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env', override=False)
    api_key = os.environ.get('MODELSCOPE_TOKEN', '').strip()
    if not api_key:
        raise ValueError('MODELSCOPE_TOKEN is required; export it or set it in twinkle/.env.')

    base_model = os.environ.get('TWINKLE_BASE_MODEL', DEFAULT_BASE_MODEL).strip()
    if base_model.startswith('ms://'):
        base_model = base_model[len('ms://'):]
    if not base_model.strip():
        raise ValueError('TWINKLE_BASE_MODEL must not be empty.')

    # Explicit base evaluation must never pick up a checkpoint from the environment.
    model_path = None
    if mode == 'lora':
        model_path = os.environ.get('TWINKLE_MODEL_PATH', '').strip()
        if not model_path:
            raise ValueError('TWINKLE_MODEL_PATH is required in lora mode; no base fallback is used.')

    os.environ['MODELSCOPE_API_TOKEN'] = api_key
    return Config(api_key=api_key, base_model=base_model, model_path=model_path)
