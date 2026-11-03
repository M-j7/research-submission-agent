"""Explicit live context policy; no network or tokenizer downloads on startup."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from .loop import LoopSettings
from .prompt import PromptManager


@dataclass(frozen=True)
class RuntimeConfig:
    prompt: PromptManager
    settings: LoopSettings
    metadata: dict


def build_runtime_config(values, project):
    # Verified model-studio/qwen3-8-flash context window. Other models require
    # explicit operator configuration rather than inheriting this capability.
    models={values.get('LLM_MODEL')}
    models.update(v for k,v in values.items() if k.startswith('AGENT_') and k.endswith('_MODEL') and v)
    window=values.get('AGENT_CONTEXT_WINDOW_TOKENS')
    if window is None:
        if models!={'qwen3.8-flash'}:
            raise ValueError('unverified model window; set AGENT_CONTEXT_WINDOW_TOKENS explicitly')
        window=1000000
    window=int(window)
    if window<4096:
        raise ValueError('context window must leave room for prompts and output')
    path=Path(values.get('AGENT_TOKENIZER_PATH') or '.cache/tokenizers/qwen3.8-flash-next/tokenizer.json')
    if not path.is_absolute(): path=Path(project)/path
    if not path.is_file():
        raise FileNotFoundError('local tokenizer missing; configure AGENT_TOKENIZER_PATH, no byte fallback in live mode')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    def count(text):
        # Public Flash-Next tokenizer is a local estimate, NOT a claim of exact
        # proprietary API tokenization/chat-template or billing equivalence.
        return math.ceil(len(tokenizer.encode(text,add_special_tokens=False).ids)*1.1)
    budget=window*7//10
    prompt=PromptManager(input_budget=budget,token_counter=count)
    prompt.compression_trigger_tokens=window*6//10
    settings=LoopSettings(max_total_tokens_estimate=int(values.get('AGENT_MAX_TOTAL_TOKENS') or
                         (budget+2048)*LoopSettings().max_steps))
    metadata={'model':values.get('LLM_MODEL'),'context_window_tokens':window,'input_budget_tokens':budget,
              'compression_trigger_tokens':prompt.compression_trigger_tokens,
              'counting_mode':'local_qwen_tokenizer_estimate_json_with_10pct_margin',
              'tokenizer_source':'Qwen/Qwen3.8-Flash-Next (public surrogate for API model)',
              'tokenizer_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'max_total_tokens_estimate':settings.max_total_tokens_estimate,
              'max_steps':settings.max_steps,'max_tool_calls':settings.max_tool_calls,
              'compression_timeout_s':30,'compression_batch_bytes':24000}
    return RuntimeConfig(prompt,settings,metadata)
