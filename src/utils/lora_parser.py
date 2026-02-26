"""Parse A1111/Forge-style LoRA tags from prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches <lora:name:weight> or <lora:name> (weight defaults to 1.0)
_LORA_TAG_RE = re.compile(r'<lora:([^:>]+)(?::([^>]+))?>')


@dataclass
class LoraSpec:
    """A single LoRA request parsed from a prompt tag or API field."""
    name: str
    weight: float = 1.0


def parse_lora_tags(prompt: str) -> tuple[str, list[LoraSpec]]:
    """Extract LoRA tags from a prompt string.

    Returns:
        (cleaned_prompt, lora_specs) where cleaned_prompt has all <lora:...>
        tags removed, and lora_specs is the list of parsed LoRA requests.
    """
    specs: list[LoraSpec] = []

    for match in _LORA_TAG_RE.finditer(prompt):
        name = match.group(1).strip()
        weight_str = match.group(2)
        weight = 1.0
        if weight_str is not None:
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = 1.0
        if name:
            specs.append(LoraSpec(name=name, weight=weight))

    cleaned = _LORA_TAG_RE.sub('', prompt).strip()
    return cleaned, specs
