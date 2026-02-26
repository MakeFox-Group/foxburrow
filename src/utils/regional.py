"""Regional prompting: prompt parsing, geometry, and mask generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


_REGIONAL_RE = re.compile(r'\bADDCOL\b|\bADDROW\b|\bADDBASE\b|\bADDCOMM\b')
_ADDCOL_KW_RE = re.compile(r'\bADDCOL(?::(\d+(?:\.\d+)?))?\b')
_ADDROW_KW_RE = re.compile(r'\bADDROW(?::(\d+(?:\.\d+)?))?\b')


@dataclass
class RegionInfo:
    prompt: str
    negative: str  # per-region negative
    x0: float      # normalized [0,1]
    y0: float
    x1: float
    y1: float


@dataclass
class RegionalPromptResult:
    regions: list[RegionInfo]
    base_prompt: str | None  # from ADDBASE, or None
    negative_prompt: str     # original shared negative (for pooled, empty-check, etc.)
    base_ratio: float        # blend ratio for base (default 0.2)
    has_per_region_neg: bool  # True if negatives differ across regions


def detect_regional_keywords(prompt: str) -> bool:
    """Lightweight check for ADDCOL / ADDROW / ADDBASE / ADDCOMM keywords."""
    return bool(_REGIONAL_RE.search(prompt))


def _extract_keyword(prompt: str, keyword: str) -> tuple[str | None, str]:
    """Extract text before a keyword, return (extracted_text, remaining_prompt).

    If keyword not found, returns (None, original_prompt).
    """
    pattern = re.compile(r'\b' + keyword + r'\b')
    match = pattern.search(prompt)
    if not match:
        return None, prompt
    before = prompt[:match.start()].strip()
    after = prompt[match.end():].strip()
    return (before if before else None), after


def _split_on_keyword(text: str, keyword_re: re.Pattern) -> tuple[list[str], list[float]]:
    """Split text on keyword matches, extracting optional :ratio suffixes.

    Returns (segments, ratios) where segments[i] corresponds to ratios[i].
    The first segment (before any keyword) gets ratio 1.0.
    """
    matches = list(keyword_re.finditer(text))
    if not matches:
        return [text.strip()], [1.0]

    segments: list[str] = []
    ratios: list[float] = []

    # First segment: text before first keyword (ratio 1.0)
    segments.append(text[:matches[0].start()].strip())
    ratios.append(1.0)

    for i, m in enumerate(matches):
        ratio = float(m.group(1)) if m.group(1) else 1.0
        ratios.append(ratio)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append(text[start:end].strip())

    return segments, ratios


def _ratios_to_spans(ratios: list[float]) -> list[tuple[float, float]]:
    """Convert ratios like [1, 2, 1] to normalized spans [(0, 0.25), (0.25, 0.75), (0.75, 1.0)]."""
    total = sum(ratios)
    if total <= 0:
        total = float(len(ratios))
    spans: list[tuple[float, float]] = []
    pos = 0.0
    for r in ratios:
        span = r / total
        spans.append((pos, pos + span))
        pos += span
    return spans


def _split_negative_like_positive(
    negative: str,
    pos_has_row: bool,
    pos_has_col: bool,
    n_regions: int,
    grid_shape: list[int] | None = None,
) -> list[str]:
    """Split the negative prompt to match the positive prompt's region structure.

    If the negative doesn't contain the matching keywords, returns the full
    negative for every region.
    """
    # Extract ADDCOMM from negative
    neg_common, negative = _extract_keyword(negative, 'ADDCOMM')

    neg_has_col = bool(_ADDCOL_KW_RE.search(negative))
    neg_has_row = bool(_ADDROW_KW_RE.search(negative))

    neg_segments: list[str] = []

    if pos_has_row and pos_has_col and grid_shape is not None:
        # 2D grid: split negative on ADDROW, then ADDCOL within each row
        if neg_has_row:
            row_segs, _ = _split_on_keyword(negative, _ADDROW_KW_RE)
            for row_idx in range(len(grid_shape)):
                expected_cols = grid_shape[row_idx]
                row_text = row_segs[row_idx] if row_idx < len(row_segs) else (
                    row_segs[-1] if row_segs else negative)
                if neg_has_col and bool(_ADDCOL_KW_RE.search(row_text)):
                    col_segs, _ = _split_on_keyword(row_text, _ADDCOL_KW_RE)
                else:
                    col_segs = [row_text]
                # Pad to match positive's column count for this row
                while len(col_segs) < expected_cols:
                    col_segs.append(col_segs[-1] if col_segs else negative)
                neg_segments.extend(col_segs[:expected_cols])
        elif neg_has_col:
            # Neg has ADDCOL but not ADDROW: tile column segments across each row
            col_segs, _ = _split_on_keyword(negative, _ADDCOL_KW_RE)
            for row_idx in range(len(grid_shape)):
                expected_cols = grid_shape[row_idx]
                row_cols = list(col_segs)
                while len(row_cols) < expected_cols:
                    row_cols.append(row_cols[-1] if row_cols else negative)
                neg_segments.extend(row_cols[:expected_cols])
        else:
            neg_segments = [negative] * n_regions
    elif pos_has_col:
        if neg_has_col:
            neg_segments, _ = _split_on_keyword(negative, _ADDCOL_KW_RE)
        else:
            neg_segments = [negative] * n_regions
    elif pos_has_row:
        if neg_has_row:
            neg_segments, _ = _split_on_keyword(negative, _ADDROW_KW_RE)
        else:
            neg_segments = [negative] * n_regions
    else:
        neg_segments = [negative]

    # Pad/truncate to match number of positive regions
    while len(neg_segments) < n_regions:
        neg_segments.append(neg_segments[-1] if neg_segments else negative)
    neg_segments = neg_segments[:n_regions]

    # Append common negative to each region
    if neg_common:
        neg_segments = [
            f"{seg}, {neg_common}" if seg.strip() else neg_common
            for seg in neg_segments
        ]

    return neg_segments


def parse_regional_prompt(prompt: str, negative_prompt: str) -> RegionalPromptResult:
    """Parse A1111-style regional prompt into regions with spatial info.

    Supports:
    - ADDCOL only → N columns (with optional :ratio, e.g. ADDCOL:2)
    - ADDROW only → N rows (with optional :ratio)
    - Mixed ADDROW + ADDCOL → 2D grid with per-row column counts and ratios
    - ADDBASE → text before ADDBASE becomes a base prompt covering the full image
    - ADDCOMM → text before ADDCOMM is appended to every region's prompt
    - Per-region negatives: if the negative prompt contains matching ADDCOL/ADDROW
      keywords, it is split the same way and paired region-by-region

    Ratio syntax: ADDCOL:2 means the region after this keyword is 2x the default
    width. E.g. "a cat ADDCOL:2 a dog ADDCOL a fish" → widths 1/4, 2/4, 1/4.
    """
    base_prompt: str | None = None
    base_ratio = 0.2

    # Extract ADDBASE from positive
    base_prompt, prompt = _extract_keyword(prompt, 'ADDBASE')

    # Extract ADDCOMM from positive
    common_prompt, prompt = _extract_keyword(prompt, 'ADDCOMM')

    # Determine structure
    has_col = bool(_ADDCOL_KW_RE.search(prompt))
    has_row = bool(_ADDROW_KW_RE.search(prompt))

    regions: list[RegionInfo] = []
    grid_shape: list[int] | None = None

    if has_row and has_col:
        # 2D grid: split on ADDROW first, then ADDCOL within each row
        row_segs, row_ratios = _split_on_keyword(prompt, _ADDROW_KW_RE)
        row_spans = _ratios_to_spans(row_ratios)
        grid_shape = []

        for row_text, (y0, y1) in zip(row_segs, row_spans):
            col_segs, col_ratios = _split_on_keyword(row_text, _ADDCOL_KW_RE)
            col_spans = _ratios_to_spans(col_ratios)
            grid_shape.append(len(col_segs))

            for col_text, (x0, x1) in zip(col_segs, col_spans):
                regions.append(RegionInfo(
                    prompt=col_text, negative="",
                    x0=x0, y0=y0, x1=x1, y1=y1,
                ))

    elif has_col:
        segs, ratios = _split_on_keyword(prompt, _ADDCOL_KW_RE)
        spans = _ratios_to_spans(ratios)
        for text, (x0, x1) in zip(segs, spans):
            regions.append(RegionInfo(
                prompt=text, negative="",
                x0=x0, y0=0.0, x1=x1, y1=1.0,
            ))

    elif has_row:
        segs, ratios = _split_on_keyword(prompt, _ADDROW_KW_RE)
        spans = _ratios_to_spans(ratios)
        for text, (y0, y1) in zip(segs, spans):
            regions.append(RegionInfo(
                prompt=text, negative="",
                x0=0.0, y0=y0, x1=1.0, y1=y1,
            ))

    else:
        regions.append(RegionInfo(
            prompt=prompt, negative="",
            x0=0.0, y0=0.0, x1=1.0, y1=1.0,
        ))

    # Append common prompt to each region's positive
    if common_prompt:
        for region in regions:
            region.prompt = (
                f"{region.prompt}, {common_prompt}" if region.prompt.strip()
                else common_prompt
            )

    # Split negative to match positive structure
    neg_segments = _split_negative_like_positive(
        negative_prompt, has_row, has_col, len(regions), grid_shape,
    )
    for region, neg in zip(regions, neg_segments):
        region.negative = neg

    # Check if negatives actually differ across regions
    neg_texts = set(r.negative for r in regions)
    has_per_region_neg = len(neg_texts) > 1

    return RegionalPromptResult(
        regions=regions,
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        base_ratio=base_ratio,
        has_per_region_neg=has_per_region_neg,
    )


def build_region_masks(
    regions: list[RegionInfo],
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build binary region masks from normalized coordinates.

    Returns [N, 1, latent_h, latent_w] masks that sum to 1.0 at every spatial position.
    """
    n = len(regions)
    masks = torch.zeros(n, 1, latent_h, latent_w, device=device, dtype=dtype)

    for i, region in enumerate(regions):
        y0 = int(round(region.y0 * latent_h))
        y1 = int(round(region.y1 * latent_h))
        x0 = int(round(region.x0 * latent_w))
        x1 = int(round(region.x1 * latent_w))
        # Clamp
        y0, y1 = max(0, y0), min(latent_h, y1)
        x0, x1 = max(0, x0), min(latent_w, x1)
        masks[i, 0, y0:y1, x0:x1] = 1.0

    # Fix gap pixels (no region covers them due to rounding): assign uniform weight
    mask_sum = masks.sum(dim=0, keepdim=True)  # [1, 1, latent_h, latent_w]
    gaps = (mask_sum == 0)  # [1, 1, latent_h, latent_w]
    if gaps.any():
        fill = gaps.expand(n, -1, -1, -1).float() * (1.0 / n)
        masks = masks + fill.to(dtype=dtype)

    # Renormalize so masks sum to 1.0 at every position (handles rounding overlaps)
    mask_sum = masks.sum(dim=0, keepdim=True).clamp(min=1e-8)
    masks = masks / mask_sum

    return masks
