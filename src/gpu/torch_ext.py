"""Detect foxburrow PyTorch fork features at import time.

When running on stock PyTorch, all HAS_* flags are False and all code
paths fall through to existing behavior with zero overhead.

Note: Symbol presence alone is not sufficient — the cudaMallocAsync
backend has the Python wrappers but doesn't support tag/scope/histogram
operations.  We probe at import time via hasattr, then downgrade flags
on first RuntimeError from the allocator (see check_runtime_support).
"""

import torch

HAS_ALLOC_TAGS = (hasattr(torch._C, '_cuda_setAllocationTag')
                  and hasattr(torch.cuda, 'tag_allocations'))
HAS_PEAK_SCOPE = (hasattr(torch._C, '_cuda_beginPeakMemoryScope')
                  and hasattr(torch.cuda, 'PeakMemoryScope'))
HAS_HISTOGRAM = (hasattr(torch._C, '_cuda_getAllocationHistogram')
                 and hasattr(torch.cuda, 'allocation_histogram'))

ALLOC_TAG_NONE = getattr(torch.cuda, 'ALLOC_TAG_NONE', 0)
ALLOC_TAG_MODEL_WEIGHTS = getattr(torch.cuda, 'ALLOC_TAG_MODEL_WEIGHTS', 1)
ALLOC_TAG_ACTIVATIONS = getattr(torch.cuda, 'ALLOC_TAG_ACTIVATIONS', 2)
ALLOC_TAG_CUDNN_WORKSPACE = getattr(torch.cuda, 'ALLOC_TAG_CUDNN_WORKSPACE', 3)

_runtime_checked = False


def check_runtime_support(device: torch.device) -> None:
    """Probe whether the active CUDA allocator actually supports fork features.

    Called once on first use.  Downgrades HAS_* flags to False if the
    allocator (e.g. cudaMallocAsync) raises RuntimeError.
    """
    global _runtime_checked, HAS_ALLOC_TAGS, HAS_PEAK_SCOPE, HAS_HISTOGRAM
    if _runtime_checked:
        return
    _runtime_checked = True

    if HAS_ALLOC_TAGS:
        try:
            torch.cuda.memory_allocated_by_tag(ALLOC_TAG_NONE, device)
        except RuntimeError:
            HAS_ALLOC_TAGS = False

    if HAS_PEAK_SCOPE:
        try:
            scope = torch.cuda.PeakMemoryScope(device=device)
            scope.__enter__()
            scope.__exit__(None, None, None)
        except RuntimeError:
            HAS_PEAK_SCOPE = False

    if HAS_HISTOGRAM:
        try:
            torch.cuda.allocation_histogram(device)
        except RuntimeError:
            HAS_HISTOGRAM = False
