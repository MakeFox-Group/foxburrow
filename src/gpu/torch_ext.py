"""Detect foxburrow PyTorch fork features at import time.

When running on stock PyTorch, all HAS_* flags are False and all code
paths fall through to existing behavior with zero overhead.
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
