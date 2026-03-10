# Autograd Tests

## Kernel Test Guidance

When adding tests for a repo-owned custom kernel, split the coverage into two layers:

1. Contract/oracle parity tests
   - Compare the repo op against the clearest semantic oracle available.
   - Prefer the repo path first (e.g. existing autograd implementation), then framework/vendor references (e.g. PyTorch).
   - Use these to answer: "does the end-to-end operator match the intended contract numerically on supported inputs?"

2. Kernel-focused invariant tests
   - Test behavior that should hold for every valid input in the supported
     contract, not just for one lucky tensor sample.
   - Prefer first-principles properties over hard-coded expected tensors.
   - These tests are the best way to catch indexing, masking, and reduction
     bugs inside the kernel implementation itself.

Avoid writing only "golden input -> golden output" tests for kernels unless the case demonstrates a real invariant. Those tests are brittle and often miss the actual failure modes in low-level kernels.
