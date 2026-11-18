# ADR 0001: Write attention by hand instead of calling `scaled_dot_product_attention`

**Status:** accepted

## Context
PyTorch ships `F.scaled_dot_product_attention`, which is faster and fuses the mask and
softmax. The point of this project is to show the mechanism, and to test it.

## Decision
`CausalSelfAttention` computes `q @ kᵀ / √d`, applies an explicit upper-triangular mask from a
registered buffer, takes the softmax and multiplies by `v`. It keeps the last attention matrix
so tests can inspect it.

## Consequences
- Tests can check the mask directly: every weight above the diagonal is exactly 0 and every
  row sums to 1 (`test_attention_weights_are_zero_above_the_diagonal`), and changing future
  tokens never changes earlier logits.
- It is slower than the fused kernel and uses O(T²) memory per head. At context 256 on a CPU
  this costs little; a production model would switch to SDPA and keep these tests as the spec.
