Greedy decoding on CPU (2 threads), trained checkpoint, 16-token prompt, best of 3 runs.

| New tokens | With KV cache (tok/s) | Without (tok/s) | Speed-up | Same tokens |
|---|---|---|---|---|
| 32 | 549.9 | 214.5 | 2.56x | yes |
| 64 | 560.0 | 178.5 | 3.14x | yes |
| 128 | 547.7 | 131.2 | 4.18x | yes |
| 224 | 533.9 | 92.4 | 5.78x | yes |
