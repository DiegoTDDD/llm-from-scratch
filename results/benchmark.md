Greedy decoding on CPU (2 threads), trained checkpoint, 16-token prompt, best of 3 runs.

| New tokens | With KV cache (tok/s) | Without (tok/s) | Speed-up | Same tokens |
|---|---|---|---|---|
| 32 | 542.4 | 218.2 | 2.49x | yes |
| 64 | 552.6 | 179.5 | 3.08x | yes |
| 128 | 545.0 | 132.4 | 4.12x | yes |
| 224 | 537.8 | 93.1 | 5.78x | yes |
