# Runtime Vectorization Claim-Safe Notes

This benchmark isolates trust aggregation on saved features. It is not detector runtime.

```text
 index                     method     mean      std      min
     0 cached_features_vectorized 0.000234 0.000035 0.000201
     1        current_loop_python 0.110529 0.002489 0.107335
     2           vectorized_numpy 0.000388 0.000102 0.000306
```

current_loop_python_ms: 0.110529
vectorized_numpy_ms: 0.000388
cached_vectorized_ms: 0.000234
speedup_loop_to_vectorized: 284.85x
Safe claim: the measured Python loop overhead is implementation-dependent; vectorization gives the measured reduction above.
Forbidden claim: do not claim embedded/CUDA runtime or full detector pipeline speed from this microbenchmark.
