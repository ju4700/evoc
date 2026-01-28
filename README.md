# EvoCode Accelerator (ECA) — Minimal Prototype

This prototype demonstrates a minimal "self-evolving on command" flow:

- Accept a target file and function name.
- Generate simple optimized variants (add `lru_cache`, optional `numba.njit`, simple loop→list-comprehension transform).
- Run each variant in a subprocess and report timings.

Usage (from workspace root):

```bash
python -m eca --file examples/example.py --fn sum_squares --arg 100000 --time 10
```

This is a small demo to show the UX and evolution loop; it is not production-ready. Next steps: correctness testing, sandboxing, richer AST transformations, DEAP-based search, NLP spec parsing, and JIT/backends.
