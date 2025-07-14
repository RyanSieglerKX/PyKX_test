# PyKX vs Polars vs Pandas As-Of Join Test

This notebook presents a comparison of Pandas, [PyKX (kdb+)](https://github.com/KxSystems/pykx) and [Polars](https://pola-rs.github.io/polars/) (both eager and lazy execution modes) for performing as-of joins and a basic slippage calculation on time-series market data.

> **Note:**  
> This is an **independent, community-driven test** and is **not an official KX (kdb+) or Polars benchmark**. It is intended for technical discussion, reproducibility, and as a baseline for further performance exploration.

---

## Overview

- **Goal:**  
  Compare the speed and memory usage of Pandas PyKX and Polars for typical financial analytics (as-of joins with slippage).
- **Approach:**  
  All engines operate on identical data loaded from local parquet files with consistent preprocessing (sorting, key alignment).
- **Metrics:**  
  - Wall-clock runtime (multiple iterations)
  - Memory usage (incremental, per run)
- **Environment:**  
  Tests are performed as “warm” runs within a single Python process, reflecting typical batch analytical workflows.

---

## Methodology & Fairness

- **Each engine** (PyKX, Polars Eager, Polars Lazy) runs the same join and calculation logic.
- **Preprocessing:**  
  - All tables sorted by `sym`, `time`.
  - kdb+ `g#` (grouped) attribute set for appropriate columns.
  - Polars DataFrames sorted.
- **No vendor-specific hacks** or hidden optimizations.
- **Memory profiler limitations:**  
  - Memory increments may be zero for highly efficient libraries or due to sampling granularity.
- **Limitations:**  
  - Not a “cold start” test (no process restart between runs).
  - Not measuring file scan or on-disk materialization.

---

## How to Reproduce

1. [Download and install](https://code.kx.com/pykx/3.1/getting-started/installing.html) PyKX & license 
2. Open and run the notebook in an environment with Python 3, PyKX, Polars, Numpy, Pandas, and memory_profiler installed.
3. Review and adjust any schema or path settings if your data differs.

---

## Disclaimer

**This test is not affiliated with, nor endorsed by, KX Systems, Kdb+, or the Polars core team.**  
Results should be interpreted as *one independent set of measurements*—for further research and community discussion.

---

## Feedback
 
Open an issue or submit a pull request to help improve the transparency and value of this comparison!