# pSTL-Bench - SYCL GPU Backend Extension

This repository extends [pSTL-Bench](https://github.com/parlab-tuwien/pSTL-Bench) with a SYCL GPU backend targeting NVIDIA V100 with AdaptiveCpp.


### New SYCL Implementations
All in `include/pstl/benchmarks/<algo>/`:

| File | Description |
|------|-------------|
| `find/find_sycl.h` | Parallel find with atomic flag |
| `for_each/for_each_sycl.h` | nd_range parallel_for |
| `reduce/reduce_sycl.h` | Tree reduction with local memory |
| `inclusive_scan/inclusive_scan_sycl.h` | blelloch scan |
| `sort/sort_sycl.h` | Bitonic sort with SYCL event-chaining  |


## Hardware (exa03)
- 2x Intel Xeon Gold 6130 (16-core, 2.10 GHz)
- NVIDIA Tesla V100 32GB (sm_70, CUDA 12.3)
- AdaptiveCpp 24.06.0, GCC 12.3.0, TBB 2020.3
