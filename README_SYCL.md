# pSTL-Bench - SYCL GPU Backend Extension

This repository extends [pSTL-Bench](https://github.com/parlab-tuwien/pSTL-Bench) with a SYCL GPU backend targeting NVIDIA V100 with AdaptiveCpp.

## Added

### New SYCL Implementations
All in `include/pstl/benchmarks/<algo>/`:

| File | Description |
|------|-------------|
| `find/find_sycl.h` | Parallel find with atomic early-exit flag |
| `for_each/for_each_sycl.h` | nd_range parallel_for |
| `reduce/reduce_sycl.h` | Tree reduction with local memory |
| `inclusive_scan/inclusive_scan_sycl.h` | Multi-pass Hillis-Steele prefix sum |
| `sort/sort_sycl.h` | Bitonic sort with SYCL event-chaining  |
| `inclusive_scan/inclusive_scan_gnu.h` | GNU/OpenMP parallel inclusive_scan |

### Modified Files
| File | Change |
|------|--------|
| `include/pstl/benchmarks/inclusive_scan/group.h` | Added GNU backend registration |
| `cmake/backends/SYCL.cmake` | AdaptiveCpp detection, skips -fsycl for acpp |
| `plot_results_stats.py` |  benchmark name prefix as label, GNU=dotted, TBB=dashed |
| `compare_wg_sizes.py` | wg_size overlay plots and heatmaps |

### Results
`results/` JSON files:
- `tbb.json` → TBB parallel baseline (GCC 12.3, 10 reps)
- `gnu.json` → GNU/OpenMP parallel baseline (GCC 12.3, 10 reps)
- `sycl_wg{32,64,128,256,512,1024}.json` → SYCL GPU results per work-group size

### Plots
- `plots_stats/` - GPU vs CPU comparison per algorithm (5 algorithms)
- `plots_wg_compare/` - wg_size overlay

## Build (SYCL backend)

```bash
source ~/setup_env.sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPSTL_BENCH_BACKEND=SYCL \
      -DCMAKE_CXX_COMPILER=$HOME/adaptivecpp/bin/acpp \
      -DPSTL_BENCH_MAX_INPUT_SIZE=268435456 \
      -DPSTL_BENCH_SYCL_WG_SIZE=256 \
      -DCMAKE_CXX_FLAGS="-I$TBB_OLD/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$GCC9/lib64 -Wl,-rpath,$GCC9/lib64 -L$TBB_OLD/lib -Wl,-rpath,$TBB_OLD/lib -ltbb" \
      -B ~/pstl-builds/sycl-wg256 -S ~/pstl-bench
cmake --build ~/pstl-builds/sycl-wg256 --target pSTL-Bench -j16
```

## Hardware (exa03)
- 2x Intel Xeon Gold 6130 (16-core, 2.10 GHz)
- NVIDIA Tesla V100 32GB (sm_70, CUDA 12.3)
- AdaptiveCpp 24.06.0, GCC 12.3.0, TBB 2020.3
