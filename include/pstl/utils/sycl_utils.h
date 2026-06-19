#pragma once
#include <sycl/sycl.hpp>

#ifndef PSTL_BENCH_SYCL_WG_SIZE
#define PSTL_BENCH_SYCL_WG_SIZE 256
#endif

namespace pstl::sycl_utils
{
    constexpr size_t wg_size = PSTL_BENCH_SYCL_WG_SIZE;

    inline size_t round_up_global_size(size_t n)
    {
        return ((n + wg_size - 1) / wg_size) * wg_size;
    }

    inline sycl::queue & get_queue()
    {
#if defined(PSTL_BENCH_FORCE_CPU_DEVICE)
        static sycl::queue q{ sycl::cpu_selector_v };
#else
        static sycl::queue q{ sycl::default_selector_v };
#endif
        return q;
    }
} // namespace pstl::sycl_utils
