#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_reduce {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
        static std::unordered_map<size_t, pstl::elem_t*> d_out;
    }

    const auto reduce_sycl = [](auto&& policy, const auto& input) {
        auto& q = pstl::sycl_utils::get_queue();
        using T = pstl::elem_t;

        const size_t n      = input.size();
        const size_t wg     = pstl::sycl_utils::wg_size;
        const size_t global = pstl::sycl_utils::round_up_global_size(n);

        // allocate GPU buffers once per input size
        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n]  = sycl::malloc_device<T>(n, q);
            usm_cache::d_out[n] = sycl::malloc_device<T>(1, q);
        }

        T* in  = usm_cache::d_in[n];
        T* out = usm_cache::d_out[n];

        // copy input to GPU
        q.memcpy(in, input.data(), n * sizeof(T)).wait();
	T zero = 0;
	q.memcpy(out, &zero, sizeof(T)).wait();

        // reduce on GPU
	q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>(global, wg),
        sycl::reduction(out, sycl::plus<T>()),
        [=](sycl::nd_item<1> it, auto& sum) {
            const size_t i = it.get_global_id(0);
            if (i < n) sum += in[i];
        }
    );
}).wait();	
        // copy result back to CPU
        T result = 0;
        q.memcpy(&result, out, sizeof(T)).wait();
        return result;
    };
} // namespace benchmark_reduce
