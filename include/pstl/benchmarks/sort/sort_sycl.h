#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_sort {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
    }
    const auto sort_sycl = [](auto && policy, auto & input) {
        auto & q = pstl::sycl_utils::get_queue();


        const size_t n = input.size();
        const size_t wg = pstl::sycl_utils::wg_size;
        const size_t global = pstl::sycl_utils::round_up_global_size(n);

        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n] = sycl::malloc_device<pstl::elem_t>(n, q);
        }

        pstl::elem_t* d = usm_cache::d_in[n];
        q.memcpy(d, input.data(), n * sizeof(pstl::elem_t)).wait();


        sycl::event dep;
        bool first = true;

        for (size_t k = 2; k <= n; k <<= 1) {
            for (size_t j = k >> 1; j >= 1; j >>= 1) {
                size_t kk = k, jj = j;

                auto e = q.submit([&](sycl::handler & h) {
                    if (!first) h.depends_on(dep);

                   h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                   size_t i = it.get_global_id(0), l = i ^ jj;
                   if (i >= n) return;

                        if (l > i) {
                            bool asc = ((i & kk) == 0);
                            
                            if ((d[i] > d[l]) == asc) {
                                pstl::elem_t t = d[i]; d[i] = d[l]; d[l] = t;
                            }
                        }
                    });
                });
                dep = e;
                first = false;
            }
        }
        if (!first) dep.wait();
        q.memcpy(input.data(), d, n * sizeof(pstl::elem_t)).wait();
    };
} // namespace benchmark_sort

