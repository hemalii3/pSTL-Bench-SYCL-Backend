#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <limits>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_sort {

namespace usm_cache {
    static std::unordered_map<size_t, pstl::elem_t*> d_in;
    static std::unordered_map<size_t, size_t>         d_padded;
}

const auto sort_sycl = [](auto && policy, auto & input) {

    auto & q = pstl::sycl_utils::get_queue();
    const size_t n  = input.size();
    const size_t wg = pstl::sycl_utils::wg_size;

    // next power of 2 >= n (bitonic sort requires power of 2)
    size_t padded = 1;
    while (padded < n) padded <<= 1;

    const size_t global = pstl::sycl_utils::round_up_global_size(padded);

    // allocate padded-size buffer once per input size
    if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
        usm_cache::d_in[n]     = sycl::malloc_device<pstl::elem_t>(padded, q);
        usm_cache::d_padded[n] = padded;
    }

    pstl::elem_t* d = usm_cache::d_in[n];

    // copy real input
    q.memcpy(d, input.data(), n * sizeof(pstl::elem_t)).wait();

    // pad extra slots with +infinity so they sort to the end
    if (padded > n) {
        const pstl::elem_t max_val = std::numeric_limits<pstl::elem_t>::max();
        q.submit([&](sycl::handler & h) {
            const size_t pad_start = n;
            const size_t pad_count = padded - n;
            h.parallel_for(sycl::range<1>(pad_count), [=](sycl::id<1> idx) {
                d[pad_start + idx[0]] = max_val;
            });
        }).wait();
    }

    sycl::event dep;
    bool first = true;

    // bitonic sort core, operating on 'padded' elements
    for (size_t k = 2; k <= padded; k <<= 1) {
        for (size_t j = k >> 1; j >= 1; j >>= 1) {
            size_t kk = k, jj = j;
            size_t p  = padded;

            auto e = q.submit([&](sycl::handler & h) {
                if (!first) h.depends_on(dep);

                h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                    size_t i = it.get_global_id(0), l = i ^ jj;
                    if (i >= p) return;

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

    // copy back only the real n elements, discarding padding
    q.memcpy(input.data(), d, n * sizeof(pstl::elem_t)).wait();
};

} // namespace benchmark_sort
