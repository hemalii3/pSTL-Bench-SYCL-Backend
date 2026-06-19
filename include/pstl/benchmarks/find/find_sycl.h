#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_find {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
        static std::unordered_map<size_t, size_t*>       d_res;
    }

    const auto find_sycl = [](auto&& policy, const auto& input, const pstl::elem_t& target) {

        auto& queue            = pstl::sycl_utils::get_queue();
        const size_t num_elems = input.size();

        if (num_elems == 0) {
            return input.end();
        }

        const size_t wg        = pstl::sycl_utils::wg_size;
        const size_t global    = pstl::sycl_utils::round_up_global_size(num_elems);

        if (usm_cache::d_in.find(num_elems) == usm_cache::d_in.end()) {
            usm_cache::d_in[num_elems]  = sycl::malloc_device<pstl::elem_t>(num_elems, queue);
            usm_cache::d_res[num_elems] = sycl::malloc_device<size_t>(1, queue);
        }

        pstl::elem_t* in     = usm_cache::d_in[num_elems];
        size_t*       result = usm_cache::d_res[num_elems];

        queue.memcpy(in, input.data(), num_elems * sizeof(pstl::elem_t)).wait();

        const size_t not_found = num_elems;
        queue.memcpy(result, &not_found, sizeof(size_t)).wait();

        queue.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(global, wg),
                sycl::reduction(result, not_found, sycl::minimum<size_t>()),
                [=](sycl::nd_item<1> it, auto& min_idx) {
                    const size_t i = it.get_global_id(0);
                    size_t candidate = (i < num_elems && in[i] == target) ? i : not_found;
                    min_idx.combine(candidate);
                }
            );
        }).wait();

        size_t host_result = num_elems;
        queue.memcpy(&host_result, result, sizeof(size_t)).wait();

        return host_result == num_elems
            ? input.end()
            : input.begin() + host_result;
    };
} // namespace benchmark_find
