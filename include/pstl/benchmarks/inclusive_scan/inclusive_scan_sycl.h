#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_inclusive_scan {

namespace usm_cache {
    static std::unordered_map<size_t, pstl::elem_t*> input_gpu;
    static std::unordered_map<size_t, pstl::elem_t*> output_gpu;
    static std::unordered_map<size_t, pstl::elem_t*> work_gpu;
    static std::unordered_map<size_t, size_t>        padded_size;
}

const auto inclusive_scan_sycl = [](auto&& policy, const auto& input, auto& output) {

    auto& queue            = pstl::sycl_utils::get_queue();
    using T                = pstl::elem_t;
    const size_t num_elems = input.size();
    const size_t wg        = pstl::sycl_utils::wg_size;

    // Blelloch scan requires power-of-2 size
    size_t padded = 1;
    while (padded < num_elems) padded <<= 1;

    // Allocate GPU memory once per input size
    if (usm_cache::input_gpu.find(num_elems) == usm_cache::input_gpu.end()) {
        usm_cache::input_gpu[num_elems]   = sycl::malloc_device<T>(num_elems, queue);
        usm_cache::output_gpu[num_elems]  = sycl::malloc_device<T>(num_elems, queue);
        usm_cache::work_gpu[num_elems]    = sycl::malloc_device<T>(padded, queue);
        usm_cache::padded_size[num_elems] = padded;
    }

    T* in   = usm_cache::input_gpu[num_elems];
    T* out  = usm_cache::output_gpu[num_elems];
    T* work = usm_cache::work_gpu[num_elems];

    // Copy input to GPU
    queue.memcpy(in,input.data(),num_elems * sizeof(T)).wait();
    sycl::event last_event;

    // -------------------------------------------------
    // Step 1
    // Copy input into work array
    // Pad remaining entries with 0
    // -------------------------------------------------
    {
        const size_t global =pstl::sycl_utils::round_up_global_size(padded);
        last_event = queue.submit([&](sycl::handler& h) {

            h.parallel_for(
                sycl::nd_range<1>(global, wg),
                [=](sycl::nd_item<1> item) {

                    const size_t i =
                        item.get_global_id(0);

                    if (i < padded) work[i] = (i < num_elems) ? in[i] : T(0);
                });
        });
    }

    // -------------------------------------------------
    // Step 2 : Up-Sweep
    //
    // Build reduction tree.
    //
    // stride=1 : combine pairs
    // stride=2 : combine groups of 4
    // stride=4 : combine groups of 8
    //
    // After completion:
    // work[padded-1] contains total sum.
    // -------------------------------------------------
    for (size_t stride = 1; stride < padded; stride <<= 1) {
        const size_t num_active = padded / (2 * stride);
        const size_t global = pstl::sycl_utils::round_up_global_size(num_active);

        const size_t s = stride;
        const size_t p = padded;

        last_event = queue.submit([&](sycl::handler& h) {

            h.depends_on(last_event);

            h.parallel_for(
                sycl::nd_range<1>(global, wg),
                [=](sycl::nd_item<1> item) {

                    const size_t i = item.get_global_id(0);
                    const size_t right = (i + 1) * 2 * s - 1;
                    const size_t left = right - s;
                    if (i < num_active && right < p) { work[right] += work[left];
                    }
                });
        });
    }

    // -------------------------------------------------
    // Step 3
    //
    // Convert reduction tree to scan tree
    // by clearing the root.
    // -------------------------------------------------
    last_event = queue.submit([&](sycl::handler& h) {h.depends_on(last_event);
    const size_t root = padded - 1;
        h.single_task([=]() { work[root] = T(0); });
    });
    // -------------------------------------------------
    // Step 4 : Down-Sweep
    //
    // Push prefix sums back down the tree.
    //
    // Result after this phase:
    // work[] contains EXCLUSIVE scan.
    // -------------------------------------------------
    for (size_t stride = padded >> 1;
         stride >= 1;
         stride >>= 1) {

        const size_t num_active = padded / (2 * stride);

        const size_t global = pstl::sycl_utils::round_up_global_size(num_active);

        const size_t s = stride;
        const size_t p = padded;

        last_event = queue.submit([&](sycl::handler& h) {
            h.depends_on(last_event);
            h.parallel_for(
                sycl::nd_range<1>(global, wg),
                [=](sycl::nd_item<1> item) {

                    const size_t i = item.get_global_id(0);

                    const size_t right = (i + 1) * 2 * s - 1;
                    const size_t left = right - s;
                    if (i < num_active && right < p) {
                        T temp = work[left];
                        work[left]  = work[right];
                        work[right] += temp;
                    }
                });
        });

        // prevent unsigned underflow
        if (stride == 1)
            break;
    }

    // -------------------------------------------------
    // Step 5
    //
    // Convert exclusive scan
    // into inclusive scan.
    //
    // inclusive[i] =
    // exclusive[i] + input[i]
    // -------------------------------------------------
    {
        const size_t global = pstl::sycl_utils::round_up_global_size(num_elems);

        last_event = queue.submit([&](sycl::handler& h) {
            h.depends_on(last_event);

            h.parallel_for(
                sycl::nd_range<1>(global, wg),
                [=](sycl::nd_item<1> item) {

                    const size_t i = item.get_global_id(0);

                    if (i < num_elems) out[i] = work[i] + in[i];
                });
        });
    }
    // Wait for all kernels
    last_event.wait();

    // Copy result back to CPU
    queue.memcpy(output.data(),out,num_elems * sizeof(T)).wait();
};

} // namespace benchmark_inclusive_scan
