#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include "pstl/benchmarks/for_each/for_each_sycl.h"

auto kernel = [](auto& elem, size_t its) {
    volatile size_t I = its;
    pstl::elem_t a{};
    for (size_t i = 0; i < I; ++i) a++;
    elem = a;
};

bool check(std::vector<pstl::elem_t> v, size_t its, const std::string& label) {
    size_t n = v.size();
    benchmark_for_each::for_each_sycl(std::execution::par, v,
        [=](auto& elem) { kernel(elem, its); });

    bool pass = std::all_of(v.begin(), v.end(),
        [its](auto elem) { return elem == static_cast<pstl::elem_t>(its); });

    std::cout << label << " (n=" << n << ", its=" << its << "): "
              << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

int main() {
    bool all_pass = true;

    std::cout << "=== Repeated calls, same size, different its ===\n\n";
    all_pass &= check(std::vector<pstl::elem_t>(8, 0), 10, "Call 1 (n=8, its=10)");
    all_pass &= check(std::vector<pstl::elem_t>(8, 0), 50, "Call 2 (n=8, its=50, same size)");
    all_pass &= check(std::vector<pstl::elem_t>(8, 0), 1, "Call 3 (n=8, its=1, same size again)");

    std::cout << "\n=== Interleaved calls across different sizes ===\n\n";
    std::vector<size_t> size_sequence = {100, 50, 100, 200, 50, 100};
    for (size_t idx = 0; idx < size_sequence.size(); idx++) {
        size_t n = size_sequence[idx];
        size_t its = (idx % 3 == 0) ? 5 : (idx % 3 == 1) ? 20 : 100;
        all_pass &= check(std::vector<pstl::elem_t>(n, 0), its,
            "Call " + std::to_string(idx) + " (n=" + std::to_string(n) + ")");
    }

    std::cout << "\n=== Same size and its, but original data differs ===\n\n";
    // confirms 'its' overrides whatever original data was, regardless of cache
    all_pass &= check({1, 2, 3, 4, 5, 6, 7, 8}, 42, "original data [1..8]");
    all_pass &= check({99, 98, 97, 96, 95, 94, 93, 92}, 42, "original data [99..92], same n and its");

    std::cout << "\n==========================================\n";
    std::cout << (all_pass ? "ALL CACHE TESTS PASSED" : "SOME CACHE TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
