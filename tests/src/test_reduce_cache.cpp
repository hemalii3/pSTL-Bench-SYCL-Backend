#include <iostream>
#include <vector>
#include <execution>
#include <numeric>
#include "pstl/benchmarks/reduce/reduce_sycl.h"

bool test_repeated_same_size() {
    std::cout << "=== Repeated calls, same size, different data ===\n\n";
    bool all_pass = true;

    std::vector<pstl::elem_t> a = {5, 3, 8, 1, 9, 2, 7, 4};
    pstl::elem_t a_expected = std::accumulate(a.begin(), a.end(), pstl::elem_t(0));
    auto r1 = benchmark_reduce::reduce_sycl(std::execution::par, a);
    bool pass1 = (r1 == a_expected);
    std::cout << "Call 1 (n=8): expected=" << a_expected << " got=" << r1 << " "
              << (pass1 ? "PASS" : "FAIL") << "\n";
    all_pass &= pass1;

    std::vector<pstl::elem_t> b = {100, 99, 98, 97, 96, 95, 94, 93};
    pstl::elem_t b_expected = std::accumulate(b.begin(), b.end(), pstl::elem_t(0));
    auto r2 = benchmark_reduce::reduce_sycl(std::execution::par, b);
    bool pass2 = (r2 == b_expected);
    std::cout << "Call 2 (n=8, different data): expected=" << b_expected << " got=" << r2 << " "
              << (pass2 ? "PASS" : "FAIL") << "\n";
    all_pass &= pass2;

    std::vector<pstl::elem_t> c = {-5, -3, -8, -1, -9, -2, -7, -4};
    pstl::elem_t c_expected = std::accumulate(c.begin(), c.end(), pstl::elem_t(0));
    auto r3 = benchmark_reduce::reduce_sycl(std::execution::par, c);
    bool pass3 = (r3 == c_expected);
    std::cout << "Call 3 (n=8, negative data): expected=" << c_expected << " got=" << r3 << " "
              << (pass3 ? "PASS" : "FAIL") << "\n\n";
    all_pass &= pass3;

    return all_pass;
}

bool test_interleaved_sizes() {
    std::cout << "=== Interleaved calls across different sizes ===\n\n";
    bool all_pass = true;

    std::vector<size_t> size_sequence = {100, 50, 100, 200, 50, 100};

    for (size_t idx = 0; idx < size_sequence.size(); idx++) {
        size_t n = size_sequence[idx];
        std::vector<pstl::elem_t> v(n);
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>((i * 3 + idx * 7) % 50);

        pstl::elem_t expected = std::accumulate(v.begin(), v.end(), pstl::elem_t(0));
        auto result = benchmark_reduce::reduce_sycl(std::execution::par, v);

        bool pass = (result == expected);
        std::cout << "Call " << idx << " (n=" << n << "): expected=" << expected
                  << " got=" << result << " " << (pass ? "PASS" : "FAIL") << "\n";
        all_pass &= pass;
    }
    std::cout << "\n";

    return all_pass;
}

int main() {
    bool all_pass = true;
    all_pass &= test_repeated_same_size();
    all_pass &= test_interleaved_sizes();

    std::cout << "==========================================\n";
    std::cout << (all_pass ? "ALL CACHE TESTS PASSED" : "SOME CACHE TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
