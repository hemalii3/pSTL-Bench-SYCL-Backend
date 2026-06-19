#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include "pstl/benchmarks/sort/sort_sycl.h"

bool test_repeated_same_size() {
    std::cout << "=== Repeated calls, same size, different data ===\n\n";
    bool all_pass = true;

    // first call: one dataset
    std::vector<pstl::elem_t> a = {5, 3, 8, 1, 9, 2, 7, 4};
    std::vector<pstl::elem_t> a_expected = a;
    std::sort(a_expected.begin(), a_expected.end());
    benchmark_sort::sort_sycl(std::execution::par, a);
    bool pass1 = (a == a_expected);
    std::cout << "Call 1 (n=8): " << (pass1 ? "PASS" : "FAIL") << "\n";
    all_pass &= pass1;

    // second call: SAME size, COMPLETELY different data
    std::vector<pstl::elem_t> b = {100, 99, 98, 97, 96, 95, 94, 93};
    std::vector<pstl::elem_t> b_expected = b;
    std::sort(b_expected.begin(), b_expected.end());
    benchmark_sort::sort_sycl(std::execution::par, b);
    bool pass2 = (b == b_expected);
    std::cout << "Call 2 (n=8, different data): " << (pass2 ? "PASS" : "FAIL") << "\n";
    all_pass &= pass2;

    // third call: back to size 8 again, yet another dataset
    std::vector<pstl::elem_t> c = {-5, -3, -8, -1, -9, -2, -7, -4};
    std::vector<pstl::elem_t> c_expected = c;
    std::sort(c_expected.begin(), c_expected.end());
    benchmark_sort::sort_sycl(std::execution::par, c);
    bool pass3 = (c == c_expected);
    std::cout << "Call 3 (n=8, negative data): " << (pass3 ? "PASS" : "FAIL") << "\n\n";
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
        // make data depend on both n and idx, so repeated sizes still get different content
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>((i * 7 + idx * 13) % 1000);

        std::vector<pstl::elem_t> expected = v;
        std::sort(expected.begin(), expected.end());

        benchmark_sort::sort_sycl(std::execution::par, v);

        bool pass = (v == expected);
        std::cout << "Call " << idx << " (n=" << n << "): " << (pass ? "PASS" : "FAIL") << "\n";
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
