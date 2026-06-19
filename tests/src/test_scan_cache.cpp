#include <iostream>
#include <vector>
#include <execution>
#include <numeric>
#include "pstl/benchmarks/inclusive_scan/inclusive_scan_sycl.h"

bool check(std::vector<pstl::elem_t>& v, const std::string& label) {
    std::vector<pstl::elem_t> expected(v.size());
    std::inclusive_scan(v.begin(), v.end(), expected.begin());

    std::vector<pstl::elem_t> output(v.size());
    benchmark_inclusive_scan::inclusive_scan_sycl(std::execution::par, v, output);

    bool pass = (output == expected);
    std::cout << label << " (n=" << v.size() << "): " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

int main() {
    bool all_pass = true;

    std::cout << "=== Repeated calls, same size, different data ===\n\n";
    std::vector<pstl::elem_t> a = {5, 3, 8, 1, 9, 2, 7, 4};
    all_pass &= check(a, "Call 1 (n=8)");

    std::vector<pstl::elem_t> b = {100, 99, 98, 97, 96, 95, 94, 93};
    all_pass &= check(b, "Call 2 (n=8, different data)");

    std::vector<pstl::elem_t> c = {-5, -3, -8, -1, -9, -2, -7, -4};
    all_pass &= check(c, "Call 3 (n=8, negative data)");

    std::cout << "\n=== Interleaved calls across different sizes ===\n\n";
    std::vector<size_t> size_sequence = {100, 50, 100, 200, 50, 100};
    for (size_t idx = 0; idx < size_sequence.size(); idx++) {
        size_t n = size_sequence[idx];
        std::vector<pstl::elem_t> v(n);
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>((i * 3 + idx * 7) % 50);
        all_pass &= check(v, "Call " + std::to_string(idx) + " (n=" + std::to_string(n) + ")");
    }

    std::cout << "\n==========================================\n";
    std::cout << (all_pass ? "ALL CACHE TESTS PASSED" : "SOME CACHE TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
