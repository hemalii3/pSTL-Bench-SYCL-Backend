#include <iostream>
#include <vector>
#include <execution>
#include "pstl/benchmarks/find/find_sycl.h"

bool check(std::vector<pstl::elem_t>& v, pstl::elem_t target, const std::string& label) {
    auto expected = std::find(v.begin(), v.end(), target);
    auto result = benchmark_find::find_sycl(std::execution::par, v, target);

    bool pass = (result == expected);
    std::cout << label << " (n=" << v.size() << ", target=" << target << "): "
              << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

int main() {
    bool all_pass = true;

    std::cout << "=== Repeated calls, same size, different data ===\n\n";
    std::vector<pstl::elem_t> a = {5, 3, 8, 1, 9, 2, 7, 4};
    all_pass &= check(a, 8, "Call 1 (n=8, target present)");

    std::vector<pstl::elem_t> b = {100, 99, 98, 97, 96, 95, 94, 93};
    all_pass &= check(b, 200, "Call 2 (n=8, different data, target absent)");

    std::vector<pstl::elem_t> c = {-5, -3, -8, -1, -9, -2, -7, -4};
    all_pass &= check(c, -8, "Call 3 (n=8, negative data, target present)");

    std::cout << "\n=== Interleaved calls across different sizes ===\n\n";
    std::vector<size_t> size_sequence = {100, 50, 100, 200, 50, 100};
    for (size_t idx = 0; idx < size_sequence.size(); idx++) {
        size_t n = size_sequence[idx];
        std::vector<pstl::elem_t> v(n);
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>((i * 3 + idx * 7) % 50);
        all_pass &= check(v, v[n / 2], "Call " + std::to_string(idx) + " (n=" + std::to_string(n) + ")");
    }

    std::cout << "\n==========================================\n";
    std::cout << (all_pass ? "ALL CACHE TESTS PASSED" : "SOME CACHE TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
