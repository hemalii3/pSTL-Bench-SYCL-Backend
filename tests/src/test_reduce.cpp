#include <iostream>
#include <vector>
#include <execution>
#include <numeric>
#include <algorithm>
#include <random>
#include "pstl/benchmarks/reduce/reduce_sycl.h"

bool run_test(std::vector<pstl::elem_t> input, const std::string& label) {
    pstl::elem_t expected = std::accumulate(input.begin(), input.end(), pstl::elem_t(0));

    size_t n = input.size();
    auto result = benchmark_reduce::reduce_sycl(std::execution::par, input);

    bool pass = (result == expected);

    std::cout << label << " (n=" << n << ")\n";
    std::cout << "  Expected: " << expected << "\n";
    std::cout << "  Got:      " << result << "\n";
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";

    return pass;
}

std::vector<pstl::elem_t> random_vector(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 100);
    std::vector<pstl::elem_t> v(n);
    for (auto& x : v) x = static_cast<pstl::elem_t>(dist(gen));
    return v;
}

int main() {
    bool all_pass = true;

    std::cout << "===== POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {8, 16, 32, 64, 1024}) all_pass &= run_test(random_vector(n, 42 + n), "power-of-2");

    std::cout << "===== NON-POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {10, 20, 25, 57, 89}) all_pass &= run_test(random_vector(n, 100 + n), "non-power-of-2");

    std::cout << "===== CORNER CASES =====\n\n";
    all_pass &= run_test({}, "EMPTY array");
    all_pass &= run_test({42}, "single element");
    all_pass &= run_test({2, 1}, "two elements");
    all_pass &= run_test(std::vector<pstl::elem_t>(20, 7), "all identical values (n=20)");

    {
        std::vector<pstl::elem_t> v(50); std::iota(v.begin(), v.end(), 0);
        all_pass &= run_test(v, "ascending sorted (n=50)");
    }
    {
        std::vector<pstl::elem_t> v(50); std::iota(v.begin(), v.end(), 0);
        std::reverse(v.begin(), v.end());
        all_pass &= run_test(v, "reverse sorted (n=50)");
    }

    all_pass &= run_test({-5, 3, -100, 0, 42, -1}, "negative values mixed");
    all_pass &= run_test(std::vector<pstl::elem_t>(30, 0), "all zeros (n=30)");
    all_pass &= run_test(random_vector(1023, 999), "power-of-2 minus 1 (n=1023)");
    all_pass &= run_test(random_vector(1025, 999), "power-of-2 plus 1 (n=1025)");

    std::cout << "==========================================\n";
    std::cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
