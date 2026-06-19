#include <iostream>
#include <vector>
#include <execution>
#include <numeric>
#include "pstl/benchmarks/find/find_sycl.h"

bool run_test(std::vector<pstl::elem_t> input, pstl::elem_t target,
              bool should_find, const std::string& label) {
    size_t n = input.size();
    auto result = benchmark_find::find_sycl(std::execution::par, input, target);
    bool found = (result != input.end());
    bool pass;
    if (should_find) {
        auto expected = std::find(input.begin(), input.end(), target);
        pass = (result == expected);
    } else {
        pass = !found;
    }
    std::cout << label << " (n=" << n << ", target=" << static_cast<long long>(target) << ")\n";
    std::cout << "  Found: " << (found ? "yes" : "no") << " (expected: "
              << (should_find ? "yes" : "no") << ")\n";
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";
    return pass;
}

int main() {
    bool all_pass = true;

    std::cout << "===== POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {8, 16, 32, 64, 1024}) {
        std::vector<pstl::elem_t> v(n);
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>(i % 100);
        all_pass &= run_test(v, v[n / 2], true, "power-of-2");
    }

    std::cout << "===== NON-POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {10, 20, 25, 57, 89}) {
        std::vector<pstl::elem_t> v(n);
        for (size_t i = 0; i < n; i++) v[i] = static_cast<pstl::elem_t>(i % 100);
        all_pass &= run_test(v, v[n / 2], true, "non-power-of-2");
    }

    std::cout << "===== CORNER CASES =====\n\n";
    all_pass &= run_test({}, static_cast<pstl::elem_t>(5), false, "EMPTY array");
    all_pass &= run_test({static_cast<pstl::elem_t>(42)}, static_cast<pstl::elem_t>(42), true, "single element, found");
    all_pass &= run_test({static_cast<pstl::elem_t>(42)}, static_cast<pstl::elem_t>(99 % 100), false, "single element, not found");

    {
        std::vector<pstl::elem_t> v = {static_cast<pstl::elem_t>(1), static_cast<pstl::elem_t>(2), static_cast<pstl::elem_t>(3)};
        all_pass &= run_test(v, v[0], true, "target at first position");
        all_pass &= run_test(v, v[2], true, "target at last position");
        all_pass &= run_test(v, static_cast<pstl::elem_t>(99), false, "target not present");
    }

    {
        std::vector<pstl::elem_t> v(20, static_cast<pstl::elem_t>(7));
        all_pass &= run_test(v, static_cast<pstl::elem_t>(7), true, "all identical values, target present");
        all_pass &= run_test(v, static_cast<pstl::elem_t>(8), false, "all identical values, target absent");
    }

    {
        std::vector<pstl::elem_t> v(1023);
        for (size_t i = 0; i < v.size(); i++) v[i] = static_cast<pstl::elem_t>(i % 100);
        all_pass &= run_test(v, v[500], true, "power-of-2 minus 1 (n=1023)");
    }
    {
        std::vector<pstl::elem_t> v(1025);
        for (size_t i = 0; i < v.size(); i++) v[i] = static_cast<pstl::elem_t>(i % 100);
        all_pass &= run_test(v, v[500], true, "power-of-2 plus 1 (n=1025)");
    }

    std::cout << "==========================================\n";
    std::cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
