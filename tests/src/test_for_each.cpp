#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include "pstl/benchmarks/for_each/for_each_sycl.h"

bool run_test(std::vector<pstl::elem_t> input, size_t its, const std::string& label) {

    auto kernel = [](auto& elem, size_t its) {
        volatile size_t I = its;
        pstl::elem_t a{};
        for (size_t i = 0; i < I; ++i) a++;
        elem = a;
    };

    size_t n = input.size();

    benchmark_for_each::for_each_sycl(std::execution::par, input,
        [=](auto& elem) { kernel(elem, its); });

    bool size_ok = (input.size() == n);
    bool correct_ok = std::all_of(input.begin(), input.end(),
        [its](auto elem) { return elem == static_cast<pstl::elem_t>(its); });
    bool pass = size_ok && correct_ok;

    std::cout << label << " (n=" << n << ", its=" << its << ")\n";
    std::cout << "  Size preserved: " << (size_ok ? "yes" : "NO") << "\n";
    std::cout << "  All elements correct: " << (correct_ok ? "yes" : "NO") << "\n";
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";

    return pass;
}

std::vector<pstl::elem_t> dummy_vector(size_t n) {
    return std::vector<pstl::elem_t>(n, 0);
}

int main() {
    bool all_pass = true;

    std::cout << "===== POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {8, 16, 32, 64, 1024}) all_pass &= run_test(dummy_vector(n), 100, "power-of-2");

    std::cout << "===== NON-POWER-OF-2 SIZES =====\n\n";
    for (size_t n : {10, 20, 25, 57, 89}) all_pass &= run_test(dummy_vector(n), 100, "non-power-of-2");

    std::cout << "===== CORNER CASES =====\n\n";
    all_pass &= run_test({}, 100, "EMPTY array");
    all_pass &= run_test(dummy_vector(1), 100, "single element");
    all_pass &= run_test(dummy_vector(2), 100, "two elements");
    all_pass &= run_test(dummy_vector(50), 1, "its=1 (minimal work)");
    all_pass &= run_test(dummy_vector(50), 0, "its=0 (zero work)");
    all_pass &= run_test(dummy_vector(1023), 100, "power-of-2 minus 1 (n=1023)");
    all_pass &= run_test(dummy_vector(1025), 100, "power-of-2 plus 1 (n=1025)");

    std::cout << "==========================================\n";
    std::cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_pass ? 0 : 1;
}
