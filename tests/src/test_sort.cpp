#include <iostream>
#include <vector>
#include <execution>
#include <algorithm>
#include <numeric>
#include <random>
#include "pstl/benchmarks/sort/sort_sycl.h"

void print_array(const std::vector<pstl::elem_t>& v) {
    size_t limit = std::min<size_t>(v.size(), 15);
    std::cout << "[";
    for (size_t i = 0; i < limit; i++) std::cout << static_cast<long long>(v[i]) << " ";
    if (v.size() > 15) std::cout << "... (" << v.size() - 15 << " more)";
    std::cout << "]\n";
}

bool run_test(std::vector<pstl::elem_t> input, const std::string& label) {

    std::vector<pstl::elem_t> original = input;
    std::vector<pstl::elem_t> expected = input;
    std::sort(expected.begin(), expected.end());

    size_t original_size = input.size();

    benchmark_sort::sort_sycl(std::execution::par, input);

    bool size_ok = (input.size() == original_size);
    bool sorted_ok = (input == expected);
    bool pass = size_ok && sorted_ok;

    std::cout << label << " (n=" << original_size << ")\n";

    std::cout << "  Input:  ";
    print_array(original);

    std::cout << "  Sorted: ";
    print_array(input);

    std::cout << "  Size preserved: " << (size_ok ? "yes" : "NO") << "\n";
    std::cout << "  Correctly sorted: " << (sorted_ok ? "yes" : "NO") << "\n";
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
    std::vector<size_t> pow2_sizes = {8, 16, 32, 64, 1024};
    for (size_t n : pow2_sizes) {
        all_pass &= run_test(random_vector(n, 42 + n), "power-of-2");
    }

    std::cout << "===== NON-POWER-OF-2 SIZES =====\n\n";
    std::vector<size_t> non_pow2_sizes = {10, 20, 25, 57, 89};
    for (size_t n : non_pow2_sizes) {
        all_pass &= run_test(random_vector(n, 100 + n), "non-power-of-2");
    }

    std::cout << "===== CORNER CASES =====\n\n";

    all_pass &= run_test({}, "EMPTY array");
    all_pass &= run_test({42}, "single element");
    all_pass &= run_test({2, 1}, "two elements (reversed)");
    all_pass &= run_test(std::vector<pstl::elem_t>(20, 7), "all identical values (n=20)");

    {
        std::vector<pstl::elem_t> v(50);
        std::iota(v.begin(), v.end(), 0);
        all_pass &= run_test(v, "already ascending sorted (n=50)");
    }
    {
        std::vector<pstl::elem_t> v(50);
        std::iota(v.begin(), v.end(), 0);
        std::reverse(v.begin(), v.end());
        all_pass &= run_test(v, "reverse sorted descending (n=50)");
    }

    all_pass &= run_test({-5, 3, -100, 0, 42, -1}, "negative values mixed");
    all_pass &= run_test(random_vector(1023, 999), "power-of-2 minus 1 (n=1023)");
    all_pass &= run_test(random_vector(1025, 999), "power-of-2 plus 1 (n=1025)");

    std::cout << "==========================================\n";
    std::cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";

    return all_pass ? 0 : 1;
}
