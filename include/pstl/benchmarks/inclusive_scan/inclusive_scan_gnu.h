#pragma once

#include <execution>
#include <numeric>

namespace benchmark_inclusive_scan
{
	const auto inclusive_scan_gnu = [](auto && policy, const auto & input, auto & output) {
		std::inclusive_scan(std::execution::par, input.begin(), input.end(), output.begin());
	};
} // namespace benchmark_inclusive_scan
