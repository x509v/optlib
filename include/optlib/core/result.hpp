#pragma once

#include <cstdint>
#include <string_view>
#include <optlib/core/types.hpp>

namespace optlib {

enum class Status : uint8_t {
    Converged,
    MaxIterationsReached,
    LineSearchFailed,
    NumericalError,
};

constexpr std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::Converged:            return "Converged";
        case Status::MaxIterationsReached: return "MaxIterationsReached";
        case Status::LineSearchFailed:     return "LineSearchFailed";
        case Status::NumericalError:       return "NumericalError";
    }
    return "Unknown";
}

struct Result {
    Vector   x;                           // solution (or best found)
    Scalar   f_val{};                     // objective at x
    Vector   grad;                        // gradient at x (if available)
    uint32_t iterations{};                // iterations performed
    uint32_t f_evals{};                   // function evaluations
    uint32_t grad_evals{};                // gradient evaluations
    Status   status{Status::Converged};

    [[nodiscard]] bool ok() const noexcept { return status == Status::Converged; }
};

} // namespace optlib
