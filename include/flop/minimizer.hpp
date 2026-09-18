// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Minimizer - the type-erased facade, one algorithm by name
// =============================================================================
//
// The template entry points inline the objective into the loop and are the
// fast path. This facade serves the caller who selects an algorithm at run
// time by its name, holds the objective as a std::function, and is what a C
// shim would wrap. It is compiled once, in the library, so every consumer of
// it runs the same code under the library's floating-point model.
//
// create() is the only place in FLOP where a name is compared with a string,
// and it throws std::invalid_argument on a name it does not know. Nothing
// falls back to a default algorithm.

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "flop/cobyla.hpp"
#include "flop/options.hpp"
#include "flop/result.hpp"

namespace flop {

class Minimizer {
public:
    using Objective = std::function<double(std::span<const double>)>;
    using BatchObjective =
        std::function<void(std::span<const std::span<const double>>, std::span<double>)>;
    using Constraints = std::function<void(std::span<const double>, std::span<double>)>;

    // "COBYLA". Case-sensitive. Unknown names throw std::invalid_argument.
    static Minimizer create(std::string_view name);
    static std::span<const std::string_view> names() noexcept;

    [[nodiscard]] std::string_view name() const noexcept;

    // The algorithm's own options, beyond the shared flop::Options. For
    // COBYLA: final_trust_radius. Every algorithm accepts the shared set.
    Minimizer& set_final_trust_radius(double rhoend);

    [[nodiscard]] Result minimize(const Objective& f, std::span<const double> x0,
                                  const Options& opts) const;
    [[nodiscard]] Result minimize(const Objective& f, const Constraints& c,
                                  std::size_t n_constraints, std::span<const double> x0,
                                  const Options& opts) const;
    [[nodiscard]] Result minimize(const BatchObjective& f, std::span<const double> x0,
                                  const Options& opts) const;
    [[nodiscard]] Result minimize(const BatchObjective& f, const Constraints& c,
                                  std::size_t n_constraints, std::span<const double> x0,
                                  const Options& opts) const;

    ~Minimizer();
    Minimizer(Minimizer&&) noexcept;
    Minimizer& operator=(Minimizer&&) noexcept;
    Minimizer(const Minimizer&) = delete;
    Minimizer& operator=(const Minimizer&) = delete;

private:
    struct Impl;
    explicit Minimizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace flop
