// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// Minimizer - implementation
// =============================================================================
//
// One Impl per algorithm, each instantiating the template entry points on
// std::function objectives. Adding an algorithm is one Impl, one name in
// kNames and one line in create(). An algorithm overrides the setters of its
// own options; the base versions throw, so a setter the selected algorithm
// has no use for is an error at the call rather than a value nothing reads.

#include "flop/minimizer.hpp"

#include <array>
#include <stdexcept>
#include <string>

#include "flop/cobyla.hpp"
#include "flop/nelder_mead.hpp"

namespace flop {

namespace {

constexpr std::array<std::string_view, 2> kNames = {"COBYLA", "NELDER_MEAD"};

[[noreturn]] void no_such_option(std::string_view algorithm, const char* setter) {
    throw std::invalid_argument(std::string("flop::Minimizer::") + setter + ": " +
                                std::string(algorithm) + " has no such option");
}

}  // namespace

struct Minimizer::Impl {
    struct Cobyla;
    struct NelderMead;
    virtual ~Impl() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void set_final_trust_radius(double) {
        no_such_option(name(), "set_final_trust_radius");
    }
    virtual void set_adaptive_coefficients(bool) {
        no_such_option(name(), "set_adaptive_coefficients");
    }
    virtual Result run(const Objective& f, const Constraints* c, std::size_t m,
                       std::span<const double> x0, const Options& opts) const = 0;
    virtual Result run(const BatchObjective& f, const Constraints* c, std::size_t m,
                       std::span<const double> x0, const Options& opts) const = 0;
};

// Impl is private to Minimizer, so the algorithm implementations are declared
// as nested types of it rather than in an anonymous namespace.
struct Minimizer::Impl::Cobyla final : Minimizer::Impl {
    double rhoend = 0.0;

    [[nodiscard]] std::string_view name() const noexcept override { return "COBYLA"; }
    void set_final_trust_radius(double r) override { rhoend = r; }

    [[nodiscard]] cobyla::Options make(const Options& shared) const {
        cobyla::Options o;
        static_cast<Options&>(o) = shared;
        o.final_trust_radius = rhoend;
        return o;
    }

    Result run(const Minimizer::Objective& f, const Minimizer::Constraints* c, std::size_t m,
               std::span<const double> x0, const Options& opts) const override {
        const cobyla::Options o = make(opts);
        if (c) return cobyla::minimize(f, *c, m, x0, o);
        return cobyla::minimize(f, x0, o);
    }

    Result run(const Minimizer::BatchObjective& f, const Minimizer::Constraints* c, std::size_t m,
               std::span<const double> x0, const Options& opts) const override {
        const cobyla::Options o = make(opts);
        if (c) return cobyla::minimize_batch(f, *c, m, x0, o);
        return cobyla::minimize_batch(f, x0, o);
    }
};

// Nelder-Mead has no nonlinear constraints. A caller who passes some is told
// so before any evaluation, rather than having them ignored.
struct Minimizer::Impl::NelderMead final : Minimizer::Impl {
    bool adaptive = true;

    [[nodiscard]] std::string_view name() const noexcept override { return "NELDER_MEAD"; }
    void set_adaptive_coefficients(bool a) override { adaptive = a; }

    [[nodiscard]] nelder_mead::Options make(const Options& shared) const {
        nelder_mead::Options o;
        static_cast<Options&>(o) = shared;
        o.adaptive_coefficients = adaptive;
        return o;
    }

    static void refuse_constraints(const Minimizer::Constraints* c) {
        if (c)
            throw std::invalid_argument(
                "flop::Minimizer::minimize: NELDER_MEAD does not take nonlinear constraints");
    }

    Result run(const Minimizer::Objective& f, const Minimizer::Constraints* c, std::size_t,
               std::span<const double> x0, const Options& opts) const override {
        refuse_constraints(c);
        return nelder_mead::minimize(f, x0, make(opts));
    }

    Result run(const Minimizer::BatchObjective& f, const Minimizer::Constraints* c, std::size_t,
               std::span<const double> x0, const Options& opts) const override {
        refuse_constraints(c);
        return nelder_mead::minimize_batch(f, x0, make(opts));
    }
};

Minimizer Minimizer::create(std::string_view name) {
    if (name == "COBYLA") return Minimizer(std::make_unique<Impl::Cobyla>());
    if (name == "NELDER_MEAD") return Minimizer(std::make_unique<Impl::NelderMead>());
    std::string known;
    for (std::string_view n : kNames) {
        if (!known.empty()) known += ", ";
        known += n;
    }
    throw std::invalid_argument("flop::Minimizer::create: unknown algorithm '" + std::string(name) +
                                "' (known: " + known + ")");
}

std::span<const std::string_view> Minimizer::names() noexcept {
    return kNames;
}

std::string_view Minimizer::name() const noexcept {
    return impl_->name();
}

Minimizer& Minimizer::set_final_trust_radius(double rhoend) {
    impl_->set_final_trust_radius(rhoend);
    return *this;
}

Minimizer& Minimizer::set_adaptive_coefficients(bool adaptive) {
    impl_->set_adaptive_coefficients(adaptive);
    return *this;
}

Result Minimizer::minimize(const Objective& f, std::span<const double> x0,
                           const Options& opts) const {
    return impl_->run(f, nullptr, 0, x0, opts);
}

Result Minimizer::minimize(const Objective& f, const Constraints& c, std::size_t n_constraints,
                           std::span<const double> x0, const Options& opts) const {
    return impl_->run(f, &c, n_constraints, x0, opts);
}

Result Minimizer::minimize(const BatchObjective& f, std::span<const double> x0,
                           const Options& opts) const {
    return impl_->run(f, nullptr, 0, x0, opts);
}

Result Minimizer::minimize(const BatchObjective& f, const Constraints& c, std::size_t n_constraints,
                           std::span<const double> x0, const Options& opts) const {
    return impl_->run(f, &c, n_constraints, x0, opts);
}

Minimizer::Minimizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Minimizer::~Minimizer() = default;
Minimizer::Minimizer(Minimizer&&) noexcept = default;
Minimizer& Minimizer::operator=(Minimizer&&) noexcept = default;

}  // namespace flop
