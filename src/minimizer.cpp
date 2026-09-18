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
// kNames and one line in create().

#include "flop/minimizer.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace flop {

namespace {

constexpr std::array<std::string_view, 1> kNames = {"COBYLA"};

}  // namespace

struct Minimizer::Impl {
    struct Cobyla;
    virtual ~Impl() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    virtual void set_final_trust_radius(double) = 0;
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

Minimizer Minimizer::create(std::string_view name) {
    if (name == "COBYLA") return Minimizer(std::make_unique<Impl::Cobyla>());
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
