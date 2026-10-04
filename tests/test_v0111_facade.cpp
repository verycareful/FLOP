// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: NELDER_MEAD behind the Minimizer facade.
//
// The facade is the template entry point behind a name and a std::function,
// compiled once into the library under the library's floating-point flags
// (docs/api/minimizer.md). When this binary shares the library's model, the
// facade and the template are held to the same trajectory to the bit; when
// it does not, to the same optimum and status. The algorithm's own option
// reaches the solver; an option or an overload the algorithm does not have
// throws rather than being ignored; names are exact.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"

namespace {

using Vec = std::vector<double>;

#if defined(FLOP_TEST_FAST_MATH) == defined(FLOP_TEST_LIBRARY_FAST_MATH)
constexpr bool kSameModelAsLibrary = true;
#else
constexpr bool kSameModelAsLibrary = false;
#endif

// The facade's run with the shared options of o and its trace.
v0111::Traced facade_run(const flop::Minimizer& m, double (*f)(std::span<const double>),
                         const Vec& x0, flop::Options o) {
    v0111::Traced t;
    o.on_evaluation = v0111::recorder(t.trace);
    const flop::Minimizer::Objective fo = f;
    t.r = m.minimize(fo, x0, o);
    return t;
}

void expect_same_trace(const v0111::Traced& a, const v0111::Traced& b, const char* what) {
    ASSERT_EQ(a.trace.size(), b.trace.size()) << what;
    for (std::size_t k = 0; k < a.trace.size(); ++k)
        ASSERT_TRUE(v0111::same_bits(a.trace[k].x, b.trace[k].x)) << what << ", evaluation " << k;
    EXPECT_EQ(a.r.status, b.r.status) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.x, b.r.x)) << what;
    EXPECT_TRUE(v0111::same_bits(a.r.final_radius, b.r.final_radius)) << what;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<flop::Minimizer>);
static_assert(!std::is_copy_assignable_v<flop::Minimizer>);
static_assert(std::is_nothrow_move_constructible_v<flop::Minimizer>);
static_assert(std::is_nothrow_move_assignable_v<flop::Minimizer>);

TEST(V0111Facade, NamesListsBothAlgorithmsAndCreateAcceptsNelderMead) {
    const auto names = flop::Minimizer::names();
    EXPECT_EQ(names.size(), 2u);
    EXPECT_NE(std::ranges::find(names, std::string_view("COBYLA")), names.end());
    EXPECT_NE(std::ranges::find(names, std::string_view("NELDER_MEAD")), names.end());
    const flop::Minimizer m = flop::Minimizer::create("NELDER_MEAD");
    EXPECT_EQ(m.name(), std::string_view("NELDER_MEAD"));
}

TEST(V0111Facade, NamesAreExactAndAnUnknownOneIsReportedWithTheKnownOnes) {
    for (const char* bad : {"nelder_mead", "NelderMead", "Nelder_Mead", "NELDER-MEAD",
                            "NELDER_MEAD ", " NELDER_MEAD", "NELDERMEAD", ""}) {
        try {
            (void)flop::Minimizer::create(bad);
            ADD_FAILURE() << "'" << bad << "' was accepted";
        } catch (const std::invalid_argument& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("NELDER_MEAD"), std::string::npos) << what;
            EXPECT_NE(what.find("COBYLA"), std::string::npos) << what;
        }
    }
}

TEST(V0111Facade, TheFacadeAndTheTemplateAgree) {
    const flop::Minimizer m = flop::Minimizer::create("NELDER_MEAD");
    if (kSameModelAsLibrary) {
        // Capped Rosenbrock, both channels: the same code under the same
        // model, so the same trajectory wherever the cap lands.
        const Vec x0{-1.2, 1.0};
        flop::nelder_mead::Options o = v0111::options(0.5, 300);
        const v0111::Traced a = v0111::traced(v0111::extended_rosenbrock, x0, o);
        const v0111::Traced b = facade_run(m, v0111::extended_rosenbrock, x0, o);
        expect_same_trace(a, b, "scalar");
        auto tb = v0111::batched(v0111::extended_rosenbrock);
        const v0111::Traced c = v0111::traced_batch(tb, x0, o);
        v0111::Traced d;
        flop::Options shared = o;
        shared.on_evaluation = v0111::recorder(d.trace);
        const flop::Minimizer::BatchObjective fb = [](std::span<const std::span<const double>> xs,
                                                      std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = v0111::extended_rosenbrock(xs[i]);
        };
        d.r = m.minimize(fb, x0, shared);
        expect_same_trace(c, d, "batch");
    }
    // Across models: the same optimum on a problem that converges.
    const Vec x0(8, 0.0);
    flop::nelder_mead::Options o = v0111::options(1.0, 100000);
    o.stopping.xtol_abs = 1e-9;
    const flop::Result a = flop::nelder_mead::minimize(v0111::sphere, x0, o);
    const flop::Minimizer::Objective fo = v0111::sphere;
    const flop::Result b = m.minimize(fo, x0, o);
    EXPECT_EQ(a.status, flop::Status::XtolReached);
    EXPECT_EQ(b.status, flop::Status::XtolReached);
    EXPECT_LE(v0111::max_abs_diff(a.x, b.x), 1e-7);
}

TEST(V0111Facade, SetAdaptiveCoefficientsReachesTheSolver) {
    flop::Minimizer on = flop::Minimizer::create("NELDER_MEAD");
    flop::Minimizer off = flop::Minimizer::create("NELDER_MEAD");
    EXPECT_EQ(&off.set_adaptive_coefficients(false), &off);  // chains
    on.set_adaptive_coefficients(true);
    const flop::Options o = v0111::options(1.0, 400);
    {
        // n = 2: the two sets are the same, so are the runs.
        const v0111::Traced a = facade_run(on, v0111::sphere, Vec(2, 0.0), o);
        const v0111::Traced b = facade_run(off, v0111::sphere, Vec(2, 0.0), o);
        expect_same_trace(a, b, "n = 2");
    }
    for (const std::size_t n : {std::size_t{3}, std::size_t{8}}) {
        const v0111::Traced a = facade_run(on, v0111::sphere, Vec(n, 0.0), o);
        const v0111::Traced b = facade_run(off, v0111::sphere, Vec(n, 0.0), o);
        bool differ = a.trace.size() != b.trace.size();
        for (std::size_t k = 0; !differ && k < a.trace.size(); ++k)
            differ = !v0111::same_bits(a.trace[k].x, b.trace[k].x);
        EXPECT_TRUE(differ) << "n " << n;
        if (kSameModelAsLibrary) {
            // And each is the template run with that setting.
            const v0111::Traced ta =
                v0111::traced(v0111::sphere, Vec(n, 0.0), v0111::options(1.0, 400, true));
            const v0111::Traced tb =
                v0111::traced(v0111::sphere, Vec(n, 0.0), v0111::options(1.0, 400, false));
            expect_same_trace(ta, a, "adaptive on");
            expect_same_trace(tb, b, "adaptive off");
        }
    }
    // The default is on.
    const flop::Minimizer fresh = flop::Minimizer::create("NELDER_MEAD");
    expect_same_trace(facade_run(fresh, v0111::sphere, Vec(3, 0.0), o),
                      facade_run(on, v0111::sphere, Vec(3, 0.0), o), "default");
}

TEST(V0111Facade, AnOptionTheAlgorithmDoesNotHaveThrowsAndNamesIt) {
    flop::Minimizer nm = flop::Minimizer::create("NELDER_MEAD");
    try {
        nm.set_final_trust_radius(1e-6);
        ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("NELDER_MEAD"), std::string::npos) << e.what();
    }
    flop::Minimizer cobyla = flop::Minimizer::create("COBYLA");
    try {
        cobyla.set_adaptive_coefficients(false);
        ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("COBYLA"), std::string::npos) << e.what();
    }
}

TEST(V0111Facade, TheConstrainedOverloadsRefuseNelderMeadBeforeAnyEvaluation) {
    const flop::Minimizer m = flop::Minimizer::create("NELDER_MEAD");
    std::size_t f_calls = 0, c_calls = 0;
    const flop::Minimizer::Objective fo = [&](std::span<const double> x) {
        ++f_calls;
        return v0111::sphere(x);
    };
    const flop::Minimizer::BatchObjective fb = [&](std::span<const std::span<const double>> xs,
                                                   std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            ++f_calls;
            out[i] = v0111::sphere(xs[i]);
        }
    };
    const flop::Minimizer::Constraints c = [&](std::span<const double> x, std::span<double> out) {
        ++c_calls;
        out[0] = 1.0 - x[0];
    };
    const flop::Options o = v0111::options(0.5, 100);
    EXPECT_THROW((void)m.minimize(fo, c, 1, Vec{0.0, 0.0}, o), std::invalid_argument);
    EXPECT_THROW((void)m.minimize(fb, c, 1, Vec{0.0, 0.0}, o), std::invalid_argument);
    EXPECT_EQ(f_calls, 0u);
    EXPECT_EQ(c_calls, 0u);
}

TEST(V0111Facade, AMinimizerIsReusableAndMovable) {
    flop::Minimizer m = flop::Minimizer::create("NELDER_MEAD");
    m.set_adaptive_coefficients(false);
    flop::Options o = v0111::options(0.5, 1000);
    o.stopping.xtol_abs = 1e-10;
    const v0111::Traced first = facade_run(m, v0111::beale, Vec{1.0, 1.0}, o);
    const v0111::Traced again = facade_run(m, v0111::beale, Vec{1.0, 1.0}, o);
    expect_same_trace(first, again, "the same minimizer twice");
    flop::Minimizer moved = std::move(m);
    EXPECT_EQ(moved.name(), std::string_view("NELDER_MEAD"));
    expect_same_trace(first, facade_run(moved, v0111::beale, Vec{1.0, 1.0}, o),
                      "moved, setting kept");
    flop::Minimizer assigned = flop::Minimizer::create("COBYLA");
    assigned = std::move(moved);
    EXPECT_EQ(assigned.name(), std::string_view("NELDER_MEAD"));
    expect_same_trace(first, facade_run(assigned, v0111::beale, Vec{1.0, 1.0}, o), "move-assigned");
}
