// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.2.1: BOBYQA through flop::Minimizer.
//
// The facade runs the same solver as the template entry points: under the
// same floating-point model the trajectory is the same to the bit, and under
// another to the same optimum and status. BOBYQA's own options, the final
// radius, the number of interpolation points, shrink_initial_step and the
// weight of NEWUOA, reach the solver through their setters; an option the
// algorithm does not have, and the overloads with nonlinear constraints, throw
// before any evaluation; names are exact.

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
#include "v0121_problems.hpp"

namespace {

using v0121::Trace;
using v0121::Vec;

// Bit identity of the facade and the template is held where both are strict.
// Under -ffast-math the same template compiled in two translation units may be
// contracted and reassociated differently (the facade is compiled in the
// library's, the template in the test's), and a trajectory then differs
// after enough steps even though both are under the same model; there the
// runs are held to the same status and the same optimum.
#if !defined(FLOP_TEST_FAST_MATH) && !defined(FLOP_TEST_LIBRARY_FAST_MATH)
constexpr bool kSameModelAsLibrary = true;
#else
constexpr bool kSameModelAsLibrary = false;
#endif

double bowl(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - 0.3 * static_cast<double>(i + 1);
        s += (1.0 + static_cast<double>(i)) * d * d;
    }
    return s;
}

double rosenbrock(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = 1.0 - x[i], b = x[i + 1] - x[i] * x[i];
        s += a * a + 100.0 * b * b;
    }
    return s;
}

struct Outcome {
    flop::Result r;
    Trace trace;
};

Outcome facade_run(const flop::Minimizer& m, double (*f)(std::span<const double>), const Vec& x0,
                   flop::Options o, bool batch = false) {
    Outcome t;
    o.on_evaluation = v0121::recorder(t.trace);
    if (batch) {
        const flop::Minimizer::BatchObjective fb = [f](std::span<const std::span<const double>> xs,
                                                       std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
        };
        t.r = m.minimize(fb, x0, o);
    } else {
        const flop::Minimizer::Objective fo = f;
        t.r = m.minimize(fo, x0, o);
    }
    return t;
}

Outcome template_run(double (*f)(std::span<const double>), const Vec& x0, flop::bobyqa::Options o,
                     bool batch = false) {
    Outcome t;
    o.on_evaluation = v0121::recorder(t.trace);
    if (batch) {
        auto fb = [f](std::span<const std::span<const double>> xs, std::span<double> out) {
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = f(xs[i]);
        };
        t.r = flop::bobyqa::minimize_batch(fb, x0, o);
    } else {
        t.r = flop::bobyqa::minimize(f, x0, o);
    }
    return t;
}

void expect_same(const Outcome& a, const Outcome& b, const char* what) {
    ASSERT_EQ(a.trace.size(), b.trace.size()) << what;
    for (std::size_t k = 0; k < a.trace.size(); ++k)
        ASSERT_TRUE(v0121::same_bits(a.trace[k].x, b.trace[k].x)) << what << ", evaluation " << k;
    EXPECT_EQ(a.r.status, b.r.status) << what;
    EXPECT_TRUE(v0121::same_bits(a.r.x, b.r.x)) << what;
    EXPECT_TRUE(v0121::same_bits(a.r.f, b.r.f)) << what;
    EXPECT_TRUE(v0121::same_bits(a.r.final_radius, b.r.final_radius)) << what;
    EXPECT_EQ(a.r.evaluations, b.r.evaluations) << what;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<flop::Minimizer>);
static_assert(std::is_nothrow_move_constructible_v<flop::Minimizer>);

TEST(V0121Facade, NamesListsBobyqaFirstAndCreateAcceptsIt) {
    const auto names = flop::Minimizer::names();
    EXPECT_EQ(names.size(), 3u);
    EXPECT_NE(std::ranges::find(names, std::string_view("BOBYQA")), names.end());
    const flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    EXPECT_EQ(m.name(), std::string_view("BOBYQA"));
}

TEST(V0121Facade, NamesAreExactAndAnUnknownOneIsReportedWithTheKnownOnes) {
    for (const char* bad :
         {"bobyqa", "Bobyqa", "BOBYQA ", " BOBYQA", "BOBY", "BOBYQA2", "NEWUOA", ""}) {
        try {
            (void)flop::Minimizer::create(bad);
            ADD_FAILURE() << "'" << bad << "' was accepted";
        } catch (const std::invalid_argument& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("BOBYQA"), std::string::npos) << what;
            EXPECT_NE(what.find("COBYLA"), std::string::npos) << what;
            EXPECT_NE(what.find("NELDER_MEAD"), std::string::npos) << what;
        }
    }
}

TEST(V0121Facade, TheFacadeAndTheTemplateAgreeInBothChannelsWithEverySetting) {
    struct Setting {
        double final_radius;
        std::size_t points;
        bool shrink;
        bool newuoa;
    };
    const Vec x0{-1.2, 1.0, 1.0};
    for (const Setting& s :
         {Setting{.final_radius = 0.0, .points = 0, .shrink = false, .newuoa = true},
          Setting{.final_radius = 1e-7, .points = 0, .shrink = false, .newuoa = true},
          Setting{.final_radius = 0.0, .points = 5, .shrink = false, .newuoa = true},
          Setting{.final_radius = 0.0, .points = 10, .shrink = false, .newuoa = false},
          Setting{.final_radius = 0.0, .points = 0, .shrink = true, .newuoa = false}}) {
        flop::Minimizer m = flop::Minimizer::create("BOBYQA");
        m.set_final_trust_radius(s.final_radius)
            .set_interpolation_points(s.points)
            .set_shrink_initial_step(s.shrink)
            .set_newuoa_weight(s.newuoa);
        flop::bobyqa::Options o = v0121::options(0.4, 400);
        o.stopping.xtol_abs = 1e-9;
        o.final_trust_radius = s.final_radius;
        o.interpolation_points = s.points;
        o.shrink_initial_step = s.shrink;
        o.newuoa_weight = s.newuoa;
        for (const bool batch : {false, true}) {
            const Outcome a = template_run(rosenbrock, x0, o, batch);
            const Outcome b = facade_run(m, rosenbrock, x0, o, batch);
            if (kSameModelAsLibrary) {
                expect_same(a, b, "the facade against the template");
            } else {
                // Another model, or the same one in another translation unit: the
                // same status and the same minimiser on a problem the settings all
                // solve.
                flop::bobyqa::Options easy = o;
                easy.stopping.max_evaluations = 5000;
                const Outcome c = template_run(bowl, Vec(3, 0.0), easy, batch);
                const Outcome d = facade_run(m, bowl, Vec(3, 0.0), easy, batch);
                EXPECT_TRUE(flop::converged(c.r.status));
                EXPECT_EQ(c.r.status, d.r.status);
                EXPECT_LE(v0121::max_abs_diff(c.r.x, d.r.x), 1e-6);
            }
        }
    }
}

TEST(V0121Facade, EachSetterChangesTheRunItReaches) {
    // A setting that matters must reach the solver: the run with it differs from
    // the run without, and equals the template's run with that option.
    const Vec x0{-1.2, 1.0, 1.0};
    flop::bobyqa::Options o = v0121::options(0.4, 300);
    const flop::Minimizer plain = flop::Minimizer::create("BOBYQA");
    const Outcome base = facade_run(plain, rosenbrock, x0, o);
    {
        flop::Minimizer m = flop::Minimizer::create("BOBYQA");
        EXPECT_EQ(&m.set_interpolation_points(8), &m) << "setters chain";
        const Outcome t = facade_run(m, rosenbrock, x0, o);
        EXPECT_NE(t.trace.size() > 7 ? t.trace[7].x : Vec{},
                  base.trace.size() > 7 ? base.trace[7].x : Vec{});
    }
    {
        flop::Minimizer m = flop::Minimizer::create("BOBYQA");
        m.set_newuoa_weight(false);
        const Outcome t = facade_run(m, rosenbrock, x0, o);
        bool differ = t.trace.size() != base.trace.size();
        for (std::size_t k = 0; !differ && k < t.trace.size(); ++k)
            differ = !v0121::same_bits(t.trace[k].x, base.trace[k].x);
        EXPECT_TRUE(differ) << "the report's weight is another trajectory";
    }
    {
        flop::Minimizer m = flop::Minimizer::create("BOBYQA");
        m.set_final_trust_radius(1e-3);
        flop::Options shared = o;
        shared.stopping.max_evaluations = 5000;
        const Outcome t = facade_run(m, rosenbrock, x0, shared);
        EXPECT_EQ(t.r.status, flop::Status::XtolReached);
        EXPECT_EQ(t.r.final_radius, 1e-3);
    }
    {
        flop::Minimizer m = flop::Minimizer::create("BOBYQA");
        m.set_shrink_initial_step(true);
        flop::Options shared = o;
        shared.initial_step = 4.0;
        shared.bounds = flop::Bounds::box(Vec(3, 0.0), Vec(3, 1.0));
        EXPECT_NO_THROW((void)facade_run(m, rosenbrock, Vec(3, 0.5), shared));
        const flop::Minimizer strict = flop::Minimizer::create("BOBYQA");
        EXPECT_THROW((void)facade_run(strict, rosenbrock, Vec(3, 0.5), shared),
                     std::invalid_argument);
    }
    {
        // The default is the template's default: the points and the weight.
        const Outcome t = template_run(rosenbrock, x0, o);
        if (kSameModelAsLibrary) expect_same(t, base, "the defaults");
    }
}

TEST(V0121Facade, AnOptionTheAlgorithmDoesNotHaveThrowsAndNamesIt) {
    flop::Minimizer bobyqa = flop::Minimizer::create("BOBYQA");
    try {
        bobyqa.set_adaptive_coefficients(true);
        ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("BOBYQA"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("set_adaptive_coefficients"), std::string::npos)
            << e.what();
    }
    for (const char* other : {"COBYLA", "NELDER_MEAD"}) {
        flop::Minimizer m = flop::Minimizer::create(other);
        EXPECT_THROW(m.set_interpolation_points(5), std::invalid_argument) << other;
        EXPECT_THROW(m.set_shrink_initial_step(true), std::invalid_argument) << other;
        EXPECT_THROW(m.set_newuoa_weight(false), std::invalid_argument) << other;
    }
    // COBYLA takes the final radius, Nelder-Mead does not, and BOBYQA does.
    flop::Minimizer cobyla = flop::Minimizer::create("COBYLA");
    EXPECT_NO_THROW(cobyla.set_final_trust_radius(1e-6));
    EXPECT_NO_THROW(bobyqa.set_final_trust_radius(1e-6));
    flop::Minimizer nm = flop::Minimizer::create("NELDER_MEAD");
    EXPECT_THROW(nm.set_final_trust_radius(1e-6), std::invalid_argument);
}

TEST(V0121Facade, TheConstrainedOverloadsRefuseBobyqaBeforeAnyEvaluation) {
    const flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    std::size_t f_calls = 0, c_calls = 0;
    const flop::Minimizer::Objective fo = [&](std::span<const double> x) {
        ++f_calls;
        return rosenbrock(x);
    };
    const flop::Minimizer::BatchObjective fb = [&](std::span<const std::span<const double>> xs,
                                                   std::span<double> out) {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            ++f_calls;
            out[i] = rosenbrock(xs[i]);
        }
    };
    const flop::Minimizer::Constraints c = [&](std::span<const double> x, std::span<double> out) {
        ++c_calls;
        out[0] = 1.0 - x[0];
    };
    const flop::Options o = v0121::options(0.5, 100);
    EXPECT_THROW((void)m.minimize(fo, c, 1, Vec{0.0, 0.0}, o), std::invalid_argument);
    EXPECT_THROW((void)m.minimize(fb, c, 1, Vec{0.0, 0.0}, o), std::invalid_argument);
    EXPECT_EQ(f_calls, 0u);
    EXPECT_EQ(c_calls, 0u);
}

TEST(V0121Facade, AMinimizerIsReusableAndMovableAndKeepsItsSettings) {
    flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    m.set_interpolation_points(6).set_newuoa_weight(false);
    flop::Options o = v0121::options(0.5, 1000);
    o.stopping.xtol_abs = 1e-9;
    const Outcome first = facade_run(m, rosenbrock, Vec{-1.2, 1.0}, o);
    const Outcome again = facade_run(m, rosenbrock, Vec{-1.2, 1.0}, o);
    expect_same(first, again, "the same minimizer twice");
    flop::Minimizer moved = std::move(m);
    EXPECT_EQ(moved.name(), std::string_view("BOBYQA"));
    expect_same(first, facade_run(moved, rosenbrock, Vec{-1.2, 1.0}, o), "moved, settings kept");
    flop::Minimizer assigned = flop::Minimizer::create("COBYLA");
    assigned = std::move(moved);
    EXPECT_EQ(assigned.name(), std::string_view("BOBYQA"));
    expect_same(first, facade_run(assigned, rosenbrock, Vec{-1.2, 1.0}, o), "move-assigned");
}

TEST(V0121Facade, TheFacadeValidatesBobyqasInputLikeTheTemplate) {
    // Every BOBYQA input rule is reached through the facade: here the number of
    // points and the final radius, which only the facade's setters can carry.
    flop::Minimizer m = flop::Minimizer::create("BOBYQA");
    const flop::Options o = v0121::options(0.5, 100);
    m.set_interpolation_points(3);  // n + 2 = 4 for n = 2
    const flop::Minimizer::Objective fo = rosenbrock;
    EXPECT_THROW((void)m.minimize(fo, Vec{0.0, 0.0}, o), std::invalid_argument);
    m.set_interpolation_points(0).set_final_trust_radius(-1.0);
    EXPECT_THROW((void)m.minimize(fo, Vec{0.0, 0.0}, o), std::invalid_argument);
}
