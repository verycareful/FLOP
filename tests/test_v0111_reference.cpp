// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.1.1: FLOP against the reference transcription of Lagarias et al.,
// section 2 (v0111_reference_nm.hpp).
//
// FLOP keeps a running sum for the centroid, a rank index for the order,
// and per-coordinate extents for the radius; the reference recomputes all
// three from scratch and states the tie rules as printed. In exact
// arithmetic the two must evaluate the same points to the bit, and the
// reference runs on its own here, so nothing of FLOP's reaches it but the
// options. Outside exact arithmetic the reference replays FLOP's trace and
// must ask for every point FLOP evaluated, to the rounding the incremental
// centroid allows, over runs many times longer than the n + 1 replacements
// after which FLOP rebuilds its sum, and must stop where FLOP stopped.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "flop/flop.hpp"
#include "v0111_problems.hpp"
#include "v0111_reference_nm.hpp"

namespace {

using Vec = std::vector<double>;

double staircase(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(4.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

// Integer values on a lattice of half-integers, with a step in the first
// coordinate: ties everywhere, so shrinks and the tie rules run constantly.
double lattice(std::span<const double> x) {
    double s = x[0] > v0111::sphere_centre(0) ? 1.0 : 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += std::floor(2.0 * std::fabs(x[i] - v0111::sphere_centre(i) - 0.11));
    return s;
}

double coupled(std::span<const double> x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - v0111::sphere_centre(i);
        s += d * d;
        if (i + 1 < x.size()) {
            const double e = x[i] - x[i + 1];
            s += 10.0 * e * e;
        }
    }
    return s;
}

}  // namespace

TEST(V0111Reference, InExactArithmeticTheSamePointsToTheBit) {
    // FLOP and the reference run independently on the same objective. Up to
    // the first point that could have been rounded (the exact prefix of
    // either trace), every evaluated point is the same to the bit.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        for (const bool adaptive : {false, true}) {
            for (auto* f : {&staircase, &lattice, &v0111::sphere}) {
                const flop::nelder_mead::Options o = v0111::options(1.0, 400, adaptive);
                const Vec x0(n, 0.0);
                const v0111::Traced flop_run = v0111::traced(f, x0, o);
                v0111::Reference ref(x0, o);
                auto oracle = v0111::direct(*f);
                const v0111::ReferenceRun ref_run = ref.run(oracle);
                const std::size_t exact = std::min(v0111::exact_prefix(flop_run.trace),
                                                   v0111::exact_prefix(oracle.trace));
                EXPECT_GE(exact, 2 * (n + 1)) << "n " << n << ": too short to say anything";
                ASSERT_GE(flop_run.trace.size(), exact);
                ASSERT_GE(oracle.trace.size(), exact);
                for (std::size_t k = 0; k < exact; ++k) {
                    ASSERT_TRUE(v0111::same_bits(flop_run.trace[k].x, oracle.trace[k].x))
                        << "n " << n << (adaptive ? " adaptive" : " standard")
                        << ", first difference at evaluation " << k;
                }
                if (exact == flop_run.trace.size() && exact == oracle.trace.size()) {
                    EXPECT_EQ(flop_run.r.status, ref_run.result.status);
                    EXPECT_TRUE(v0111::same_bits(flop_run.r.x, ref_run.result.x));
                }
            }
        }
    }
}

TEST(V0111Reference, LongRunsAgreeToRoundingAndStopTogether) {
    struct Case {
        const char* name;
        double (*f)(std::span<const double>);
        std::size_t n;
        Vec x0;
        double step;
        double xtol;
        std::size_t cap;
    };
    std::vector<Case> cases{
        {.name = "sphere 3",
         .f = v0111::sphere,
         .n = 3,
         .x0 = {},
         .step = 1.0,
         .xtol = 1e-12,
         .cap = 50000},
        {.name = "coupled 5",
         .f = coupled,
         .n = 5,
         .x0 = {},
         .step = 1.0,
         .xtol = 1e-12,
         .cap = 50000},
        {.name = "sphere 16",
         .f = v0111::sphere,
         .n = 16,
         .x0 = {},
         .step = 1.0,
         .xtol = 1e-10,
         .cap = 20000},
        {.name = "coupled 33",
         .f = coupled,
         .n = 33,
         .x0 = {},
         .step = 1.0,
         .xtol = 0.0,
         .cap = 20000},
        {.name = "sphere 96",
         .f = v0111::sphere,
         .n = 96,
         .x0 = {},
         .step = 1.0,
         .xtol = 0.0,
         .cap = 20000},
        {.name = "rosenbrock 2",
         .f = v0111::extended_rosenbrock,
         .n = 2,
         .x0 = {-1.2, 1.0},
         .step = 0.5,
         .xtol = 1e-12,
         .cap = 50000},
        {.name = "powell singular",
         .f = v0111::powell_singular,
         .n = 4,
         .x0 = {3.0, -1.0, 0.0, 1.0},
         .step = 1.0,
         .xtol = 1e-12,
         .cap = 50000},
        {.name = "staircase 5",
         .f = staircase,
         .n = 5,
         .x0 = {},
         .step = 1.0,
         .xtol = 0.0,
         .cap = 3000},
    };
    for (Case& c : cases) {
        if (c.x0.empty()) c.x0.assign(c.n, 0.0);
        for (const bool adaptive : {false, true}) {
            flop::nelder_mead::Options o = v0111::options(c.step, c.cap, adaptive);
            o.stopping.xtol_abs = c.xtol;
            const v0111::Traced run = v0111::traced(c.f, c.x0, o);
            v0111::ReferenceRun ref;
            const v0111::AuditReport rep = v0111::audit(c.x0, o, run.trace, run.r, false, &ref);
            EXPECT_TRUE(rep.points_agree) << c.name << ": " << rep.detail;
            EXPECT_TRUE(rep.same_length) << c.name << ": " << rep.detail;
            EXPECT_TRUE(rep.same_outcome) << c.name << ": " << rep.detail;
            EXPECT_TRUE(rep.same_radius) << c.name << ": " << rep.detail;
            // Many replacements past every rebuild of the running sum.
            EXPECT_GE(ref.top_radius.size(), 10 * (c.n + 1)) << c.name;
        }
    }
}

TEST(V0111Reference, TheAuditCatchesADeviation) {
    // The comparison is only worth what it can catch: a trace with one point
    // moved by far less than any step is reported at that point.
    flop::nelder_mead::Options o = v0111::options(1.0, 300);
    const Vec x0(3, 0.0);
    v0111::Traced run = v0111::traced(v0111::sphere, x0, o);
    ASSERT_GT(run.trace.size(), 100u);
    run.trace[100].x[1] += 1e-9;
    const v0111::AuditReport rep = v0111::audit(x0, o, run.trace, run.r, false);
    EXPECT_FALSE(rep.points_agree);
    EXPECT_NE(rep.detail.find("evaluation 100:"), std::string::npos) << rep.detail;
}
