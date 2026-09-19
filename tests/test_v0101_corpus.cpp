// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: MPL-2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// 0.1.0.1 test wave: replay of recorded NLopt COBYLA runs.
//
// tests/data/nlopt-corpus/ holds nineteen runs of NLopt 2.7.1's COBYLA on
// QAOA and MA-QAOA energy landscapes (2 to 80 parameters, initial step 0.3,
// no active bounds), each as the sequence of points NLopt evaluated with
// the energy at each. A run is replayed by handing FLOP the same x0 and
// step and an objective that answers only the recorded points, in order:
// the k-th call must be the k-th recorded point to the bit, and the answer
// is the k-th recorded energy. The first call FLOP makes that is not the
// next recorded point ends the replay.
//
// What is asserted is the initial simplex: x0 and the n points x0 + 0.3 e_i
// built from the running best base, which the paper fixes and which is a
// single addition per coordinate, so it is exact under both floating-point
// models. How far past the simplex the two implementations agree is
// recorded as a test property and printed, not asserted: they share the
// paper, not the code, and the divergence point is information about where
// the paper leaves a choice.
//
// VQE runs are absent from the corpus because they used NLopt's default
// initial step, which is |x0_i| per coordinate; a single Powell rhobeg
// cannot reproduce it.

#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "flop/flop.hpp"
#include "v0101_problems.hpp"

#ifndef FLOP_TEST_DATA_DIR
#error "FLOP_TEST_DATA_DIR must name the tests/data directory"
#endif

namespace {

struct Recording {
    std::string name;
    std::size_t n = 0;
    std::vector<double> f;               // one per evaluation
    std::vector<std::vector<double>> x;  // one point per evaluation
};

// One token of a line as a double, exactly as written (from_chars rounds
// correctly, so a shortest round-trip decimal gives back its double).
bool parse_double(std::string_view tok, double& out) {
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), out);
    return res.ec == std::errc{} && res.ptr == tok.data() + tok.size();
}

Recording load(const std::filesystem::path& file) {
    Recording run;
    run.name = file.stem().string();
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open " + file.string());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> vals;
        std::size_t pos = 0;
        while (pos < line.size()) {
            const std::size_t end = line.find(' ', pos);
            const std::string_view tok(line.data() + pos,
                                       (end == std::string::npos ? line.size() : end) - pos);
            double v = 0.0;
            if (!parse_double(tok, v)) throw std::runtime_error("bad number in " + file.string());
            vals.push_back(v);
            pos = (end == std::string::npos) ? line.size() : end + 1;
        }
        if (vals.size() < 2) throw std::runtime_error("short line in " + file.string());
        run.f.push_back(vals[0]);
        run.x.emplace_back(vals.begin() + 1, vals.end());
    }
    if (run.x.empty()) throw std::runtime_error("no evaluations in " + file.string());
    run.n = run.x[0].size();
    return run;
}

std::vector<std::filesystem::path> corpus_files() {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(
             std::filesystem::path(FLOP_TEST_DATA_DIR) / "nlopt-corpus"))
        if (entry.path().extension() == ".txt") files.push_back(entry.path());
    std::ranges::sort(files);
    return files;
}

// Thrown by the replay objective at the first point FLOP asks for that is
// not the next recorded one; carries how many agreed.
struct Diverged {
    std::size_t agreed;
};

// Replays one run; returns the number of leading evaluations that matched.
std::size_t replay(const Recording& run) {
    std::size_t k = 0;
    auto f = [&](std::span<const double> x) -> double {
        if (k >= run.x.size() || !v0101::same_bits(x, run.x[k])) throw Diverged{k};
        return run.f[k++];
    };
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.max_evaluations = run.x.size();
    o.stopping.xtol_rel = 1e-12;
    try {
        (void)flop::cobyla::minimize(f, run.x[0], o);
    } catch (const Diverged& d) {
        return d.agreed;
    }
    return k;
}

// The same replay to a tolerance: a point counts as the recorded one when
// every coordinate is within tol (relative to 1 + |x|) of it, and the
// recorded energy is handed back for it. Rounding grows slowly along a
// trajectory and a rule difference moves a whole step at once, so the
// distance of the first mismatch, in units of the initial step, tells the
// two apart. Returns the agreed count and that distance.
struct TolerantReplay {
    std::size_t agreed;
    double mismatch_over_step;
};

TolerantReplay replay_to(const Recording& run, double tol) {
    std::size_t k = 0;
    double mismatch = 0.0;
    auto f = [&](std::span<const double> x) -> double {
        if (k >= run.x.size()) throw Diverged{k};
        double worst = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double scale = 1.0 + std::fabs(run.x[k][i]);
            worst = std::max(worst, std::fabs(x[i] - run.x[k][i]) / scale);
        }
        if (worst > tol) {
            mismatch = v0101::max_abs_diff(x, run.x[k]);
            throw Diverged{k};
        }
        return run.f[k++];
    };
    flop::cobyla::Options o;
    o.initial_step = 0.3;
    o.stopping.max_evaluations = run.x.size();
    o.stopping.xtol_rel = 1e-12;
    try {
        (void)flop::cobyla::minimize(f, run.x[0], o);
    } catch (const Diverged& d) {
        return {.agreed = d.agreed, .mismatch_over_step = mismatch / o.initial_step};
    }
    return {.agreed = k, .mismatch_over_step = 0.0};
}

class V0101Corpus : public ::testing::TestWithParam<std::filesystem::path> {};

}  // namespace

TEST_P(V0101Corpus, TheInitialSimplexReplaysToTheBit) {
    const Recording run = load(GetParam());
    ASSERT_GE(run.x.size(), run.n + 1) << run.name;
    const std::size_t agreed = replay(run);
    EXPECT_GE(agreed, run.n + 1) << run.name << ": diverged inside the initial simplex";
    RecordProperty("n", static_cast<int>(run.n));
    RecordProperty("recorded", static_cast<int>(run.x.size()));
    RecordProperty("agreed", static_cast<int>(agreed));
    std::cout << "[ corpus   ] " << run.name << ": n=" << run.n << " recorded=" << run.x.size()
              << " agreed=" << agreed << '\n';
}

TEST_P(V0101Corpus, TheTrajectoryAgreesToATolerance) {
    const Recording run = load(GetParam());
    const TolerantReplay r = replay_to(run, 1e-9);
    EXPECT_GE(r.agreed, run.n + 1) << run.name;
    RecordProperty("agreed_1e-9", static_cast<int>(r.agreed));
    std::cout << "[ corpus   ] " << run.name << ": agreed_1e-9=" << r.agreed << " of "
              << run.x.size() << " first mismatch " << r.mismatch_over_step << " steps\n";
}

TEST(V0101CorpusFiles, TheCorpusHoldsNineteenRuns) {
    EXPECT_EQ(corpus_files().size(), 19u);
}

INSTANTIATE_TEST_SUITE_P(Runs, V0101Corpus, ::testing::ValuesIn(corpus_files()),
                         [](const ::testing::TestParamInfo<std::filesystem::path>& tpi) {
                             std::string s = tpi.param.stem().string();
                             for (char& ch : s)
                                 if (ch == '-') ch = '_';
                             return s;
                         });
