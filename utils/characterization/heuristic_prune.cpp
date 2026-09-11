/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

///
/// Reproducer that drives ``svs::index::vamana::heuristic_prune_neighbors`` through its
/// ``IterativePruneStrategy`` overload, for performance analysis of that overload.
///
/// The pruning strategy is a compile-time function of the distance functor
/// (``svs::index::vamana::PruneStrategy``):
///
///   - ``DistanceIP``               -> ``IterativePruneStrategy``
///   - ``DistanceCosineSimilarity`` -> ``IterativePruneStrategy``
///   - ``DistanceL2``               -> ``ProgressivePruneStrategy``
///
/// This utility therefore builds a Vamana index with MIP (default) or cosine similarity.
/// A ``static_assert`` pins the selected strategy so the reproducer cannot silently stop
/// exercising the intended code path.
///
/// A Vamana build reaches ``heuristic_prune_neighbors`` from two call sites, both in
/// ``svs/index/vamana/vamana_build.h``:
///
///   - ``VamanaBuilder::generate_neighbors`` (pruning to ``graph_max_degree``) -- the
///     dominant one, reached twice per build since ``VamanaIndex::build`` runs two passes
///     (``alpha = 1.0``, then the configured ``alpha``).
///   - ``VamanaBuilder::add_reverse_edges`` (pruning to ``prune_to``) for vertices that
///     exceed the max degree after back-edge insertion.
///
/// Example:
///
///     ./heuristic_prune --num-vectors 200000 --dims 128 --threads 16
///     perf record -g -- ./heuristic_prune --num-vectors 200000 --threads 1
///

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/logging.h"
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/prune.h"
#include "svs/lib/timing.h"
#include "svs/third-party/fmt.h"

#include "svsmain.h"

// stdlib
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Eltype = float;

struct Options {
    // Dataset.
    std::string data_path{};
    size_t num_vectors = 100'000;
    size_t dims = 128;
    size_t num_clusters = 200;
    uint64_t seed = 0xC0FFEE;

    // Build.
    std::string distance = "mip";
    size_t graph_max_degree = 64;
    size_t window_size = 128;
    size_t max_candidate_pool_size = 750;
    size_t prune_to = svs::UNSIGNED_INTEGER_PLACEHOLDER;
    float alpha = svs::FLOAT_PLACEHOLDER;
    bool full_search_history = true;
    size_t num_threads = 0; // 0 -> hardware concurrency
    bool verbose = false;
};

[[noreturn]] void usage() {
    fmt::print(
        R"(Usage: heuristic_prune [options]

Builds a Vamana index whose distance functor selects IterativePruneStrategy.

Dataset:
  --data <path>                  Load vectors from a file instead of generating them.
  --num-vectors <n>              Synthetic dataset size. Default: 100000.
  --dims <n>                     Synthetic dimensionality. Default: 128.
  --num-clusters <n>             Synthetic cluster count. Default: 200.
  --seed <n>                     RNG seed. Default: 12648430.

Build:
  --distance <mip|cosine>        Both select IterativePruneStrategy. Default: mip.
  --graph-max-degree <n>         Prune target of the main call site. Default: 64.
  --window-size <n>              Construction search window size. Default: 128.
  --max-candidate-pool-size <n>  Candidate pool cap, i.e. the prune input size.
                                 Default: 750.
  --prune-to <n>                 Prune target of the back-edge call site.
                                 Default: graph_max_degree - 4.
  --alpha <f>                    Must be <= 1.0 for mip/cosine. Default: 0.95.
  --full-search-history <0|1>    Larger candidate pools when enabled. Default: 1.
  --threads <n>                  Default: hardware concurrency.
  --verbose                      Emit the builder's internal per-phase trace timings.
)"
    );
    std::exit(EXIT_FAILURE);
}

bool parse_bool(const std::string& value) {
    if (value == "1" || value == "true" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no") {
        return false;
    }
    throw ANNEXCEPTION("Could not parse \"{}\" as a boolean!", value);
}

Options parse_options(const std::vector<std::string>& args) {
    auto options = Options{};
    for (size_t i = 1, imax = args.size(); i < imax; ++i) {
        const auto& key = args[i];
        if (key == "--help" || key == "-h") {
            usage();
        }
        if (key == "--verbose") {
            options.verbose = true;
            continue;
        }

        // All remaining options take a value.
        if (i + 1 == imax) {
            throw ANNEXCEPTION("Option \"{}\" requires a value!", key);
        }
        const auto& value = args[++i];

        if (key == "--data") {
            options.data_path = value;
        } else if (key == "--num-vectors") {
            options.num_vectors = std::stoull(value);
        } else if (key == "--dims") {
            options.dims = std::stoull(value);
        } else if (key == "--num-clusters") {
            options.num_clusters = std::stoull(value);
        } else if (key == "--seed") {
            options.seed = std::stoull(value);
        } else if (key == "--distance") {
            options.distance = value;
        } else if (key == "--graph-max-degree") {
            options.graph_max_degree = std::stoull(value);
        } else if (key == "--window-size") {
            options.window_size = std::stoull(value);
        } else if (key == "--max-candidate-pool-size") {
            options.max_candidate_pool_size = std::stoull(value);
        } else if (key == "--prune-to") {
            options.prune_to = std::stoull(value);
        } else if (key == "--alpha") {
            options.alpha = std::stof(value);
        } else if (key == "--full-search-history") {
            options.full_search_history = parse_bool(value);
        } else if (key == "--threads") {
            options.num_threads = std::stoull(value);
        } else {
            throw ANNEXCEPTION("Unrecognized option \"{}\"!", key);
        }
    }

    if (options.num_clusters == 0) {
        throw ANNEXCEPTION("--num-clusters must be positive!");
    }
    return options;
}

// Clustered synthetic data. Uniformly random vectors make every candidate roughly
// equidistant, which is not representative of the pruning workload; clusters give the
// greedy search something to converge on and therefore realistic candidate pools.
svs::data::SimpleData<Eltype> generate_dataset(const Options& options) {
    auto data = svs::data::SimpleData<Eltype>(options.num_vectors, options.dims);
    auto rng = std::mt19937_64(options.seed);
    auto normal = std::normal_distribution<float>(0.0F, 1.0F);

    // Cluster centers.
    auto centers = std::vector<float>(options.num_clusters * options.dims);
    for (auto& v : centers) {
        v = normal(rng);
    }

    auto pick_cluster = std::uniform_int_distribution<size_t>(0, options.num_clusters - 1);
    auto jitter = std::normal_distribution<float>(0.0F, 0.35F);
    auto datum = std::vector<float>(options.dims);
    for (size_t i = 0; i < options.num_vectors; ++i) {
        const size_t cluster = pick_cluster(rng);
        const float* center = centers.data() + cluster * options.dims;
        for (size_t d = 0; d < options.dims; ++d) {
            datum[d] = center[d] + jitter(rng);
        }
        data.set_datum(i, datum);
    }
    return data;
}

template <typename Distance> int run(const Options& options, Distance distance) {
    namespace vamana = svs::index::vamana;

    // The whole point of the reproducer: guarantee we are on the iterative overload.
    static_assert(
        std::is_same_v<vamana::prune_strategy_t<Distance>, vamana::IterativePruneStrategy>,
        "This reproducer must exercise the IterativePruneStrategy overload of "
        "heuristic_prune_neighbors!"
    );

    const size_t num_threads =
        options.num_threads == 0
            ? std::max<size_t>(1, std::thread::hardware_concurrency())
            : options.num_threads;

    auto timer = svs::lib::Timer();
    auto data = [&]() {
        auto handle = timer.push_back("dataset");
        if (options.data_path.empty()) {
            fmt::print(
                "Generating {} x {} synthetic vectors in {} clusters (seed {})\n",
                options.num_vectors,
                options.dims,
                options.num_clusters,
                options.seed
            );
            return generate_dataset(options);
        }
        fmt::print("Loading vectors from {}\n", options.data_path);
        return svs::load_data<Eltype>(options.data_path);
    }();

    fmt::print(
        "Dataset: {} vectors, {} dimensions\n"
        "Distance: {} -> IterativePruneStrategy\n"
        "Building with {} threads...\n",
        data.size(),
        data.dimensions(),
        options.distance,
        num_threads
    );

    auto parameters = vamana::VamanaBuildParameters{
        options.alpha,
        options.graph_max_degree,
        options.window_size,
        options.max_candidate_pool_size,
        options.prune_to,
        options.full_search_history};

    auto handle = timer.push_back("index build");
    auto index =
        vamana::auto_build(parameters, SVS_LAZY(std::move(data)), distance, num_threads);
    const double elapsed = svs::lib::as_seconds(handle.finish());

    // Echo what the build settled on: `alpha` and `prune_to` are defaulted per-distance
    // inside `verify_and_set_default_index_parameters`, and both feed the prune call sites.
    fmt::print(
        "Build completed in {:.3f} seconds\n"
        "Effective parameters: alpha = {}, graph_max_degree = {}, window_size = {}, "
        "max_candidate_pool_size = {}, prune_to = {}, use_full_search_history = {}\n",
        elapsed,
        index.get_alpha(),
        index.get_graph_max_degree(),
        index.get_construction_window_size(),
        index.get_max_candidates(),
        index.get_prune_to(),
        index.get_full_search_history()
    );

    fmt::print("\n{}\n", timer);
    return 0;
}

} // namespace

int svs_main(std::vector<std::string> args) {
    auto options = parse_options(args);
    if (options.verbose) {
        auto logger = svs::logging::get();
        svs::logging::set_level(logger, svs::logging::Level::Trace);
    }

    if (options.distance == "mip") {
        return run(options, svs::distance::DistanceIP());
    }
    if (options.distance == "cosine") {
        return run(options, svs::distance::DistanceCosineSimilarity());
    }
    throw ANNEXCEPTION(
        "Unrecognized distance \"{}\". Expected mip or cosine -- these are the distances "
        "that select IterativePruneStrategy.",
        options.distance
    );
}

SVS_DEFINE_MAIN();
