// Architecture-neutral packed CKKS executor. Evaluation has no decryption API.
#include <fideslib.hpp>
#include "packed_program.hpp"
#include "packed_routing.hpp"
#include "packed_depth.hpp"
#include "packed_lifetime.hpp"
#include "packed_schedule.hpp"
#include "fideslib_rotation_batch.hpp"
#include "chebyshev_basis_cache.hpp"
#include "fideslib_owned_arithmetic.hpp"
#include "plaintext_cache.hpp"
#include "fideslib_plaintext_ops.hpp"
#include "fideslib_plaintext_encoder.hpp"
#include "stage1_mamba2_plan.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <cstdint>
#include <sys/resource.h>

extern "C" int cudaDeviceSynchronize(void);
extern "C" int cudaProfilerStart(void);
extern "C" int cudaProfilerStop(void);
using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;
using Clock = std::chrono::steady_clock;
static void sync_gpu() {
  if (cudaDeviceSynchronize() != 0) throw std::runtime_error("CUDA synchronization failed");
}
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct PackedEvaluator {
  CryptoContext<DCRTPoly> cc;
  PublicKey<DCRTPoly> public_key;
  int slots;
  double bound;
  long long ct_ct = 0, ct_pt = 0, rotations = 0, adds = 0, bootstraps = 0, reencodes = 0;
  double bootstrap_seconds = 0;
  bool replicated_linear = true;
  bool legacy_routing = false;
  bool trace_levels = false;
  bool planned_refresh = false;
  bool batch_refresh = false;
  int bootstrap_passes = 2;
  int refresh_ceiling = 39, refreshed_level = 22;
  long long logical_refreshes = 0, refresh_batches = 0, largest_refresh_batch = 1;
  long long evaluated_nodes = 0;
  int planned_logical_refreshes = 0;
  bool profile_evaluation = false;
  bool inplace_ops = false;
  bool naf_rotations = false, reuse_dead_inputs = false;
  bool bsgs_routing_stages = false;
  bool frontier_refresh = false;
  bool hoist_rotations = false, share_chebyshev = false;
  fhemamba::RotationBatchStats rotation_batch_stats;
  long long shared_basis_hits = 0, shared_basis_invalidations = 0;
  std::map<int, fhemamba::ChebyshevBasisState<Ct>> shared_bases;
  long long frontier_deferrals = 0;
  std::size_t maximum_ready_nodes = 0;
  long long optimized_routing_stages = 0, routing_stage_rotations_saved = 0;
  long long scratch_clones_eliminated = 0;
  fhemamba::OwnedArithmeticStats owned_arithmetic;
  fhemamba::OwnedArithmeticStats square_arithmetic;
  long long lifetime_clones_eliminated = 0, refresh_rotations = 0;
  double host_encoding_seconds = 0, mask_preparation_seconds = 0;
  long long host_encodes = 0;
  bool cache_plaintexts = false;
  fhemamba::MaskPlaintextCache<Plaintext> plaintext_cache{64};
  std::unique_ptr<fhemamba::PlaintextPreparation> plaintexts;
  struct OperationStats { double seconds = 0, bootstrap_seconds = 0; long long nodes = 0, bootstraps = 0; };
  std::map<std::string, OperationStats> operation_stats;
  std::function<Ct(const Ct&)> client_feedback;
  double current_bound = 64;

  auto scale(const Ct& a, double c) -> Ct {
    ++ct_pt; auto out = a->Clone(); cc->EvalMultInPlace(out, c); sync_gpu(); return out;
  }
  auto aligned(const Ct& a, const Ct& b) -> std::pair<Ct, Ct> {
    auto x = a->Clone(), y = b->Clone();
    align_owned(x, y); return {x, y};
  }
  void align_owned(Ct& x, Ct& y) {
    auto level = std::max(x->GetLevel(), y->GetLevel());
    x->SetLevel(level); y->SetLevel(level);
  }
  auto binary_owned(Ct x, Ct y, bool multiply) -> Ct {
    align_owned(x, y);
    if (multiply) { ++ct_ct; cc->EvalMultMutableInPlace(x, y); }
    else { ++adds; cc->EvalAddInPlace(x, y); }
    sync_gpu(); ++scratch_clones_eliminated; return x;
  }
  auto add(const Ct& a, const Ct& b) -> Ct {
    auto [x, y] = aligned(a, b); ++adds;
    // aligned() already owns both scratch values. Reuse x instead of copying
    // it a third time inside the non-mutating API; input DAG values stay intact.
    if (inplace_ops) {
      cc->EvalAddInPlace(x, y); sync_gpu(); ++scratch_clones_eliminated; return x;
    }
    auto out = cc->EvalAdd(x, y); sync_gpu(); return out;
  }
  auto mul(const Ct& a, const Ct& b) -> Ct {
    if (a == b) return square(a);
    auto [x, y] = aligned(a, b); ++ct_ct;
    if (inplace_ops) {
      cc->EvalMultMutableInPlace(x, y); sync_gpu(); ++scratch_clones_eliminated; return x;
    }
    auto out = cc->EvalMult(x, y); sync_gpu(); return out;
  }
  auto square(Ct value) -> Ct {
    ++ct_ct;
    return fhemamba::owned_ciphertext_square(cc, std::move(value), sync_gpu, square_arithmetic);
  }
  auto add_temporaries(Ct a, Ct b) -> Ct {
    if (!inplace_ops) return add(a, b);
    ++adds; ++scratch_clones_eliminated;
    return fhemamba::owned_ciphertext_binary(cc, std::move(a), std::move(b),
        fhemamba::CiphertextBinaryOp::Add,
        [&](Ct& x, Ct& y) { align_owned(x, y); }, sync_gpu, owned_arithmetic);
  }
  auto scalar_add(const Ct& a, double b) -> Ct { ++adds; return cc->EvalAdd(a, b); }
  auto plain(const Ct& a, std::vector<double> values, bool multiply) -> Ct {
    values.resize(slots);
    auto encode = [&] {
      const auto start = profile_evaluation ? Clock::now() : Clock::time_point{};
      auto p = multiply ? plaintexts->encode(values, a->GetLevel())
                        : plaintexts->encode_addend(a, values, reencodes);
      ++host_encodes;
      if (profile_evaluation) host_encoding_seconds += elapsed(start);
      return p;
    };
    // multPt reads its plaintext; any level alignment uses a private copy in
    // FIDESlib. Additive degree correction stays on the original uncached path.
    auto p = multiply && cache_plaintexts ? plaintext_cache.get(values, a->GetLevel(), encode) : encode();
    plaintexts->load(p);
    if (multiply) { ++ct_pt; auto out = cc->EvalMult(a, p); sync_gpu(); return out; }
    ++adds; auto out = cc->EvalAdd(a, p); sync_gpu(); return out;
  }
  auto encrypt(std::vector<double> values) -> Ct {
    values.resize(slots);
    auto p = cc->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, slots);
    return cc->Encrypt(public_key, p);
  }
  auto rotate(const Ct& value, int offset) -> Ct {
    auto out = value;
    bool owns_output = false;
    for (const int step : fhemamba::rotation_steps(offset, slots, naf_rotations)) {
        ++rotations;
        if (inplace_ops) {
          if (!owns_output) { out = value->Clone(); owns_output = true; }
          else ++scratch_clones_eliminated;
          cc->EvalRotateInPlace(out, step); sync_gpu();
        } else {
          auto next = cc->EvalRotate(out, step); sync_gpu(); out = next;
        }
    }
    return out;
  }
  auto rotate_many(const Ct& input, const std::vector<int>& offsets) -> std::vector<Ct> {
    if (!hoist_rotations) {
      std::vector<Ct> out;
      for (int offset : offsets) out.push_back(rotate(input, offset));
      return out;
    }
    const auto before = rotation_batch_stats.edges;
    auto out = fhemamba::hoisted_rotation_batch(cc, input, offsets, slots,
        naf_rotations, sync_gpu, rotation_batch_stats);
    rotations += rotation_batch_stats.edges - before;
    return out;
  }
  auto bootstrap_components(Ct input) -> std::pair<Ct, Ct> {
    while (input->GetNoiseScaleDeg() > 1) cc->RescaleInPlace(input);
    sync_gpu(); auto first = cc->EvalBootstrap(input); sync_gpu(); ++bootstraps;
    if (bootstrap_passes == 1) return {first, {}};
    auto a = input->Clone(), b = first->Clone();
    while (b->GetNoiseScaleDeg() > 1) cc->RescaleInPlace(b);
    const auto target = std::max(a->GetLevel(), b->GetLevel());
    for (auto* v : {&a, &b}) {
      while ((*v)->GetLevel() < target) {
        *v = scale(*v, 1.0);
        while ((*v)->GetNoiseScaleDeg() > 1) cc->RescaleInPlace(*v);
      }
    }
    auto residual = cc->EvalSub(a, b); ++adds;
    residual = scale(residual, 4096.0);
    while (residual->GetNoiseScaleDeg() > 1) cc->RescaleInPlace(residual);
    sync_gpu(); auto second = cc->EvalBootstrap(residual); sync_gpu(); ++bootstraps;
    return {first, second};
  }
  auto refresh(const Ct& value, double refresh_bound) -> Ct {
    auto start = Clock::now();
    const auto prior_rotations = rotations;
    ++logical_refreshes;
    auto [first, second] = bootstrap_components(scale(value, 1.0 / refresh_bound));
    auto out = !second ? scale(first, refresh_bound) : planned_refresh
        ? add(scale(first, refresh_bound), scale(second, refresh_bound / 4096.0))
        : scale(add(first, scale(second, 1.0 / 4096.0)), refresh_bound);
    refresh_rotations += rotations - prior_rotations;
    bootstrap_seconds += elapsed(start); return out;
  }
  void refresh_group(const std::vector<int>& indices, std::vector<Ct>& values,
                     const fhemamba::PackedProgram& program) {
    auto start = Clock::now();
    const auto prior_rotations = rotations;
    Ct packed;
    int offset = 0;
    for (int index : indices) {
      const auto& node = program.nodes[index];
      std::vector<double> mask(slots);
      std::fill_n(mask.begin(), node.size, 1.0 / node.bound);
      auto term = rotate(plain(values[index], std::move(mask), true), -offset);
      packed = packed ? add(packed, term) : term;
      offset += node.size;
    }
    auto [first, second] = bootstrap_components(packed);
    offset = 0;
    for (int index : indices) {
      const auto& node = program.nodes[index];
      std::vector<double> mask(slots);
      std::fill_n(mask.begin(), node.size, node.bound);
      auto a = plain(rotate(first, offset), mask, true);
      for (auto& value : mask) value /= 4096.0;
      auto b = plain(rotate(second, offset), std::move(mask), true);
      values[index] = add(a, b);
      offset += node.size;
    }
    ++refresh_batches;
    logical_refreshes += indices.size();
    largest_refresh_batch = std::max(largest_refresh_batch, static_cast<long long>(indices.size()));
    refresh_rotations += rotations - prior_rotations;
    bootstrap_seconds += elapsed(start);
    if (trace_levels) {
      std::cout << "refresh_batch nodes=";
      for (int index : indices) std::cout << index << ',';
      std::cout << " occupied_slots=" << offset << std::endl;
    }
  }
  auto transform(const Ct& x, const std::map<int, std::vector<double>>& masks) -> Ct {
    Ct out;
    if (planned_refresh) {
      std::vector<int> offsets;
      for (const auto& [offset, mask] : masks) offsets.push_back(offset);
      const auto plan = fhemamba::plan_packed_diagonals(std::move(offsets), slots, naf_rotations);
      if (plan.baby_step) {
        const int step = plan.offsets[1] - plan.offsets[0];
        std::vector<int> baby_offsets(plan.offsets.begin(), plan.offsets.begin() + plan.baby_step);
        auto babies = rotate_many(x, baby_offsets);
        for (int first = 0; first < static_cast<int>(plan.offsets.size()); first += plan.baby_step) {
          const int giant = first * step;
          Ct inner;
          for (int j = 0; j < plan.baby_step && first + j < static_cast<int>(plan.offsets.size()); ++j) {
            const auto preparation = profile_evaluation ? Clock::now() : Clock::time_point{};
            auto mask = fhemamba::packed_diagonal_pre_mask(masks.at(plan.offsets[first + j]), giant);
            if (profile_evaluation) mask_preparation_seconds += elapsed(preparation);
            auto term = plain(babies[j], std::move(mask), true);
            inner = inner ? add_temporaries(std::move(inner), std::move(term)) : std::move(term);
          }
          auto term = rotate(inner, giant);
          out = out ? add_temporaries(std::move(out), std::move(term)) : std::move(term);
        }
        sync_gpu();
        return out;
      }
    }
    for (const auto& [offset, mask] : masks) {
      auto term = plain(rotate(x, offset), mask, true);
      out = out ? add_temporaries(std::move(out), std::move(term)) : std::move(term);
    }
    return out ? out : scale(x, 0.0);
  }
  auto gather(const Ct& x, const std::vector<double>& indices, int size, bool scatter) -> Ct {
    if (!legacy_routing && replicated_linear) {
      bool monotone = true, identity = true;
      std::set<int> offsets;
      for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        monotone = monotone && (!i || indices[i] > indices[i - 1]);
        identity = identity && indices[i] == i;
        offsets.insert(static_cast<int>(indices[i]) - i);
      }
      // Scatter sources have clean padding at node boundaries. Gather sources
      // may be a sliding reduction, so they still require a selection mask.
      if (scatter && identity) return x;
      if (monotone && offsets.size() > 32) {
        Ct out = x;
        for (const auto& stage : fhemamba::monotone_routing(indices, scatter)) {
          if (out->GetLevel() + (planned_refresh ? 1 : 3) > refresh_ceiling) out = refresh(out, current_bound);
          if (bsgs_routing_stages) {
            std::vector<int> offsets;
            int direct_cost = 0;
            for (const auto& [offset, positions] : stage) {
              offsets.push_back(offset);
              direct_cost += fhemamba::packed_rotation_cost(offset, slots, naf_rotations);
            }
            const auto plan = fhemamba::plan_packed_diagonals(std::move(offsets), slots, naf_rotations);
            if (plan.baby_step && plan.rotations < direct_cost) {
              out = transform(out, fhemamba::packed_routing_masks(stage, slots));
              ++optimized_routing_stages;
              routing_stage_rotations_saved += direct_cost - plan.rotations;
              continue;
            }
          }
          Ct next;
          for (const auto& [offset, positions] : stage) {
            std::vector<double> mask(slots);
            for (int position : positions) mask[position] = 1;
            auto term = rotate(plain(out, std::move(mask), true), offset);
            next = next ? add_temporaries(std::move(next), std::move(term)) : std::move(term);
          }
          out = next;
        }
        return out;
      }
    }
    std::map<int, std::vector<double>> masks;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      const int source = scatter ? i : static_cast<int>(indices[i]);
      const int destination = scatter ? static_cast<int>(indices[i]) : i;
      auto& mask = masks[source - destination];
      if (mask.empty()) mask.resize(slots);
      mask[destination] = 1;
    }
    // Monotone compaction/expansion (slices, strides and head reductions).
    // Route each selected coordinate by the bits of its displacement. For
    // compaction, ascending bits keep positions strictly ordered; expansion
    // uses descending bits. This replaces thousands of diagonals by log(slots)
    // masked rotations and never merges two live coordinates.
    bool monotone = true;
    for (std::size_t i = 1; i < indices.size(); ++i)
      monotone = monotone && indices[i] > indices[i - 1];
    if (legacy_routing && replicated_linear && monotone && masks.size() > 16) {
      std::vector<int> positions(indices.size()), distance(indices.size());
      std::vector<double> selected(slots);
      int maximum = 0;
      for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        positions[i] = scatter ? i : static_cast<int>(indices[i]);
        distance[i] = static_cast<int>(indices[i]) - i;
        selected[positions[i]] = 1;
        maximum = std::max(maximum, distance[i]);
      }
      Ct out = plain(x, std::move(selected), true);
      std::vector<int> shifts;
      for (int shift = 1; shift <= maximum; shift *= 2) shifts.push_back(shift);
      if (scatter) std::reverse(shifts.begin(), shifts.end());
      for (int shift : shifts) {
        std::vector<double> moving_mask(slots);
        bool any = false;
        for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
          if (distance[i] & shift) {
            any = true; moving_mask[positions[i]] = 1;
            positions[i] += scatter ? shift : -shift;
          }
        }
        if (!any) continue;
        if (out->GetLevel() + 3 > refresh_ceiling) out = refresh(out, current_bound);
        auto moving = plain(out, std::move(moving_mask), true);
        auto [a, b] = aligned(out, moving);
        ++adds;
        out = add(cc->EvalSub(a, b), rotate(moving, scatter ? -shift : shift));
      }
      return out;
    }
    return transform(x, masks);
  }
  auto linear(const Ct& x, const fhemamba::PackedNode& node, int columns, bool mask_output = true) -> Ct {
    if (replicated_linear && node.size + columns <= slots) {
      using namespace fhemamba::stage1;
      auto shape = resolve_interleaved_replicated_shape(node.size, columns, slots, 0);
      if (shape.replicas > 1) {
      shape.logarithmic_replication = true;
      // The shared helper reserves baby_step=1 for direct-diagonal masks.
      shape.baby_step = std::max(2, static_cast<int>(std::sqrt(shape.per_replica)));
      auto rot = [&](const Ct& a, int shift) { return rotate(a, shift); };
      auto sum = [&](const Ct& a, const Ct& b) { return add(a, b); };
      auto extended = rotation_sum(x, shape.reps, -columns, true, rot, sum);
      auto replicated = rotation_sum(extended, shape.replicas + shape.guard_windows,
                                     -shape.window, true, rot, sum);
      std::vector<int> baby_offsets;
      for (int i = 0; i < shape.baby_step; ++i) baby_offsets.push_back(i * shape.replicas);
      auto babies = rotate_many(replicated, baby_offsets);
      Ct out;
      for (int first = 0; first < shape.per_replica; first += shape.baby_step) {
        Ct inner;
        for (int j = 0; j < shape.baby_step && first + j < shape.per_replica; ++j) {
          const auto preparation = profile_evaluation ? Clock::now() : Clock::time_point{};
          auto mask = replicated_bsgs_pre_mask(node.weights(), node.size, columns,
                                               first + j, shape, slots, 0.0);
          if (profile_evaluation) mask_preparation_seconds += elapsed(preparation);
          auto term = plain(babies[j], std::move(mask), true);
          inner = inner ? add_temporaries(std::move(inner), std::move(term)) : std::move(term);
        }
        auto term = rotate(inner, first * shape.replicas);
        out = out ? add_temporaries(std::move(out), std::move(term)) : std::move(term);
      }
      out = rotation_sum(out, shape.replicas, shape.window + 1, true, rot, sum);
      if (!mask_output) { sync_gpu(); return out; }
      std::vector<double> mask(slots);
      std::fill_n(mask.begin(), node.size, 1.0);
      auto result = plain(out, std::move(mask), true);
      sync_gpu();
      return result;
      }
    }
    std::map<int, std::vector<double>> masks;
    for (int i = 0; i < node.size; ++i) for (int j = 0; j < columns; ++j) {
      const auto weight = node.weights()[i * columns + j];
      if (weight == 0) continue;
      auto& mask = masks[j - i];
      if (mask.empty()) mask.resize(slots);
      mask[i] = weight;
    }
    return transform(x, masks);
  }
  auto chebyshev(const Ct& x, const fhemamba::PackedNode& node, int shared_group = -1) -> Ct {
    const auto lo = node.data[0], hi = node.data[1];
    std::vector<double> affine(slots), bias(slots), mask(slots);
    for (int i = 0; i < node.size; ++i) {
      affine[i] = 2 / (hi - lo); bias[i] = -(lo + hi) / (hi - lo); mask[i] = 1;
    }
    // Zero padded normalized arguments keep every inactive slot in [-1,1].
    fhemamba::ChebyshevBasisState<Ct> local_basis;
    auto& state = shared_group < 0 ? local_basis : shared_bases[shared_group];
    if (!state.matches(x)) {
      if (state.normalized) ++shared_basis_invalidations;
      state.reset(x);
      state.normalized = plain(plain(x, affine, true), bias, false);
      state.basis.emplace(1, state.normalized);
    } else ++shared_basis_hits;
    const auto& u = state.normalized;
    std::vector<double> coefficients(node.data.begin() + 2, node.data.end());
    double at_zero = 0;
    if (planned_refresh) {
      // P(0) is public. P(u)-P(0) has zero inactive slots because the affine
      // map sets u=0 there. Add P(0) only to active slots, saving a mask level.
      for (int i = 0; i < static_cast<int>(coefficients.size()); i += 2)
        at_zero += (i % 4 ? -1 : 1) * coefficients[i];
      coefficients[0] -= at_zero;
    }
    int levels = 0;
    while ((1 << levels) < static_cast<int>(coefficients.size())) ++levels;
    const int baby = 1 << ((std::max(1, levels) + 1) / 2);
    auto& cache = state.basis;
    std::function<Ct(int)> basis = [&](int i) -> Ct {
      if (cache.count(i)) return cache[i];
      Ct out;
      if (i % 2 == 0) {
        auto square = mul(basis(i / 2), basis(i / 2));
        out = scalar_add(add(square, square), -1);
      } else {
        auto product = mul(basis((i + 1) / 2), basis(i / 2));
        auto [a, b] = aligned(add(product, product), u);
        out = cc->EvalSub(a, b); ++adds;
      }
      cache[i] = out; return out;
    };
    std::function<Ct(std::vector<double>)> evaluate = [&](std::vector<double> c) -> Ct {
      const int degree = static_cast<int>(c.size()) - 1;
      if (degree < baby) {
        Ct out;
        for (int i = 1; i <= degree; ++i) {
          if (std::abs(c[i]) < 1e-14) continue;
          auto term = scale(basis(i), c[i]);
          out = out ? add_temporaries(std::move(out), std::move(term)) : std::move(term);
        }
        if (!out) out = scale(u, 0.0);
        return scalar_add(out, c[0]);
      }
      int k = baby;
      while (2 * k - 1 < degree) k *= 2;
      std::vector<double> upper(c.begin() + k, c.end()), lower(c.begin(), c.begin() + k);
      for (int i = 1; i < static_cast<int>(upper.size()); ++i) upper[i] *= 2;
      for (int i = k + 1; i <= degree; ++i) lower[2 * k - i] -= c[i];
      return add_temporaries(evaluate(lower), mul(basis(k), evaluate(upper)));
    };
    if (planned_refresh) {
      for (auto& value : mask) value *= at_zero;
      return plain(evaluate(coefficients), mask, false);
    }
    return plain(evaluate(coefficients), mask, true);
  }
  auto evaluate(const fhemamba::PackedProgram& program) -> std::vector<Ct> {
    fhemamba::PackedDepthPlan depth_plan;
    if (planned_refresh) {
      if (legacy_routing || !replicated_linear)
        throw std::invalid_argument("planned refresh requires replicated linear and radix8 routing");
      depth_plan = fhemamba::plan_packed_depth(program, refresh_ceiling, refreshed_level);
      if (frontier_refresh) {
        // Advance independent branches to a common refresh frontier. Actual
        // ciphertext levels below remain authoritative for refresh decisions.
        std::fill(depth_plan.refresh_after.begin(), depth_plan.refresh_after.end(), false);
        depth_plan.refreshes = fhemamba::simulate_packed_depth(program, depth_plan);
      }
      planned_logical_refreshes = depth_plan.refreshes;
      std::cout << "planned_logical_refreshes=" << depth_plan.refreshes << std::endl;
    }
    auto sharing = fhemamba::plan_chebyshev_sharing(program, depth_plan.live);
    shared_bases.clear();
    std::vector<Ct> values(program.nodes.size());
    std::vector<bool> dirty(program.nodes.size());
    auto clean = [&](int index) {
      if (!dirty[index]) return;
      std::vector<double> mask(slots);
      std::fill_n(mask.begin(), program.nodes[index].size, 1);
      values[index] = plain(values[index], std::move(mask), true);
      dirty[index] = false;
    };
    const auto uses = fhemamba::plan_packed_uses(program, depth_plan.live);
    const auto& last = uses.last;
    const auto& consumers = uses.consumers;
    std::unique_ptr<fhemamba::PackedReadySchedule> schedule;
    if (frontier_refresh) schedule = std::make_unique<fhemamba::PackedReadySchedule>(program, depth_plan.live);
    auto refresh_value = [&](int requested, int next_operation) {
      std::vector<int> group{requested};
      int occupied = program.nodes[requested].size;
      if (batch_refresh) for (int j = 0; j < static_cast<int>(values.size()) && group.size() < 16; ++j) {
        if (j == requested || !values[j] || occupied + program.nodes[j].size > slots) continue;
        int next = -1;
        if (schedule) next = schedule->next_consumer(j);
        else {
          const auto use = std::lower_bound(consumers[j].begin(), consumers[j].end(), next_operation);
          if (use != consumers[j].end()) next = *use;
        }
        if (next < 0 || program.nodes[next].operation == "feedback") continue;
        const int level = values[j]->GetLevel();
        if (level <= depth_plan.refreshed || (level < refresh_ceiling - 6 && level + depth_plan.cost[next] <= refresh_ceiling)) continue;
        group.push_back(j); occupied += program.nodes[j].size;
      }
      if (group.size() > 1) {
        refresh_group(group, values, program);
        for (int index : group) dirty[index] = false;
      } else {
        clean(requested); // Never bootstrap unmasked replica/guard coordinates.
        values[requested] = refresh(values[requested], program.nodes[requested].bound);
      }
    };
    const int live_nodes = planned_refresh
        ? std::count(depth_plan.live.begin(), depth_plan.live.end(), true)
        : program.nodes.size();
    const auto start = Clock::now();
    int sequential = 0, completed = 0;
    while (schedule ? !schedule->empty() : sequential < static_cast<int>(program.nodes.size())) {
      int i = sequential++;
      if (schedule) {
        maximum_ready_nodes = std::max(maximum_ready_nodes, schedule->ready().size());
        i = *schedule->ready().begin();
        for (int ready : schedule->ready()) {
          const auto& candidate = program.nodes[ready];
          bool fits = true;
          if (candidate.operation != "feedback") for (int parent : candidate.parents)
            fits = fits && values[parent]->GetLevel() + depth_plan.cost[ready] <= refresh_ceiling;
          if (fits) {
            if (ready != i) ++frontier_deferrals;
            i = ready; break;
          }
        }
      } else if (planned_refresh && !depth_plan.live[i]) continue;
      const auto& n = program.nodes[i];
      const auto operation_start = Clock::now();
      const auto prior_bootstrap_seconds = bootstrap_seconds;
      const auto prior_bootstraps = bootstraps;
      current_bound = n.bound;
      const auto& op = n.operation;
      int need = 3;
      if (op == "cheb") {
        int log = 0;
        while ((1 << log) < static_cast<int>(n.data.size()) - 2) ++log;
        need = log + 5;
      }
      if (planned_refresh) need = depth_plan.cost[i];
      for (int parent : n.parents) {
        if ((!planned_refresh || op != "feedback") && values[parent]->GetLevel() + need > refresh_ceiling) {
          refresh_value(parent, i);
        }
      }
      // Do not create an extra handle before checking whether a dying binary
      // input is uniquely owned. Outputs stay pinned by last[] above.
      const bool consume_binary = reuse_dead_inputs && (op == "add" || op == "mul");
      Ct x = n.parents.empty() || consume_binary ? Ct{} : values[n.parents[0]];
      if (trace_levels) {
        std::cout << "level_input node=" << i << " op=" << op << " bound=" << n.bound << " parents=";
        for (int parent : n.parents) std::cout << parent << ':' << values[parent]->GetLevel()
            << ':' << values[parent]->GetNoiseScaleDeg() << ',';
        std::cout << " refreshed=" << bootstraps - prior_bootstraps << std::endl;
      }
      Ct out;
      if (op == "input" || op == "public") out = encrypt(n.data);
      else if (op == "feedback") {
        if (!client_feedback) throw std::runtime_error("client feedback requires --client-head");
        out = client_feedback(x);
      }
      else if (consume_binary) {
        const int a = n.parents[0], b = n.parents[1];
        auto take = [&](int parent) {
          return fhemamba::consume_or_clone(values[parent], schedule ? schedule->final_use(parent, i) : last[parent] == i,
              [](const Ct& v) { return v->Clone(); }, lifetime_clones_eliminated);
        };
        if (op == "mul" && a == b) out = square(take(a));
        else {
          // Binary level alignment may mutate both buffers, including doubles.
          Ct right = a == b ? values[b]->Clone() : take(b);
          out = binary_owned(take(a), std::move(right), op == "mul");
        }
      }
      else if (op == "add") out = add(x, values[n.parents[1]]);
      else if (op == "mul") out = mul(x, values[n.parents[1]]);
      else if (planned_refresh && fhemamba::packed_negation(n)) {
        auto zero = cc->EvalSub(x, x); sync_gpu();
        out = cc->EvalSub(zero, x); adds += 2; sync_gpu();
      }
      else if (op == "addp" || op == "mulp") out = plain(x, n.data, op == "mulp");
      else if (op == "gather" || op == "scatter") out = gather(x, n.data, n.size, op == "scatter");
      else if (op == "linear" || op == "linear_ref") {
        const auto& weight = op == "linear" ? n : program.nodes[static_cast<int>(n.data[0])];
        dirty[i] = planned_refresh && depth_plan.defer_linear_mask[i];
        out = linear(x, weight, program.nodes[n.parents[0]].size, !dirty[i]);
      }
      else if (op == "cheb") {
        const int group = share_chebyshev ? sharing.group[i] : -1;
        out = chebyshev(x, n, group);
        if (group >= 0 && !--sharing.remaining[group]) shared_bases.erase(group);
      }
      else if (op == "repeat") {
        const int outer = n.data[0], inner = n.data[1], repeat = n.data[2];
        std::vector<double> destinations(outer * inner);
        for (int group = 0; group < outer; ++group)
          for (int j = 0; j < inner; ++j) destinations[group * inner + j] = group * inner * repeat + j;
        out = gather(x, destinations, n.size, true);
        out = fhemamba::stage1::rotation_sum(out, repeat, -inner, true,
            [&](const Ct& a, int shift) { return rotate(a, shift); },
            [&](const Ct& a, const Ct& b) { return add(a, b); });
      }
      else if (op == "sum") {
        out = x;
        const int width = static_cast<int>(n.data[0]);
        for (int step = 1; step < width; step *= 2) out = add(out, rotate(out, step));
        std::vector<double> indices(n.size);
        for (int j = 0; j < n.size; ++j) indices[j] = j * width;
        out = gather(out, indices, n.size, false);
      }
      values[i] = out;
      if (planned_refresh && depth_plan.refresh_after[i] && out->GetLevel() > depth_plan.refreshed) {
        refresh_value(i, i + 1);
        out = values[i];
      }
      sync_gpu();
      if (trace_levels) std::cout << "level_output node=" << i << " level=" << out->GetLevel()
          << " degree=" << out->GetNoiseScaleDeg() << " bootstraps=" << bootstraps - prior_bootstraps << std::endl;
      if (schedule) schedule->complete(i);
      for (int parent : n.parents)
        if (schedule ? schedule->releasable(parent) : last[parent] == i) values[parent].reset();
      ++completed;
      auto& stats = operation_stats[op];
      ++stats.nodes;
      ++evaluated_nodes;
      stats.seconds += elapsed(operation_start);
      stats.bootstrap_seconds += bootstrap_seconds - prior_bootstrap_seconds;
      stats.bootstraps += bootstraps - prior_bootstraps;
      if (completed % 10 == 0 || completed == live_nodes) {
        const auto seconds = elapsed(start);
        std::cout << "node=" << completed << '/' << live_nodes << " op=" << op
                  << " seconds=" << seconds << " eta_seconds="
                  << seconds * (live_nodes - completed) / completed
                  << " bootstraps=" << bootstraps << std::endl;
      }
    }
    std::vector<Ct> outputs;
    for (const auto& output : program.outputs) outputs.push_back(values[output.node]);
    return outputs;
  }
};

// Protocol client: the evaluator receives only an encrypted-embedding callback.
// Public BF16 checkpoint weights are exported losslessly as little-endian FP32.
struct GenerationClient {
  uint32_t vocabulary = 0, width = 0;
  std::vector<float> weights;
  std::vector<int> tokens;
  std::vector<double> margins;
  void load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&vocabulary), sizeof(vocabulary));
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    if (!in || vocabulary < 2 || vocabulary > 262144 || width < 1 || width > 8192 ||
        static_cast<uint64_t>(vocabulary) * width > 268435456)
      throw std::invalid_argument("invalid client vocabulary head");
    weights.resize(static_cast<std::size_t>(vocabulary) * width);
    in.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
    if (!in || in.peek() != std::char_traits<char>::eof())
      throw std::invalid_argument("invalid client head length");
    for (float v : weights) if (!std::isfinite(v))
      throw std::invalid_argument("nonfinite client head");
  }
  auto select(const std::vector<double>& hidden) -> std::vector<double> {
    if (hidden.size() < width) throw std::invalid_argument("client hidden width mismatch");
    for (uint32_t j = 0; j < width; ++j)
      if (!std::isfinite(hidden[j])) throw std::runtime_error("nonfinite client output");
    int best = -1;
    double first = -INFINITY, second = -INFINITY;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      double logit = 0;
      for (uint32_t j = 0; j < width; ++j)
        logit += hidden[j] * weights[static_cast<std::size_t>(token) * width + j];
      if (logit > first) { second = first; first = logit; best = token; }
      else if (logit > second) second = logit;
    }
    tokens.push_back(best); margins.push_back(first - second);
    std::cout << "client_token=" << best << " logit_margin=" << first - second << std::endl;
    const auto begin = weights.begin() + static_cast<std::size_t>(best) * width;
    return {begin, begin + width};
  }
};

auto main(int argc, char** argv) -> int {
  try {
    if (argc < 5) throw std::invalid_argument("usage: packed_fideslib PROGRAM RESULT POLY_TOL EXACT_TOL [--direct-linear] [--legacy-routing] [--planned-refresh] [--batch-refresh] [--bootstrap-passes 1|2] [--trace-levels] [--profile-evaluation] [--inplace-ops] [--naf-rotations] [--reuse-dead-inputs] [--compact-weights] [--cache-plaintexts] [--fast-plaintext-upload] [--direct-plaintext-upload] [--gpu-plaintext-ntt] [--move-plaintext-coefficients] [--borrow-plaintext-upload] [--bsgs-routing-stages] [--frontier-refresh] [--s2c-first] [--gpu-plaintext-rns] [--hoist-rotations] [--share-chebyshev] [--client-head FILE]");
    bool replicated_linear = true;
    bool legacy_routing = false;
    bool trace_levels = false;
    bool planned_refresh = false;
    bool batch_refresh = false;
    bool profile_evaluation = false;
    bool inplace_ops = false;
    bool naf_rotations = false, reuse_dead_inputs = false;
    bool compact_weights = false;
    bool cache_plaintexts = false;
    bool fast_plaintext_upload = false, gpu_plaintext_ntt = false;
    bool direct_plaintext_upload = false;
    bool move_plaintext_coefficients = false;
    bool borrow_plaintext_upload = false;
    bool bsgs_routing_stages = false;
    bool frontier_refresh = false;
    bool s2c_first = false;
    bool gpu_plaintext_rns = false;
    bool hoist_rotations = false, share_chebyshev = false;
    int bootstrap_passes = 2;
    std::string client_path;
    for (int i = 5; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--direct-linear") replicated_linear = false;
      else if (option == "--legacy-routing") legacy_routing = true;
      else if (option == "--trace-levels") trace_levels = true;
      else if (option == "--planned-refresh") planned_refresh = true;
      else if (option == "--batch-refresh") batch_refresh = true;
      else if (option == "--profile-evaluation") profile_evaluation = true;
      else if (option == "--inplace-ops") inplace_ops = true;
      else if (option == "--naf-rotations") naf_rotations = true;
      else if (option == "--reuse-dead-inputs") reuse_dead_inputs = true;
      else if (option == "--compact-weights") compact_weights = true;
      else if (option == "--cache-plaintexts") cache_plaintexts = true;
      else if (option == "--fast-plaintext-upload") fast_plaintext_upload = true;
      else if (option == "--direct-plaintext-upload") direct_plaintext_upload = true;
      else if (option == "--gpu-plaintext-ntt") gpu_plaintext_ntt = true;
      else if (option == "--move-plaintext-coefficients") move_plaintext_coefficients = true;
      else if (option == "--borrow-plaintext-upload") borrow_plaintext_upload = true;
      else if (option == "--bsgs-routing-stages") bsgs_routing_stages = true;
      else if (option == "--frontier-refresh") frontier_refresh = true;
      else if (option == "--s2c-first") s2c_first = true;
      else if (option == "--gpu-plaintext-rns") gpu_plaintext_rns = true;
      else if (option == "--hoist-rotations") hoist_rotations = true;
      else if (option == "--share-chebyshev") share_chebyshev = true;
      else if (option == "--bootstrap-passes" && i + 1 < argc) {
        bootstrap_passes = std::stoi(argv[++i]);
        if (bootstrap_passes != 1 && bootstrap_passes != 2)
          throw std::invalid_argument("bootstrap passes must be 1 or 2");
      }
      else if (option == "--client-head" && i + 1 < argc) client_path = argv[++i];
      else throw std::invalid_argument("unknown packed option: " + option);
    }
    const double poly_tol = std::stod(argv[3]), exact_tol = std::stod(argv[4]);
    gpu_plaintext_ntt |= move_plaintext_coefficients;
    direct_plaintext_upload |= borrow_plaintext_upload;
    if (gpu_plaintext_ntt || direct_plaintext_upload) fast_plaintext_upload = true;
    if (reuse_dead_inputs) inplace_ops = true;
    if (bsgs_routing_stages && (!planned_refresh || legacy_routing || !replicated_linear))
      throw std::invalid_argument("BSGS routing stages require planned refresh and radix8 routing");
    if (batch_refresh && (!planned_refresh || bootstrap_passes != 2))
      throw std::invalid_argument("batch refresh requires planned refresh and two bootstrap passes");
    if (s2c_first && !batch_refresh) throw std::invalid_argument("S2C-first requires planned two-pass batch refresh");
#ifndef FIDESLIB_S2C_FIRST_BOOTSTRAP
    if (s2c_first) throw std::invalid_argument("S2C-first requires the optional FIDESlib bootstrap patch");
#endif
#ifndef FHEMAMBA_GPU_PLAINTEXT_RNS
    if (gpu_plaintext_rns) throw std::invalid_argument("GPU plaintext RNS requires -DFHE_STAGE0_GPU_RNS=ON");
#endif
    if (frontier_refresh && !batch_refresh)
      throw std::invalid_argument("frontier refresh requires batch refresh");
    if (planned_refresh && (legacy_routing || !replicated_linear))
      throw std::invalid_argument("planned refresh requires replicated linear and radix8 routing");
    if (!std::isfinite(poly_tol) || !std::isfinite(exact_tol) || poly_tol <= 0 || exact_tol <= 0)
      throw std::invalid_argument("invalid error tolerance");
    std::ifstream input(argv[1]);
    const auto program = fhemamba::read_packed_program(input, compact_weights);
    std::size_t weight_count = 0, weight_bytes = 0, bf16_weight_count = 0;
    for (const auto& node : program.nodes) if (node.operation == "linear") {
      weight_count += node.weights().size();
      weight_bytes += node.data.size() * sizeof(double) + node.bf16_weights.size() * sizeof(uint16_t);
      bf16_weight_count += node.bf16_weights.size();
    }
    GenerationClient client;
    if (!client_path.empty()) {
      client.load(client_path);
      if (program.nodes[program.outputs.back().node].size != static_cast<int>(client.width))
        throw std::invalid_argument("client head and final hidden width mismatch");
    }
    const auto setup = Clock::now();
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_NotSet); p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetCKKSDataType(REAL); p.SetRingDim(65536); p.SetBatchSize(program.slots);
    p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(3); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(p);
    for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(feature);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    std::vector<int> rotations;
    for (int i = 1; i < program.slots; i *= 2) { rotations.push_back(i); rotations.push_back(-i); }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);
#ifdef FIDESLIB_S2C_FIRST_BOOTSTRAP
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, program.slots, 0, s2c_first);
#else
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, program.slots, 0);
#endif
    cc->EvalBootstrapKeyGen(keys.secretKey, program.slots);
    cc->LoadContext(keys.publicKey); sync_gpu();
    PackedEvaluator evaluator{cc, keys.publicKey, program.slots, program.bound};
    evaluator.replicated_linear = replicated_linear;
    evaluator.legacy_routing = legacy_routing;
    evaluator.trace_levels = trace_levels;
    evaluator.planned_refresh = planned_refresh;
    evaluator.batch_refresh = batch_refresh;
    evaluator.profile_evaluation = profile_evaluation;
    evaluator.inplace_ops = inplace_ops;
    evaluator.naf_rotations = naf_rotations;
    evaluator.reuse_dead_inputs = reuse_dead_inputs;
    evaluator.bsgs_routing_stages = bsgs_routing_stages;
    evaluator.frontier_refresh = frontier_refresh;
    // Moving four StC levels before ModRaise also returns four extra levels.
    // Keep the usable interval and the two-pass wrapper's safety margin equal.
    evaluator.refresh_ceiling = s2c_first ? 35 : 39;
    evaluator.refreshed_level = s2c_first ? 18 : 22;
    evaluator.cache_plaintexts = cache_plaintexts;
    evaluator.plaintexts = std::make_unique<fhemamba::PlaintextPreparation>(
        cc, program.slots, fhemamba::PlaintextPreparationOptions{
            .fast_upload = fast_plaintext_upload, .gpu_ntt = gpu_plaintext_ntt,
            .profile = profile_evaluation, .direct_upload = direct_plaintext_upload,
            .move_coefficients = move_plaintext_coefficients, .borrow_upload = borrow_plaintext_upload,
            .gpu_rns = gpu_plaintext_rns});
    evaluator.bootstrap_passes = bootstrap_passes;
    evaluator.hoist_rotations = hoist_rotations;
    evaluator.share_chebyshev = share_chebyshev;
    const double setup_seconds = elapsed(setup);
    std::cout << "setup_seconds=" << setup_seconds << std::endl;
    auto decrypt_client = [&](const Ct& value, int size) {
      Plaintext plain; auto handle = value->Clone();
      cc->Decrypt(keys.secretKey, handle, &plain);
      plain->SetLength(size); return plain->GetRealPackedValue();
    };
    if (!client_path.empty()) evaluator.client_feedback = [&](const Ct& hidden) {
      return evaluator.encrypt(client.select(decrypt_client(hidden, client.width)));
    };
    if (profile_evaluation && cudaProfilerStart() != 0) throw std::runtime_error("could not start CUDA profiler");
    const auto start = Clock::now();
    const auto encrypted_outputs = evaluator.evaluate(program);
    sync_gpu(); const double eval_seconds = elapsed(start);
    if (profile_evaluation && cudaProfilerStop() != 0) throw std::runtime_error("could not stop CUDA profiler");
    if (!client_path.empty()) client.select(decrypt_client(encrypted_outputs.back(), client.width));
    double polynomial_error = 0, exact_error = 0;
    int non_finite = 0;
    std::vector<double> per_output_poly, per_output_exact;
    for (std::size_t i = 0; i < program.outputs.size(); ++i) {
      Plaintext plaintext; auto handle = encrypted_outputs[i]->Clone();
      cc->Decrypt(keys.secretKey, handle, &plaintext);
      plaintext->SetLength(program.outputs[i].polynomial.size());
      const auto values = plaintext->GetRealPackedValue();
      double pe = 0, ee = 0;
      for (std::size_t j = 0; j < program.outputs[i].polynomial.size(); ++j) {
        if (!std::isfinite(values[j])) { ++non_finite; continue; }
        pe = std::max(pe, std::abs(values[j] - program.outputs[i].polynomial[j]));
        ee = std::max(ee, std::abs(values[j] - program.outputs[i].exact[j]));
      }
      polynomial_error = std::max(polynomial_error, pe); exact_error = std::max(exact_error, ee);
      per_output_poly.push_back(pe); per_output_exact.push_back(ee);
    }
    const bool passed = non_finite == 0 && polynomial_error <= poly_tol && exact_error <= exact_tol;
    struct rusage usage {}; getrusage(RUSAGE_SELF, &usage);
    std::ofstream report(argv[2]);
    report << std::setprecision(12) << "{\n\"schema\":\"fhemamba-packed-result-v1\","
           << "\"backend\":\"fideslib\",\"encrypted\":true,\"security\":\"not-set\","
           << "\"passed\":" << (passed ? "true" : "false")
           << ",\"slots\":" << program.slots << ",\"ring_dimension\":65536,\"depth\":44,\"scale_bits\":59"
           << ",\"nodes\":" << program.nodes.size() << ",\"setup_seconds\":" << setup_seconds
           << ",\"frontier_refresh\":" << (frontier_refresh ? "true" : "false")
           << ",\"s2c_first\":" << (s2c_first ? "true" : "false")
           << ",\"hoist_rotations\":" << (hoist_rotations ? "true" : "false")
           << ",\"share_chebyshev\":" << (share_chebyshev ? "true" : "false")
           << ",\"rotation_batch_edges\":" << evaluator.rotation_batch_stats.edges
           << ",\"rotation_batch_preparations\":" << evaluator.rotation_batch_stats.preparations
           << ",\"rotation_sibling_batches\":" << evaluator.rotation_batch_stats.sibling_batches
           << ",\"shared_basis_hits\":" << evaluator.shared_basis_hits
           << ",\"shared_basis_invalidations\":" << evaluator.shared_basis_invalidations
           << ",\"gpu_plaintext_rns\":" << (gpu_plaintext_rns ? "true" : "false")
           << ",\"compact_rns_encodes\":" << evaluator.plaintexts->compact_rns_encodes
           << ",\"compact_rns_uploads\":" << evaluator.plaintexts->compact_rns_uploads
           << ",\"compact_rns_fallbacks\":" << evaluator.plaintexts->compact_rns_fallbacks
           << ",\"compact_rns_saved_host_bytes\":" << evaluator.plaintexts->compact_rns_saved_host_bytes
           << ",\"refresh_ceiling\":" << evaluator.refresh_ceiling
           << ",\"refreshed_level\":" << evaluator.refreshed_level
           << ",\"frontier_deferrals\":" << evaluator.frontier_deferrals
           << ",\"maximum_ready_nodes\":" << evaluator.maximum_ready_nodes
           << ",\"evaluated_nodes\":" << evaluator.evaluated_nodes
           << ",\"eval_seconds\":" << eval_seconds << ",\"bootstrap_seconds\":" << evaluator.bootstrap_seconds
           << ",\"bootstraps\":" << evaluator.bootstraps << ",\"ct_ct_mul\":" << evaluator.ct_ct
           << ",\"ct_pt_mul\":" << evaluator.ct_pt << ",\"rotations\":" << evaluator.rotations
           << ",\"refresh_rotations\":" << evaluator.refresh_rotations
           << ",\"rotation_decomposition\":\"" << (naf_rotations ? "naf" : "binary") << '"'
           << ",\"reuse_dead_inputs\":" << (reuse_dead_inputs ? "true" : "false")
           << ",\"lifetime_clones_eliminated\":" << evaluator.lifetime_clones_eliminated
           << ",\"compact_weights\":" << (compact_weights ? "true" : "false")
           << ",\"public_weight_count\":" << weight_count
           << ",\"public_weight_bytes\":" << weight_bytes
           << ",\"bf16_weight_count\":" << bf16_weight_count
           << ",\"linear_method\":\"" << (evaluator.replicated_linear ? "replicated-bsgs" : "direct") << '"'
           << ",\"routing_method\":\"" << (!replicated_linear ? "direct" : legacy_routing ? "bitwise" :
                planned_refresh ? "radix8-bsgs32" : "radix8-direct32") << '"'
           << ",\"refresh_policy\":\"" << (planned_refresh ? "planned" : "baseline") << '"'
           << ",\"bootstrap_passes\":" << bootstrap_passes
           << ",\"batch_refresh\":" << (batch_refresh ? "true" : "false")
           << ",\"logical_refreshes\":" << evaluator.logical_refreshes
           << ",\"planned_logical_refreshes\":" << evaluator.planned_logical_refreshes
           << ",\"refresh_batches\":" << evaluator.refresh_batches
           << ",\"largest_refresh_batch\":" << evaluator.largest_refresh_batch
           << ",\"profile_evaluation\":" << (profile_evaluation ? "true" : "false")
           << ",\"inplace_ops\":" << (inplace_ops ? "true" : "false")
           << ",\"cache_plaintexts\":" << (cache_plaintexts ? "true" : "false")
           << ",\"fast_plaintext_upload\":" << (fast_plaintext_upload ? "true" : "false")
           << ",\"direct_plaintext_upload\":" << (direct_plaintext_upload ? "true" : "false")
           << ",\"direct_plaintext_uploads\":" << evaluator.plaintexts->direct_uploads
           << ",\"gpu_plaintext_ntt\":" << (gpu_plaintext_ntt ? "true" : "false")
           << ",\"move_plaintext_coefficients\":" << (move_plaintext_coefficients ? "true" : "false")
           << ",\"moved_coefficient_encodes\":" << evaluator.plaintexts->moved_coefficient_encodes
           << ",\"borrow_plaintext_upload\":" << (borrow_plaintext_upload ? "true" : "false")
           << ",\"borrowed_plaintext_uploads\":" << evaluator.plaintexts->borrowed_uploads
           << ",\"bsgs_routing_stages\":" << (bsgs_routing_stages ? "true" : "false")
           << ",\"optimized_routing_stages\":" << evaluator.optimized_routing_stages
           << ",\"routing_stage_rotations_saved\":" << evaluator.routing_stage_rotations_saved
           << ",\"plaintext_ntt_batch\":" << (gpu_plaintext_ntt ? fhemamba::kPlaintextNttBatch : 0)
           << ",\"gpu_ntt_encodes\":" << evaluator.plaintexts->gpu_ntt_encodes
           << ",\"fast_plaintext_uploads\":" << evaluator.plaintexts->fast_uploads
           << ",\"plaintext_upload_seconds\":" << evaluator.plaintexts->upload_seconds
           << ",\"plaintext_cache_capacity\":" << evaluator.plaintext_cache.capacity()
           << ",\"plaintext_cache_entries\":" << evaluator.plaintext_cache.size()
           << ",\"plaintext_cache_hits\":" << evaluator.plaintext_cache.hits
           << ",\"plaintext_cache_misses\":" << evaluator.plaintext_cache.misses
           << ",\"plaintext_cache_bypasses\":" << evaluator.plaintext_cache.bypasses
           << ",\"plaintext_cache_evictions\":" << evaluator.plaintext_cache.evictions
           << ",\"scratch_clones_eliminated\":" << evaluator.scratch_clones_eliminated
           << ",\"owned_arithmetic_calls\":" << evaluator.owned_arithmetic.calls
           << ",\"owned_arithmetic_reused_inputs\":" << evaluator.owned_arithmetic.reused_inputs
           << ",\"owned_arithmetic_cloned_inputs\":" << evaluator.owned_arithmetic.cloned_inputs
           << ",\"square_arithmetic_calls\":" << evaluator.square_arithmetic.calls
           << ",\"square_arithmetic_reused_inputs\":" << evaluator.square_arithmetic.reused_inputs
           << ",\"square_arithmetic_cloned_inputs\":" << evaluator.square_arithmetic.cloned_inputs
           << ",\"host_encoding_seconds\":" << evaluator.host_encoding_seconds
           << ",\"host_encodes\":" << evaluator.host_encodes
           << ",\"mask_preparation_seconds\":" << evaluator.mask_preparation_seconds
           << ",\"max_abs_error_vs_polynomial\":" << polynomial_error
           << ",\"max_abs_error_vs_exact\":" << exact_error << ",\"non_finite\":" << non_finite
           << ",\"polynomial_tolerance\":" << poly_tol << ",\"exact_tolerance\":" << exact_tol
           << ",\"peak_rss_gib\":" << usage.ru_maxrss / (1024.0 * 1024.0)
           << ",\"per_output_errors\":[";
    for (std::size_t i = 0; i < per_output_poly.size(); ++i) {
      if (i) report << ',';
      report << "{\"polynomial\":" << per_output_poly[i] << ",\"exact\":" << per_output_exact[i] << '}';
    }
    report << "],\"evaluation_decryptions\":0,\"client_output_decrypt_count\":" << client.tokens.size()
           << ",\"generated_token_ids\":[";
    for (std::size_t i = 0; i < client.tokens.size(); ++i) {
      if (i) report << ',';
      report << client.tokens[i];
    }
    report << "],\"client_logit_margins\":[";
    for (std::size_t i = 0; i < client.margins.size(); ++i) {
      if (i) report << ',';
      report << client.margins[i];
    }
    report << "],\"operation_stats\":{";
    bool first_operation = true;
    for (const auto& [name, stats] : evaluator.operation_stats) {
      if (!first_operation) report << ',';
      first_operation = false;
      report << '"' << name << "\":{\"nodes\":" << stats.nodes << ",\"seconds\":" << stats.seconds
             << ",\"bootstrap_seconds\":" << stats.bootstrap_seconds << ",\"bootstraps\":" << stats.bootstraps << '}';
    }
    report << "},\"scope\":\"" << (client_path.empty() ? "packed feasibility; inline client fixture" :
        "encrypted backbone with inline client vocabulary head and actual greedy feedback") << "\"}\n";
    if (!report) throw std::runtime_error("could not write packed result");
    std::cout << "passed=" << passed << " polynomial_error=" << polynomial_error
              << " exact_error=" << exact_error << " eval_seconds=" << eval_seconds << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl; return 2;
  }
}
