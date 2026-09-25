#include "fideslib_plaintext_encoder.hpp"
#include "fideslib_plaintext_ops.hpp"
#include "plaintext_rns.hpp"
#ifdef FHEMAMBA_GPU_PLAINTEXT_RNS
#include "fideslib_plaintext_rns.hpp"
#endif
#include <CKKS/Context.cuh>
#include <CKKS/Plaintext.cuh>
#include <math/dftransform.h>
#include <bit>
#include <chrono>
#include <stdexcept>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fhemamba {
namespace {
void synchronize() {
  if (cudaDeviceSynchronize() != 0) throw std::runtime_error("plaintext GPU synchronization failed");
}

// Pinned FIDESlib load_convert<uint64_t> and load<uint64_t> each duplicate
// their input vector. The bridge already owns stable uint64_t staging arrays;
// copy those directly to the same limb streams and retain them through sync.
// Keep the library path for layouts outside the ordinary modulus prefix.
bool load_raw_direct(FIDESlib::CKKS::Plaintext& gpu,
                     const FIDESlib::CKKS::RawPlainText& raw) {
  if (raw.numRes <= 0 || raw.numRes > gpu.cc.L + 1 || raw.N != gpu.cc.N) return false;
  for (int i = 0; i < raw.numRes; ++i)
    if (raw.moduli[i] != gpu.cc.prime.at(i).p || raw.sub_0[i].size() != raw.N) return false;
  FIDESlib::CKKS::SetCurrentContext(gpu.cc_);
  gpu.c0.grow(raw.numRes - 1, false, raw.format != ::Format::COEFFICIENT);
  for (int i = 0; i < raw.numRes; ++i) {
    const auto position = gpu.cc.limbGPUid[i];
    auto& partition = gpu.c0.GPU.at(position.x);
    if (cudaSetDevice(partition.device) != cudaSuccess)
      throw std::runtime_error("could not select plaintext limb device");
    std::visit([&](auto& limb) {
      using Word = std::remove_pointer_t<decltype(limb.v.data)>;
      if constexpr (std::is_same_v<Word, uint64_t>) {
        if (cudaMemcpyAsync(limb.v.data, raw.sub_0[i].data(), raw.N * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, limb.stream.ptr()) != cudaSuccess)
          throw std::runtime_error("direct plaintext upload failed");
      } else {
        limb.load_convert(raw.sub_0[i]);
      }
    }, partition.limb.at(position.y));
  }
  gpu.NoiseFactor = raw.Noise; gpu.NoiseLevel = raw.NoiseLevel; gpu.slots = raw.slots;
  return true;
}

// The pinned NativeInteger is a standard-layout wrapper with one uint64_t
// member at offset zero; NativeVector uses contiguous std::vector storage.
// Read its object representation through CUDA's const-void byte interface,
// never through a reinterpreted uint64_t lvalue. This does not require the
// wrapper's user-provided copy constructor to be trivially copyable.
bool load_native_borrowed(FIDESlib::CKKS::Plaintext& gpu, const lbcrypto::Plaintext& cpu) {
#if BLOCK_VECTOR_ALLOCATION == 1
  return false;
#else
  using Native = lbcrypto::NativeInteger;
  if constexpr (!std::is_standard_layout_v<Native> || sizeof(Native) != sizeof(uint64_t) ||
                !std::is_same_v<typename Native::Integer, uint64_t> ||
                std::endian::native != std::endian::little) {
    return false;
  } else {
    const auto& poly = cpu->GetElement<lbcrypto::DCRTPoly>();
    const auto& limbs = poly.GetAllElements();
    if (limbs.empty() || limbs.size() > static_cast<std::size_t>(gpu.cc.L + 1) ||
        poly.GetRingDimension() != gpu.cc.N) return false;
    for (std::size_t i = 0; i < limbs.size(); ++i)
      if (limbs[i].GetModulus().ConvertToInt() != gpu.cc.prime.at(i).p ||
          limbs[i].GetValues().GetLength() != gpu.cc.N) return false;
    FIDESlib::CKKS::SetCurrentContext(gpu.cc_);
    gpu.c0.grow(limbs.size() - 1, false, poly.GetFormat() != ::Format::COEFFICIENT);
    // Validate every target limb before scheduling any borrowed read. Narrow
    // limbs retain the conversion path in load_raw_direct.
    for (std::size_t i = 0; i < limbs.size(); ++i) {
      const auto position = gpu.cc.limbGPUid[i];
      const auto& limb = gpu.c0.GPU.at(position.x).limb.at(position.y);
      const bool native_word = std::visit([](const auto& value) {
        return std::is_same_v<std::remove_pointer_t<decltype(value.v.data)>, uint64_t>;
      }, limb);
      if (!native_word) return false;
    }
    for (std::size_t i = 0; i < limbs.size(); ++i) {
      const auto position = gpu.cc.limbGPUid[i];
      auto& partition = gpu.c0.GPU.at(position.x);
      if (cudaSetDevice(partition.device) != cudaSuccess)
        throw std::runtime_error("could not select borrowed plaintext device");
      const void* source = std::addressof(limbs[i].GetValues()[0]);
      std::visit([&](auto& limb) {
        if (cudaMemcpyAsync(limb.v.data, source, gpu.cc.N * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, limb.stream.ptr()) != cudaSuccess)
          throw std::runtime_error("borrowed plaintext upload failed");
      }, partition.limb.at(position.y));
    }
    gpu.NoiseFactor = cpu->GetScalingFactor();
    gpu.NoiseLevel = cpu->GetNoiseScaleDeg();
    gpu.slots = cpu->GetSlots();
    return true;
  }
#endif
}
}

CoefficientPlaintextEncoder::CoefficientPlaintextEncoder(Context context, uint32_t slots,
                                                       bool move_coefficients, bool compact_rns)
    : context_(std::move(context)),
      cpu_(std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(context_->cpu)),
      slots_(slots), move_coefficients_(move_coefficients), compact_rns_(compact_rns) {
  const auto n = cpu_->GetRingDimension();
  if (!std::has_single_bit(slots_) || slots_ > n / 2)
    throw std::invalid_argument("plaintext slots must divide N/2");
  const auto original = cpu_->GetElementParams();
  std::vector<lbcrypto::NativeInteger> moduli, roots;
  for (const auto& tower : original->GetParams()) {
    moduli.push_back(tower->GetModulus());
    roots.emplace_back(1);
  }
  compact_coefficient_params_ = std::make_shared<Params>(2 * n,
      std::vector<lbcrypto::NativeInteger>{moduli.front()}, std::vector<lbcrypto::NativeInteger>{1});
  compact_params_ = std::make_shared<Params>(*original);
  while (compact_params_->GetParams().size() > 1) compact_params_->PopLastParam();
  const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu_->GetCryptoParameters());
  compact_rns_ = compact_rns_ && slots_ == n / 2 && context_->loaded && context_->devices.size() == 1 &&
      cp->GetScalingTechnique() == lbcrypto::FLEXIBLEAUTO;
#if NATIVEINT != 64 || BLOCK_VECTOR_ALLOCATION == 1
  compact_rns_ = false;
#endif
  for (const auto& tower : original->GetParams())
    compact_rns_ = compact_rns_ && tower->GetModulus().GetMSB() <= 60;
  for (std::size_t level = 0; level < original->GetParams().size(); ++level) {
    auto full = std::make_shared<Params>(*original);
    for (std::size_t drop = 0; drop < level; ++drop) full->PopLastParam();
    full_params_.push_back(std::move(full));
    // As in PeriodicPlaintextEncoder, identity roots suppress only the last
    // format transform, without modifying OpenFHE's process-global NTT tables.
    coefficient_params_.push_back(std::make_shared<Params>(2 * n, moduli, roots));
    moduli.pop_back(); roots.pop_back();
  }
  lbcrypto::DiscreteFourierTransform::Initialize(2 * n, slots_);
}

auto CoefficientPlaintextEncoder::encode(const std::vector<double>& values,
    uint32_t level, std::size_t degree) const -> fideslib::Plaintext {
  if (values.size() > slots_ || level >= full_params_.size() || degree == 0)
    throw std::invalid_argument("plaintext values, level or degree out of range");
  const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu_->GetCryptoParameters());
  const bool compact = compact_rns_ && degree == 1 && full_params_[level]->GetParams().size() > 1 &&
      compact_plaintext_fits(values, slots_, cp->GetScalingFactorReal(level),
                            compact_params_->GetParams()[0]->GetModulus().ConvertToInt());
  lbcrypto::Plaintext plaintext;
  {
#ifdef _OPENMP
    struct SerialIdentity {
      int previous = omp_get_max_active_levels();
      SerialIdentity() { omp_set_max_active_levels(0); }
      ~SerialIdentity() { omp_set_max_active_levels(previous); }
    } serial;
#endif
    plaintext = cpu_->MakeCKKSPackedPlaintext(values, degree, level,
        compact ? compact_coefficient_params_ : coefficient_params_[level], slots_);
  }
  auto& temporary = plaintext->GetElement<lbcrypto::DCRTPoly>();
  lbcrypto::DCRTPoly coefficients(compact ? compact_params_ : full_params_[level], ::Format::COEFFICIENT, false);
  auto& output = coefficients.GetAllElements();
  auto& input = temporary.GetAllElements();
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (move_coefficients_) {
      // The freshly encoded temporary exclusively owns these arrays. The
      // pinned OpenFHE exposes its unique_ptr; use its validated move setter
      // to retain each allocation while restoring real roots and COEFFICIENT
      // format. Never move from caller input or shared context parameters.
      output[i].SetValues(std::move(*input[i].m_values), ::Format::COEFFICIENT);
    } else {
      output[i].SetValues(input[i].GetValues(), ::Format::COEFFICIENT);
    }
  }
  temporary = std::move(coefficients);
  auto context_copy = context_;
  auto result = std::make_shared<fideslib::PlaintextImpl>(std::move(context_copy));
  result->cpu = std::make_any<lbcrypto::Plaintext>(std::move(plaintext));
  // Do not use the ordinary loader: the pinned version assumes EVALUATION.
  return result;
}

auto load_plaintext(fideslib::CryptoContext<fideslib::DCRTPoly>& context,
                    fideslib::Plaintext& plaintext, int ntt_batch, PlaintextUploadMode mode,
                    PlaintextRnsWorkspace* workspace)
    -> PlaintextUploadMode {
  if (ntt_batch <= 0) throw std::invalid_argument("NTT batch must be positive");
  if (plaintext->loaded) return PlaintextUploadMode::AlreadyLoaded;
  if (!context->loaded || context->devices.empty())
    throw std::invalid_argument("plaintext upload requires a loaded GPU context");
  const auto& cpu = std::any_cast<const lbcrypto::Plaintext&>(plaintext->cpu);
  const auto& poly = cpu->GetElement<lbcrypto::DCRTPoly>();
  const auto& limbs = poly.GetAllElements();
  auto& gpu_context = std::any_cast<FIDESlib::CKKS::Context&>(context->gpu);
  auto gpu = std::make_shared<FIDESlib::CKKS::Plaintext>(gpu_context);
  auto actual = PlaintextUploadMode::Staged;
  FIDESlib::CKKS::RawPlainText raw{};
  const auto expected_towers = gpu->cc.L + 1 - static_cast<int>(cpu->GetLevel());
  const bool compact = static_cast<int>(limbs.size()) != expected_towers;
  if (compact && mode != PlaintextUploadMode::CompactRns)
    throw std::invalid_argument("compact plaintext requires the GPU RNS loader");
  if (mode == PlaintextUploadMode::CompactRns && compact) {
#ifdef FHEMAMBA_GPU_PLAINTEXT_RNS
    using Native = lbcrypto::NativeInteger;
    if constexpr (!std::is_standard_layout_v<Native> || sizeof(Native) != sizeof(uint64_t) ||
                  !std::is_same_v<typename Native::Integer, uint64_t> ||
                  std::endian::native != std::endian::little)
      throw std::invalid_argument("unsupported compact plaintext word layout");
    if (context->devices.size() != 1 || limbs.size() != 1 || expected_towers <= 1 ||
        expected_towers > gpu->cc.L + 1 || cpu->GetNoiseScaleDeg() != 1 ||
        poly.GetFormat() != ::Format::COEFFICIENT || poly.GetRingDimension() != gpu->cc.N ||
        cpu->GetSlots() != gpu->cc.N / 2 || limbs[0].GetValues().GetLength() != gpu->cc.N ||
        limbs[0].GetModulus().ConvertToInt() != gpu->cc.prime[0].p)
      throw std::invalid_argument("invalid compact plaintext layout");
    FIDESlib::CKKS::SetCurrentContext(gpu->cc_);
    gpu->c0.grow(expected_towers - 1, false);
    PlaintextRnsWorkspace temporary_workspace;
    if (!workspace) workspace = &temporary_workspace;
    const auto* source = workspace->upload(std::addressof(limbs[0].GetValues()[0]),
                                          gpu->cc.N * sizeof(uint64_t), context->devices[0]);
    for (int i = 0; i < expected_towers; ++i) {
      const auto position = gpu->cc.limbGPUid[i];
      auto& partition = gpu->c0.GPU.at(position.x);
      if (partition.device != context->devices[0])
        throw std::invalid_argument("compact plaintext requires one device");
      std::visit([&](auto& limb) {
        if constexpr (std::is_same_v<std::remove_pointer_t<decltype(limb.v.data)>, uint64_t>)
          expand_plaintext_rns(source, limb.v.data, gpu->cc.N, gpu->cc.prime[0].p,
                               gpu->cc.prime[i].p, limb.stream.ptr());
        else throw std::invalid_argument("compact plaintext requires 64-bit limbs");
      }, partition.limb.at(position.y));
    }
    // A temporary workspace must not die before any limb finishes reading it.
    synchronize();
    gpu->NoiseFactor = cpu->GetScalingFactor(); gpu->NoiseLevel = 1; gpu->slots = cpu->GetSlots();
    actual = PlaintextUploadMode::CompactRns;
#else
    throw std::invalid_argument("GPU RNS expansion is not built");
#endif
  } else if ((mode == PlaintextUploadMode::Borrowed || mode == PlaintextUploadMode::CompactRns) &&
             load_native_borrowed(*gpu, cpu)) {
    actual = PlaintextUploadMode::Borrowed;
  } else {
    raw.originalPlainText = cpu;
    raw.numRes = limbs.size(); raw.N = poly.GetRingDimension(); raw.format = poly.GetFormat();
    raw.Noise = cpu->GetScalingFactor(); raw.NoiseLevel = cpu->GetNoiseScaleDeg();
    raw.slots = cpu->GetSlots();
    raw.sub_0.reserve(limbs.size()); raw.moduli.reserve(limbs.size());
    for (const auto& limb : limbs) {
      const auto& values = limb.GetValues();
      auto& output = raw.sub_0.emplace_back(values.GetLength());
      for (std::size_t i = 0; i < output.size(); ++i) output[i] = values[i].ConvertToInt();
      raw.moduli.push_back(limb.GetModulus().ConvertToInt());
    }
    // Constant plaintext buffers omit NTT scratch. Allocate ordinary limbs
    // (including aux pointers) before loadConstant so the transform can use them.
    const bool direct = (mode == PlaintextUploadMode::Direct || mode == PlaintextUploadMode::Borrowed ||
                         mode == PlaintextUploadMode::CompactRns) &&
                        load_raw_direct(*gpu, raw);
    if (direct) actual = PlaintextUploadMode::Direct;
    if (!direct) {
      if (raw.format == ::Format::COEFFICIENT) gpu->c0.grow(raw.numRes - 1);
      gpu->load(raw);
    }
  }
  if (poly.GetFormat() == ::Format::COEFFICIENT) gpu->c0.NTT(ntt_batch, true);
  // Both raw staging and borrowed CPU storage remain alive until every device
  // has finished. The measured configuration uses one device.
  int original_device = 0;
  if (cudaGetDevice(&original_device) != cudaSuccess)
    throw std::runtime_error("could not read plaintext device");
  for (int device : context->devices) {
    if (cudaSetDevice(device) != cudaSuccess)
      throw std::runtime_error("could not synchronize plaintext device");
    synchronize();
  }
  if (cudaSetDevice(original_device) != cudaSuccess)
    throw std::runtime_error("could not restore plaintext device");
  plaintext->gpu = context->RegisterDevicePlaintext(std::move(gpu));
  plaintext->loaded = true;
  return actual;
}

auto readback_plaintext(const fideslib::Plaintext& plaintext) -> lbcrypto::DCRTPoly {
  if (!plaintext->loaded) throw std::invalid_argument("plaintext is not loaded");
  auto& cc = plaintext->parent_context;
  auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(cc->GetDevicePlaintext(plaintext->gpu));
  FIDESlib::CKKS::RawPlainText raw{};
  gpu->store(raw); synchronize();
  const auto& cpu = std::any_cast<const lbcrypto::Plaintext&>(plaintext->cpu);
  auto params = std::make_shared<lbcrypto::DCRTPoly::Params>(
      *std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu)->GetElementParams());
  for (uint32_t i = 0; i < cpu->GetLevel(); ++i) params->PopLastParam();
  lbcrypto::DCRTPoly out(params, ::Format::EVALUATION, false);
  auto& limbs = out.GetAllElements();
  if (raw.sub_0.size() != limbs.size()) throw std::runtime_error("GPU plaintext limb count differs");
  for (std::size_t i = 0; i < limbs.size(); ++i) {
    lbcrypto::NativeVector values(raw.sub_0[i].size(), limbs[i].GetModulus());
    for (std::size_t j = 0; j < values.GetLength(); ++j) values[j] = raw.sub_0[i][j];
    limbs[i].SetValues(std::move(values), ::Format::EVALUATION);
  }
  return out;
}

PlaintextPreparation::PlaintextPreparation(Context context, uint32_t slots,
    PlaintextPreparationOptions options)
    : context_(std::move(context)), slots_(slots),
      fast_upload_(options.fast_upload || options.gpu_ntt || options.direct_upload ||
                   options.move_coefficients || options.borrow_upload || options.gpu_rns), profile_(options.profile),
      move_coefficients_(options.move_coefficients), gpu_rns_(options.gpu_rns),
      upload_mode_(options.gpu_rns ? PlaintextUploadMode::CompactRns : options.borrow_upload ? PlaintextUploadMode::Borrowed :
                   options.direct_upload ? PlaintextUploadMode::Direct : PlaintextUploadMode::Staged) {
  if (options.gpu_rns) {
#ifdef FHEMAMBA_GPU_PLAINTEXT_RNS
    rns_workspace_ = std::make_unique<PlaintextRnsWorkspace>();
#else
    throw std::invalid_argument("GPU RNS expansion is not built");
#endif
  }
  if (options.gpu_ntt || options.move_coefficients || options.gpu_rns)
    coefficient_ = std::make_unique<CoefficientPlaintextEncoder>(context_, slots_, options.move_coefficients, options.gpu_rns);
}

PlaintextPreparation::~PlaintextPreparation() = default;

void PlaintextPreparation::enable_periodic_encoding(uint32_t slots) {
  periodic_ = std::make_unique<stage1::PeriodicPlaintextEncoder>(context_, slots);
}

auto PlaintextPreparation::encode(const std::vector<double>& values, uint32_t level,
    std::size_t degree, uint32_t packing_slots) -> fideslib::Plaintext {
  if (packing_slots && periodic_) {
    if (packing_slots != periodic_->slots())
      throw std::invalid_argument("unexpected periodic plaintext slot count");
    ++subring_encodes;
    return periodic_->encode(values, level, degree);
  }
  if (coefficient_ && (!packing_slots || packing_slots == slots_)) {
    ++gpu_ntt_encodes;
    if (move_coefficients_) ++moved_coefficient_encodes;
    auto result = coefficient_->encode(values, level, degree);
    if (gpu_rns_) {
      const auto& cpu = std::any_cast<const lbcrypto::Plaintext&>(result->cpu);
      const auto original = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(context_->cpu)->GetElementParams();
      const auto towers = original->GetParams().size() - level;
      if (cpu->GetElement<lbcrypto::DCRTPoly>().GetNumOfElements() < towers) {
        ++compact_rns_encodes;
        compact_rns_saved_host_bytes += (towers - 1) * original->GetRingDimension() * sizeof(uint64_t);
      } else ++compact_rns_fallbacks;
    }
    return result;
  }
  return context_->MakeCKKSPackedPlaintext(
      values, degree, level, nullptr, packing_slots ? packing_slots : slots_);
}

auto PlaintextPreparation::align_addend(const Ciphertext& ciphertext,
    fideslib::Plaintext plaintext, const std::vector<double>& values,
    long long& reencodes) -> fideslib::Plaintext {
  return stage1::additive_plaintext(ciphertext, std::move(plaintext), values, reencodes,
      [&](const auto& v, uint32_t level, std::size_t degree) {
        return encode(v, level, degree);
      });
}

auto PlaintextPreparation::encode_addend(const Ciphertext& ciphertext,
    const std::vector<double>& values, long long& reencodes) -> fideslib::Plaintext {
  if (coefficient_) return encode(values, ciphertext->GetLevel(), ciphertext->GetNoiseScaleDeg());
  return align_addend(ciphertext, encode(values, ciphertext->GetLevel()), values, reencodes);
}

void PlaintextPreparation::load(fideslib::Plaintext& plaintext) {
  if (!fast_upload_ || plaintext->loaded) return;
  const auto start = profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  PlaintextRnsWorkspace* workspace = nullptr;
#ifdef FHEMAMBA_GPU_PLAINTEXT_RNS
  workspace = rns_workspace_.get();
#endif
  const auto mode = load_plaintext(context_, plaintext, kPlaintextNttBatch, upload_mode_, workspace);
  if (mode == PlaintextUploadMode::CompactRns) ++compact_rns_uploads;
  if (mode == PlaintextUploadMode::Direct || mode == PlaintextUploadMode::Borrowed) ++direct_uploads;
  if (mode == PlaintextUploadMode::Borrowed) ++borrowed_uploads;
  ++fast_uploads;
  if (profile_) upload_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
}

}  // namespace fhemamba
