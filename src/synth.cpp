// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Synthetic GGUF / raw checkpoint writer.
#include "bonsai/synth.hpp"

#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

#include "bonsai/ggml_types.hpp"
#include "bonsai/quant.hpp"

namespace bonsai {

namespace {

SynthTensor make(const std::string& name, std::vector<uint64_t> ne, uint32_t type, std::mt19937& rng) {
  SynthTensor t{name, ne, type, {}};
  uint64_t n = 1;
  for (uint64_t d : ne) n *= d;
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  const uint64_t row = ne[0], rows = n / row;
  if (type == GGML_PTQ1_0 || type == GGML_Q4_0) {
    TypeTraits tt = type_traits(type);
    t.data.resize(n / tt.block_elems * tt.block_bytes);
    std::vector<float> x(row);
    for (uint64_t r = 0; r < rows; ++r) {
      for (auto& v : x) {
        if (type == GGML_PTQ1_0) {
          v = (float)((int)(rng() % 3) - 1) * 0.03125f;  // exactly ternary
        } else {
          v = uni(rng);
        }
      }
      uint8_t* dst = t.data.data() + r * (row / tt.block_elems * tt.block_bytes);
      if (type == GGML_PTQ1_0) quantize_row_ptq1_0(x.data(), dst, (int64_t)row);
      else quantize_row_q4_0(x.data(), dst, (int64_t)row);
    }
  } else if (type == GGML_F32) {
    t.data.resize(n * 4);
    for (uint64_t i = 0; i < n; ++i) {
      float v = uni(rng);
      std::memcpy(&t.data[i * 4], &v, 4);
    }
  } else if (type == GGML_F16 || type == GGML_BF16) {
    t.data.resize(n * 2);
    for (uint64_t i = 0; i < n; ++i) {
      uint16_t h = type == GGML_F16 ? float_to_half_bits(uni(rng)) : float_to_bf16_bits(uni(rng));
      std::memcpy(&t.data[i * 2], &h, 2);
    }
  } else {  // opaque type: arbitrary bytes
    t.data.resize(40);
    for (auto& b : t.data) b = (uint8_t)rng();
  }
  return t;
}

void put_u32(std::vector<uint8_t>& o, uint32_t v) { o.insert(o.end(), (uint8_t*)&v, (uint8_t*)&v + 4); }
void put_u64(std::vector<uint8_t>& o, uint64_t v) { o.insert(o.end(), (uint8_t*)&v, (uint8_t*)&v + 8); }
void put_str(std::vector<uint8_t>& o, const std::string& s) {
  put_u64(o, s.size());
  o.insert(o.end(), s.begin(), s.end());
}

}  // namespace

std::vector<SynthTensor> make_synth_tensors(uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<SynthTensor> v;
  v.push_back(make("token_embd.weight", {256, 40}, GGML_PTQ1_0, rng));
  v.push_back(make("output_norm.weight", {48}, GGML_F32, rng));
  v.push_back(make("blk.0.attn_q.weight", {48, 37}, GGML_F32, rng));
  v.push_back(make("blk.0.ffn_up.weight", {64, 33}, GGML_F16, rng));
  v.push_back(make("blk.0.opaque", {64, 5}, 200, rng));  // geometry unknown to this loader
  v.push_back(make("blk.0.ffn_down.weight", {96, 21}, GGML_Q4_0, rng));
  v.push_back(make("blk.0.ssm_a", {16, 3, 2}, GGML_BF16, rng));
  v.push_back(make("blk.0.ffn_gate.weight", {384, 19}, GGML_PTQ1_0, rng));
  return v;
}

uint64_t write_synth_gguf(const std::string& path, const std::vector<SynthTensor>& tensors,
                          uint32_t alignment) {
  std::vector<uint8_t> o;
  o.insert(o.end(), {'G', 'G', 'U', 'F'});
  put_u32(o, 3);
  put_u64(o, tensors.size());
  put_u64(o, 7);  // kv count
  put_str(o, "general.architecture"); put_u32(o, 8); put_str(o, "bonsai-synth");
  put_str(o, "general.alignment"); put_u32(o, 4); put_u32(o, alignment);
  put_str(o, "general.name"); put_u32(o, 8); put_str(o, "synthetic");
  put_str(o, "synth.tokens"); put_u32(o, 9); put_u32(o, 8); put_u64(o, 3);
  put_str(o, "a"); put_str(o, "bc"); put_str(o, "def");
  put_str(o, "synth.f32"); put_u32(o, 6); { float f = 1.5f; uint32_t u; std::memcpy(&u, &f, 4); put_u32(o, u); }
  put_str(o, "synth.flag"); put_u32(o, 7); o.push_back(1);
  put_str(o, "synth.ids"); put_u32(o, 9); put_u32(o, 5); put_u64(o, 4);
  for (int32_t i = 0; i < 4; ++i) put_u32(o, (uint32_t)i);

  uint64_t off = 0;
  std::vector<uint64_t> offs;
  for (const SynthTensor& t : tensors) {
    offs.push_back(off);
    off = (off + t.data.size() + alignment - 1) / alignment * alignment;
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    const SynthTensor& t = tensors[i];
    put_str(o, t.name);
    put_u32(o, (uint32_t)t.ne.size());
    for (uint64_t d : t.ne) put_u64(o, d);
    put_u32(o, t.type);
    put_u64(o, offs[i]);
  }
  o.resize((o.size() + alignment - 1) / alignment * alignment, 0);
  const uint64_t data_start = o.size();
  for (size_t i = 0; i < tensors.size(); ++i) {
    o.resize(data_start + offs[i], 0);
    o.insert(o.end(), tensors[i].data.begin(), tensors[i].data.end());
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write((const char*)o.data(), (std::streamsize)o.size());
  return o.size();
}

void write_synth_raw(const std::string& blob_path, const std::string& manifest_path,
                     const std::vector<SynthTensor>& tensors) {
  std::ofstream blob(blob_path, std::ios::binary), man(manifest_path);
  if (!blob || !man) throw std::runtime_error("cannot write raw checkpoint");
  man << "# name offset bytes dtype shape(ne0,ne1,...)\n";
  uint64_t off = 0;
  for (const SynthTensor& t : tensors) {
    if (!type_traits(t.type).known) continue;
    blob.write((const char*)t.data.data(), (std::streamsize)t.data.size());
    man << t.name << ' ' << off << ' ' << t.data.size() << ' ' << type_name(t.type) << ' ';
    for (size_t d = 0; d < t.ne.size(); ++d) man << (d ? "," : "") << t.ne[d];
    man << '\n';
    off += t.data.size();
  }
}

}  // namespace bonsai
