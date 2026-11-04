// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GGUF header parser. Every field is bounds-checked; nothing is assumed about
// offsets beyond what the header states.
//
// Layout (gguf v3): "GGUF" u32 version, u64 n_tensors, u64 n_kv,
//   n_kv x { string key, u32 type, value },
//   n_tensors x { string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset },
//   pad to general.alignment (default 32), tensor data.
// Strings are u64 length + bytes. Arrays are u32 elem type + u64 count + elems.
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"

namespace bonsai {

namespace {

enum GgufValueType : uint32_t {
  GV_U8 = 0, GV_I8 = 1, GV_U16 = 2, GV_I16 = 3, GV_U32 = 4, GV_I32 = 5,
  GV_F32 = 6, GV_BOOL = 7, GV_STRING = 8, GV_ARRAY = 9, GV_U64 = 10,
  GV_I64 = 11, GV_F64 = 12,
};

class Reader {
 public:
  Reader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  size_t pos() const { return p_; }
  void need(size_t k) const {
    if (k > n_ || p_ > n_ - k)
      throw std::runtime_error("gguf: truncated header at byte " + std::to_string(p_));
  }
  template <typename T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, d_ + p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  std::string str() {
    uint64_t len = get<uint64_t>();
    if (len > (1ull << 30)) throw std::runtime_error("gguf: string length too large");
    need(len);
    std::string s(reinterpret_cast<const char*>(d_ + p_), len);
    p_ += len;
    return s;
  }
  void skip(uint64_t k) {
    need(k);
    p_ += k;
  }

 private:
  const uint8_t* d_;
  size_t n_;
  size_t p_ = 0;
};

size_t scalar_size(uint32_t t) {
  switch (t) {
    case GV_U8: case GV_I8: case GV_BOOL: return 1;
    case GV_U16: case GV_I16: return 2;
    case GV_U32: case GV_I32: case GV_F32: return 4;
    case GV_U64: case GV_I64: case GV_F64: return 8;
    default: return 0;
  }
}

std::string read_scalar(Reader& r, uint32_t t, uint64_t* as_u64) {
  uint64_t u = 0;
  std::string s;
  switch (t) {
    case GV_U8: u = r.get<uint8_t>(); s = std::to_string(u); break;
    case GV_I8: { int8_t v = r.get<int8_t>(); u = (uint64_t)v; s = std::to_string(v); break; }
    case GV_U16: u = r.get<uint16_t>(); s = std::to_string(u); break;
    case GV_I16: { int16_t v = r.get<int16_t>(); u = (uint64_t)v; s = std::to_string(v); break; }
    case GV_U32: u = r.get<uint32_t>(); s = std::to_string(u); break;
    case GV_I32: { int32_t v = r.get<int32_t>(); u = (uint64_t)v; s = std::to_string(v); break; }
    case GV_F32: s = std::to_string(r.get<float>()); break;
    case GV_BOOL: u = r.get<uint8_t>(); s = u ? "true" : "false"; break;
    case GV_U64: u = r.get<uint64_t>(); s = std::to_string(u); break;
    case GV_I64: { int64_t v = r.get<int64_t>(); u = (uint64_t)v; s = std::to_string(v); break; }
    case GV_F64: s = std::to_string(r.get<double>()); break;
    case GV_STRING: s = r.str(); break;
    default: throw std::runtime_error("gguf: bad value type " + std::to_string(t));
  }
  if (as_u64) *as_u64 = u;
  return s;
}

std::string read_value(Reader& r, uint32_t t, uint64_t* as_u64) {
  if (t != GV_ARRAY) return read_scalar(r, t, as_u64);
  uint32_t et = r.get<uint32_t>();
  uint64_t count = r.get<uint64_t>();
  if (et == GV_ARRAY) throw std::runtime_error("gguf: nested arrays are not supported");
  if (et == GV_STRING) {
    for (uint64_t i = 0; i < count; ++i) r.str();
  } else {
    size_t es = scalar_size(et);
    if (es == 0) throw std::runtime_error("gguf: bad array element type");
    if (count > (uint64_t)1 << 40) throw std::runtime_error("gguf: array too large");
    r.skip(count * es);
  }
  return "[array of " + std::to_string(count) + "]";
}

}  // namespace

const KeyValue* Checkpoint::find_kv(const std::string& key) const {
  for (const KeyValue& k : kv)
    if (k.key == key) return &k;
  return nullptr;
}

uint64_t Checkpoint::total_tensor_bytes() const {
  uint64_t s = 0;
  for (const TensorInfo& t : tensors) s += t.bytes;
  return s;
}

Checkpoint parse_gguf(const uint8_t* data, size_t size, bool allow_header_only) {
  Reader r(data, size);
  r.need(4);
  if (std::memcmp(data, "GGUF", 4) != 0) throw std::runtime_error("gguf: bad magic");
  r.skip(4);
  Checkpoint c;
  c.format = "gguf";
  c.file_bytes = size;
  c.header_only = allow_header_only;
  c.version = r.get<uint32_t>();
  if (c.version != 2 && c.version != 3)
    throw std::runtime_error("gguf: unsupported version " + std::to_string(c.version));
  uint64_t n_tensors = r.get<uint64_t>();
  uint64_t n_kv = r.get<uint64_t>();
  if (n_tensors > (1u << 24) || n_kv > (1u << 24))
    throw std::runtime_error("gguf: implausible tensor/kv count");

  c.alignment = 32;  // GGUF default when general.alignment is absent
  for (uint64_t i = 0; i < n_kv; ++i) {
    KeyValue k;
    k.key = r.str();
    k.type = r.get<uint32_t>();
    uint64_t u = 0;
    k.value = read_value(r, k.type, &u);
    if (k.key == "general.alignment") {
      if (k.type != GV_U32 || u == 0 || (u & (u - 1)) != 0)
        throw std::runtime_error("gguf: general.alignment must be a power-of-two u32");
      c.alignment = u;
    }
    c.kv.push_back(std::move(k));
  }

  c.tensors.resize(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo& t = c.tensors[i];
    t.name = r.str();
    uint32_t nd = r.get<uint32_t>();
    if (nd == 0 || nd > 8) throw std::runtime_error("gguf: bad n_dims for " + t.name);
    t.ne.resize(nd);
    t.n_elements = 1;
    for (uint32_t d = 0; d < nd; ++d) {
      t.ne[d] = r.get<uint64_t>();
      if (t.ne[d] == 0 || t.n_elements > (UINT64_MAX / t.ne[d]))
        throw std::runtime_error("gguf: bad shape for " + t.name);
      t.n_elements *= t.ne[d];
    }
    t.type = r.get<uint32_t>();
    t.offset = r.get<uint64_t>();
    if (t.offset % c.alignment != 0)
      throw std::runtime_error("gguf: misaligned offset for " + t.name);
  }
  c.data_offset = (r.pos() + c.alignment - 1) / c.alignment * c.alignment;
  if (!allow_header_only && c.data_offset > size)
    throw std::runtime_error("gguf: data section starts past end of file");

  // Sizes: from type geometry when known, then checked against the gap to the
  // next tensor (or end of file). Unknown types take the gap itself.
  std::vector<size_t> order(n_tensors);
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return c.tensors[a].offset < c.tensors[b].offset; });
  const uint64_t data_bytes = size >= c.data_offset ? size - c.data_offset : 0;
  for (size_t oi = 0; oi < order.size(); ++oi) {
    TensorInfo& t = c.tensors[order[oi]];
    t.file_offset = c.data_offset + t.offset;
    bool last = oi + 1 == order.size();
    bool have_end = !(last && allow_header_only);
    uint64_t end = last ? data_bytes : c.tensors[order[oi + 1]].offset;
    if (have_end && end < t.offset) throw std::runtime_error("gguf: tensor " + t.name + " starts past end of file");
    if (!last && end == t.offset) throw std::runtime_error("gguf: overlapping tensors at " + t.name);
    TypeTraits tt = type_traits(t.type);
    if (tt.known) {
      if (t.ne[0] % tt.block_elems != 0)
        throw std::runtime_error("gguf: row of " + t.name + " is not whole blocks");
      t.bytes = t.n_elements / tt.block_elems * tt.block_bytes;
      t.size_from_type = true;
      if (have_end && t.offset + t.bytes > end)
        throw std::runtime_error("gguf: tensor " + t.name + " overruns its slot (" +
                                 std::to_string(t.bytes) + " bytes)");
    } else {
      if (!have_end)
        throw std::runtime_error("gguf: cannot size last tensor of unknown type without data");
      t.bytes = end - t.offset;
    }
    if (!allow_header_only && t.file_offset + t.bytes > size)
      throw std::runtime_error("gguf: tensor " + t.name + " extends past end of file");
  }
  return c;
}

}  // namespace bonsai
