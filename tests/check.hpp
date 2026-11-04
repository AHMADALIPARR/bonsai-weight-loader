// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Minimal test helpers. Exit code 0 = pass, 1 = fail, 77 = skipped (no GPU).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>

namespace bonsai_test {
inline int& failures() {
  static int n = 0;
  return n;
}
inline int finish(const char* name) {
  if (failures()) {
    std::printf("%s: %d check(s) FAILED\n", name, failures());
    return 1;
  }
  std::printf("%s: all checks passed\n", name);
  return 0;
}
inline std::string tmp_path(const std::string& leaf) {
  const char* d = std::getenv("TMPDIR");
  return std::string(d && *d ? d : "/tmp") + "/" + leaf;
}
}  // namespace bonsai_test

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++bonsai_test::failures();                                               \
    }                                                                          \
  } while (0)

#define CHECK_THROWS(expr)                                                     \
  do {                                                                         \
    bool threw_ = false;                                                       \
    try {                                                                      \
      (void)(expr);                                                            \
    } catch (const std::exception&) {                                          \
      threw_ = true;                                                           \
    }                                                                          \
    if (!threw_) {                                                             \
      std::printf("CHECK_THROWS failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      ++bonsai_test::failures();                                               \
    }                                                                          \
  } while (0)
