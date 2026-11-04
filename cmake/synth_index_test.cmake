# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
# Generates the synthetic GGUF with make_synth_gguf and checks bonsai_index reads it.
execute_process(COMMAND ${GEN} ${OUT} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "make_synth_gguf failed")
endif()
execute_process(COMMAND ${INDEX} ${OUT} --list RESULT_VARIABLE rc OUTPUT_VARIABLE out)
message("${out}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "tensors       8," OR NOT out MATCHES "PTQ1_0")
  message(FATAL_ERROR "bonsai_index did not read the synthetic checkpoint")
endif()
