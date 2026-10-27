#!/usr/bin/env bash
# Adds the OLMoE model code and the two new tools to CMakeLists.txt (safe to run twice).
set -e
grep -q 'src/olmoe.cpp' CMakeLists.txt || sed -i 's|src/tokenizer.cpp)|src/tokenizer.cpp src/olmoe.cpp)|' CMakeLists.txt
grep -q 'tierlm-logits' CMakeLists.txt || printf '\nadd_executable(tierlm-logits tools/logits.cpp)\ntarget_link_libraries(tierlm-logits PRIVATE tierlm_core)\n' >> CMakeLists.txt
grep -q 'tierlm-run' CMakeLists.txt || printf '\nadd_executable(tierlm-run tools/run.cpp)\ntarget_link_libraries(tierlm-run PRIVATE tierlm_core)\n' >> CMakeLists.txt
grep -q 'src/olmoe.cpp' CMakeLists.txt && echo "CMakeLists.txt updated"
