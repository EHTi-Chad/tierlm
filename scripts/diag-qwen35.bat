@echo off
rem Diagnostic for the Qwen3.5 (qwen35) GPU path: runs the reference test on the CPU, on the GPU, and on the GPU
rem with one suspect operation type moved to the CPU at a time. The suspect that brings the error down is the
rem GPU kernel that disagrees with the reference.
setlocal
set PY=C:\dev\tierlm-oracle\Scripts\python
set T=tests\tiny_qwen35.py build\tierlm-logits.exe
set TIERLM_DUMP_SCHED=
echo === CPU only
set TIERLM_DEVICE=cpu
%PY% %T% | findstr /c:"exact"
set TIERLM_DEVICE=
echo === GPU (automatic)
%PY% %T% | findstr /c:"exact"
for %%O in (GATED_DELTA_NET SSM_CONV RMS_NORM CONCAT CPY FLASH_ATTN_EXT ROPE SOFTPLUS) do (
  echo === GPU, with %%O on the CPU
  set TIERLM_CPU_OPS=%%O
  %PY% %T% | findstr /c:"exact"
)
set TIERLM_CPU_OPS=
endlocal
