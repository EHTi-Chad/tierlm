@echo off
rem Second round of the Qwen3.5 GPU diagnostic: per prompt-length errors, then more op types moved to the CPU.
setlocal
set PY=C:\dev\tierlm-oracle\Scripts\python
set T=tests\tiny_qwen35.py build\tierlm-logits.exe
set TIERLM_DUMP_SCHED=
set TIERLM_DEVICE=
echo === GPU, error for each prompt length (exact cache)
set TIERLM_VERBOSE=1
%PY% %T% | findstr /c:"tokens:" /c:"exact"
set TIERLM_VERBOSE=
for %%O in (MUL_MAT CONT MUL ADD SILU SIGMOID SCALE GET_ROWS) do (
  echo === GPU, with %%O on the CPU
  set TIERLM_CPU_OPS=%%O
  %PY% %T% | findstr /c:"exact"
)
set TIERLM_CPU_OPS=
echo === GPU, with everything in the DeltaNet path on the CPU except matrix multiplies
set TIERLM_CPU_OPS=GATED_DELTA_NET,SSM_CONV,CONCAT,RMS_NORM,SCALE,SILU,SIGMOID,SOFTPLUS,MUL,ADD,CONT
%PY% %T% | findstr /c:"exact"
set TIERLM_CPU_OPS=
endlocal
