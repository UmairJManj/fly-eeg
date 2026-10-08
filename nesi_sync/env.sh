# Shared environment for fly-eeg on NeSI. Home is full, so every cache lives in the project.
ROOT=/nesi/project/aut04653/Manj/Fly
module purge 2>/dev/null
module load uv/0.10.3-GCC-12.3.0 CUDA/12.6.3 2>/dev/null
export UV_CACHE_DIR=$ROOT/.uv-cache
export UV_PYTHON_INSTALL_DIR=$ROOT/.uv-python
export XDG_CACHE_HOME=$ROOT/.cache
export HF_HOME=$ROOT/.cache/hf
export PYTHONUNBUFFERED=1
export NFLY_CSR=1   # cuSPARSE CSR recurrent product (validated 2026-10-02 job 9459475: grads match 1e-6, 2.7x faster)
