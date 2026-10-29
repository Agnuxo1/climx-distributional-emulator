#!/bin/bash
# Predicción con generación paralela y malloc que devuelve memoria al sistema. Uso: bash predict_go3.sh <tag> <miembros> <procs>
set -u
. /home/ai/proyectos/climx/.venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
python run_sota.py predict --members $2 --workers 1 --tag $1 --set gen_procs=$3
