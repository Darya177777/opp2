#!/bin/bash

#PBS -l select=1:ncpus=4:ompthreads=4
#PBS -l walltime=00:00:10
#PBS -m n
#PBS -o out-omp.txt
#PBS -e err-omp.txt

if [[ -n $PBS_O_WORKDIR ]]; then
  cd $PBS_O_WORKDIR || :
fi

echo "Using $OMP_NUM_THREADS threads"
gcc -std=gnu99 -O3 -Wno-unknown-pragmas -Wpedantic -Werror -Wextra -o omp main.c -fopenmp -lm

