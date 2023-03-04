#!/bin/bash

#PBS -l select=1:ncpus=4:ompthreads=4:mem=8000,place=scatter
#PBS -l walltime=00:08:00
#PBS -m n
#PBS -o out-omp.txt
#PBS -e err-omp.txt

if [[ -n $PBS_O_WORKDIR ]]; then
  cd $PBS_O_WORKDIR || :
fi

echo "Using $OMP_NUM_THREADS threads"
./omp
