#!/bin/bash

for M in 100 200 400 800 1600; do
  for maps in 1 10 100; do
    ./bench_linear_parallel.lua -M $M -reps 10000 -maps $maps -gpu
    ./bench_linear_parallel.lua -M $M -reps 10000 -maps $maps
  done
done
