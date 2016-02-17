#!/bin/bash

th model-1_layer-variable.lua -hidden 16 -batch 16 -rate 0.10 -iter 12 \
  | tee model-1_layer-variable.out
