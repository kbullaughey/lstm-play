#!/bin/bash

th model-1_layer-variable-direct.lua -hidden 16 -batch 16 -rate 0.10 -iter 5 \
  | tee model-1_layer-variable-direct.out
