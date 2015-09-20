#!/bin/bash

th model-1_layer.lua -data '../toy/variable_width_2-4-200k.t7' -hidden 16 \
    -batch 16 -rate 0.15 -iter 5 -trained 'trained_model-1_layer-200k' \
    -grid 'grid_predictions-1_layer-200k' \
  | tee model-1_layer-200k.out
