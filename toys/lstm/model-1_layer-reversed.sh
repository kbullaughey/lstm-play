#!/bin/bash

th model-1_layer.lua -data '../toy/variable_width_2-4-reversed.t7' -hidden 16 \
    -batch 32 -rate 0.15 -iter 5 -trained 'trained_model-1_layer-reversed.t7' \
    -grid 'grid_predictions-1_layer-reversed.csv' \
  | tee model-1_layer-reversed.out
