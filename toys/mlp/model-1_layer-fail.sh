#!/bin/bash

./model-1_layer.lua -seed 1 -hidden 152 -batch 20 -rate 0.06 -iter 50 \
  -grid 'grid_predictions-1_layer-fail.csv' -trained 'trained_model-1_layer-fail.t7'
