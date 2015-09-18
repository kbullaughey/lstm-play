#!/bin/bash

./model-1_layer.lua -seed 1 -hidden 152 -batch 20 -rate 0.05 -iter 50 | tee model-1_layer.out
