#!/bin/bash

th model-1_layer-variable.lua -hidden 16 -batch 32 -rate 0.20 -iter 5 | tee model-1_layer-variable.out
