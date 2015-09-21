#!/bin/bash

th model-1_layer-variable.lua -hidden 17 -batch 16 -rate 0.10 -iter 8 | tee model-1_layer-variable.out
