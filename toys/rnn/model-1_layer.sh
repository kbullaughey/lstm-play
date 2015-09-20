#!/bin/bash

th model-1_layer.lua -hidden 26 -batch 20 -rate 0.02 -iter 20 | tee model-1_layer.out
