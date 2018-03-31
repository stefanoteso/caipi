#!/bin/bash

N_SAMPLES=2000 # make LIME stable enough

for S in random; do
    ./caipi.py colors-rule0 svm $S -T 70 -p 0.01 -S $N_SAMPLES -F 4
    ./caipi.py colors-rule1 svm $S -T 70 -p 0.01 -S $N_SAMPLES -F 4
done

for S in random; do
    ./caipi.py colors-rule0 svm $S -T 70 -p 0.01 -S $N_SAMPLES -F 4 -E 0 -I
    ./caipi.py colors-rule1 svm $S -T 70 -p 0.01 -S $N_SAMPLES -F 4 -E 0 -I
done
