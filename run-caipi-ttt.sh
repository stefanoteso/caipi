#!/bin/bash

for S in random least-confident; do
    ./caipi.py ttt svm $S -p 0.01 -S 10000 -F 3 -S 15000
done

for S in random least-confident; do
    ./caipi.py ttt svm $S -p 0.01 -S 10000 -F 3 -S 15000 -E 0 -I
done
