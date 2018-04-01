#!/bin/bash

for S in least-confident; do
    #./caipi.py toy svm $S -p 0.0001 -T 101 -e 1 -S 200 -F 3
    ./caipi.py toy svm $S -p 0.0001 -T 101 -e 1 -S 200 -F 3 -E 0 -I
done
