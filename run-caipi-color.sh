#!/bin/bash

for L in l1svm svm lr; do

    # No corrections
    for S in random least-confident; do
        ./caipi.py colors-rule0 $L $S -k 10 -p 0 -P 5 -T 101 -e 20 -E -1 -F 4 -S 2000 -R 100
        ./caipi.py colors-rule1 $L $S -k 10 -p 0 -P 5 -T 101 -e 20 -E -1 -F 3 -S 2000 -R 100
    done

    # With corrections
    for S in random least-confident; do
        ./caipi.py colors-rule0 $L $S -k 10 -p 0 -P 5 -T 101 -e 20 -E 0 -F 4 -S 2000 -R 100
        ./caipi.py colors-rule1 $L $S -k 10 -p 0 -P 5 -T 101 -e 20 -E 0 -F 3 -S 2000 -R 100
    done

done
