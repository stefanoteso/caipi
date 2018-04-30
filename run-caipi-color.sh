#!/bin/bash

for L in l1svm svm lr; do

    for S in random least-confident; do
        ./caipi.py colors-rule1 l1svm $S -k 3 -p 0 -P 5 -T 101 -e 20 -E -1 -F 4 -S 2000 -R 100
        ./caipi.py colors-rule1 l1svm $S -k 3 -p 0 -P 5 -T 101 -e 20 -E -1 -F 3 -S 2000 -R 100
    done

    for S in random least-confident; do
        ./caipi.py colors-rule1 l1svm $S -k 3 -p 0 -P 5 -T 101 -e 20 -E 0 -F 4 -S 2000 -R 100
        ./caipi.py colors-rule1 l1svm $S -k 3 -p 0 -P 5 -T 101 -e 20 -E 0 -F 3 -S 2000 -R 100
    done

done
