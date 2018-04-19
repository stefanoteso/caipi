#!/bin/bash

for PROBLEM in toy-fst toy-lst; do
    for L in lr svm l1svm; do
        for S in least-confident; do
            ./caipi.py $PROBLEM $L $S -p 0 -P 1 -k 3 -T 101 -e 5 -S 1000 -F 2 -E 0 2>/dev/null &
            ./caipi.py $PROBLEM $L $S -p 0 -P 1 -k 3 -T 101 -e 5 -S 1000 -F 2 -E 0 -I 2>/dev/null &
            wait
        done
    done
done
