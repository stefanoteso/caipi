#!/bin/bash

./caipi-draw.py colors-rule0-10folds \
    results/colors-rule0__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    --min-pred-f1 0.6

./caipi-draw.py colors-rule1-10folds \
    results/colors-rule0__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule1__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule1__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    --min-pred-f1 0.6

./caipi-draw-weights.py colors-ruleX__l1svm__least-confident__noei results/colors-rule0__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule0__l1svm__least-confident__ei results/colors-rule0__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule1__l1svm__least-confident__ei results/colors-rule1__l1svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=3__S=2000__K=0.75__R=100__s=0-params.pickle

./caipi-draw-weights.py colors-ruleX__svm__least-confident__noei results/colors-rule0__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=-1__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule0__svm__least-confident__ei results/colors-rule0__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule1__svm__least-confident__ei results/colors-rule1__svm__least-confident__k=10__n=None__p=0.0__P=5.0__T=101__e=-1__E=0__F=3__S=2000__K=0.75__R=100__s=0-params.pickle

exit

./caipi-draw.py colors-rule0 \
    results/colors-rule0__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule0__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=4__S=2000__K=0.75__R=100__s=0.pickle \
    --min-pred-f1 0.6

./caipi-draw.py colors-rule1 \
    results/colors-rule1__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule1__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule1__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    results/colors-rule1__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=3__S=2000__K=0.75__R=100__s=0.pickle \
    --min-pred-f1 0.6

./caipi-draw-weights.py colors-ruleX__l1svm__least-confident__noei results/colors-rule0__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule0__l1svm__least-confident__ei results/colors-rule0__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule1__l1svm__least-confident__ei results/colors-rule1__l1svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=3__S=2000__K=0.75__R=100__s=0-params.pickle

./caipi-draw-weights.py colors-ruleX__lr__least-confident__noei results/colors-rule1__lr__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=3__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule1__lr__least-confident__ei results/colors-rule1__lr__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=3__S=2000__K=0.75__R=100__s=0-params.pickle

./caipi-draw-weights.py colors-ruleX__svm__least-confident__noei results/colors-rule0__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=-1__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule0__svm__least-confident__ei results/colors-rule0__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=4__S=2000__K=0.75__R=100__s=0-params.pickle
./caipi-draw-weights.py colors-rule1__svm__least-confident__ei results/colors-rule1__svm__least-confident__k=3__n=None__p=0.0__P=5.0__T=101__e=20__E=0__F=3__S=2000__K=0.75__R=100__s=0-params.pickle
