#!/usr/bin/env bash
export WD='/Users/aripiralasrinivas/Data Science/repos/mlframework/'

export TRAINING_DATA="$WD/input/train_median_imputed_5fold.csv"
export TEST_DATA="$WD/input/test_median_imputed.csv"

#export TRAINING_DATA=$TRAINING_DATA
#export TEST_DATA=$TEST_DATA
export TARGET=SalePrice
export NUM_FOLDS=5

export MODEL=randomforest
export SCALAR=min_max_scalar

python create_folds.py -i train_median_imputed.csv -o train_median_imputed_5fold.csv
for fold in {0..4}
do
#    echo "Computing for fold $fold"
    python train.py -f $fold
done

#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
#python -m src.predict