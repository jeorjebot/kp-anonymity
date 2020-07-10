#!/bin/sh

DATA_DIR='../Dataset/Input/Facebook_Economy_final_'
OUT_DIR='../Dataset/Anonymized/' 
K='7'
P='2'
PAA='5'

DATASET_TO_SPLIT='../Dataset/Input/Facebook_Economy.csv'
DIM_DATASETS='100 250 500 1000 1500'
TYPE_DAT='.csv'
ALGS='naive kapra'
counter=0

# removing old anonymized output if any
# ./clean.sh
rm -f $OUT_DIR*.csv
rm -f $DATA_DIR*
rm -f 'tmp.txt'


# create datasets
python3 ../create_dataset.py $DATASET_TO_SPLIT $DIM_DATASETS

# test algs
for dim in $DIM_DATASETS; do
    counter=$((counter+1))
    for alg in $ALGS; do
        echo "\nProcessing Dataset:\tAlgorithm ==> $alg\tInstances ==> $dim"
        start=$(python2 -c 'import time; print time.time()')
        python3 ../kp-anonymity.py $alg $K $P $PAA $DATA_DIR$dim$TYPE_DAT $OUT_DIR$alg$counter$TYPE_DAT
        stop=$(python2 -c 'import time; print time.time()')
        elapsed=$(echo "$stop - $start" | bc)
        echo $alg $dim $elapsed >> 'tmp.txt'
    done
done

# draw stat
echo "\nComputing statistics..."
python3 ./draw_stat.py
echo "\nDone!"

# final clean
rm -f $OUT_DIR*.csv
rm -f $DATA_DIR*
rm -f 'tmp.txt'