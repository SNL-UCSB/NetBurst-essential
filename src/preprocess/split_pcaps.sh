#!/bin/bash
if [ $# -ne 3 ]
then
    # echo "Usage: $0 input_directory output_directory seconds_per_file"
    echo "Usage: $0 input_directory output_directory packets_per_file"
    exit 1
fi
export IN_DIR=$1
export OUT_DIR=$2
# export SECONDS_PER_FILE=$3
export PACKETS_PER_FILE=$3
mkdir -p $OUT_DIR
# ls * | parallel -j 56 "editcap -A \"2021-10-17 16:02:00\" -B \"2021-10-17 16:12:00\" {} tmp_{}"
# ls $IN_DIR/* | parallel -j 56 'editcap -i $SECONDS_PER_FILE {} $OUT_DIR/$(basename {})'
ls $IN_DIR/* | parallel -j 128 'editcap -c $PACKETS_PER_FILE {} $OUT_DIR/$(basename {})'
