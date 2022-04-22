#!/bin/sh

for filename in $(ls /home/iiitd/Desktop/Ahmad/datasets/doa1/)
do
  python ./doa1/train.py "$filename" "$filename"
done

