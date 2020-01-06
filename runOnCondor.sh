#!/usr/bin/env bash

tar -zxf coffeaenv.tar.gz
source coffeaenv/bin/activate

tar -zxf ttgamma.tgz

python runFullDataset.py $1

