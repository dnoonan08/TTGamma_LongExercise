#!/usr/bin/env bash

tar -zxf ttgenv.tar.gz
source ttgenv/bin/activate

tar -zxf ttgamma.tgz

python runFullDataset.py $1 --condor

