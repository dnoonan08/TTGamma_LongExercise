#!/usr/bin/env bash
NAME=ttgenv
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt

source ${LCG}/setup.sh
# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
python -m venv --copies $NAME
source $NAME/bin/activate
python -m pip install setuptools pip --upgrade
python -m pip install "uproot4<1"
python -m pip install "coffea==0.6.47"
python -m pip install xxhash

sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i "2a source ${LCG}/setup.sh" $NAME/bin/activate
sed -i "4a source ${LCG}/setup.csh" $NAME/bin/activate.csh
ipython kernel install --user --name=$NAME
tar -zcf ${NAME}.tar.gz ${NAME}
