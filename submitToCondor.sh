# source me
mkdir -p condorOutputs
tar -zcf ttgamma.tgz ttgamma
condor_submit submitToCondor.jdl
