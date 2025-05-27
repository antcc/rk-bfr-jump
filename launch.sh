#!/bin/bash

KIND=$1  # 'linear' or 'logistic'
DATA=$2  # 'rkhs', 'l2', 'gbm', 'mixture', 'real'
KERNEL=$3  # 'bm', 'fbm', 'ou', 'sqexp', 'gbm', or 'homo/heteroscedastic'
SEED=$4

if [ "$DATA" != "real" ]; then
	DATA_NAME="--kernel ${KERNEL}"
else
	DATA_NAME="--data-name ${KERNEL}"
fi

python experiments.py \
	${KIND} ${DATA} ${DATA_NAME} \
	--seed ${SEED} \
	--nreps 3 \
	--nwalkers 32 --ntemps 5 --nleaves-max 5 \
	--nsteps 500 --nburn 500 --num-try 2 \
	--scale-prior-beta 2.5 --lambda-p 3 \
	--verbose 2
#	--prediction-noise
