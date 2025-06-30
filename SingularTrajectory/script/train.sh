#!/bin/bash

# Hyperparameters
tag="SingularTrajectory-stochastic"
config_path="./config/"
config_prefix="stochastic/singulartrajectory"
baseline="transformerdiffusion"
variant="orig"
device="gpu"

# Arguments
while getopts t:b:c:p:d:v:g: flag
do
  case "${flag}" in
    t) tag=${OPTARG};;
    b) baseline=${OPTARG};;
    c) config_path=${OPTARG};;
    p) config_prefix=${OPTARG};;
    d) dataset=${OPTARG};;
    v) variant=${OPTARG};;
    g) device=${OPTARG};;
    *) echo "usage: $0 -t TAG -b BASELINE -p CONFIG_PREFIX -d {eth|hotel|univ|zara1|zara2|sdd|pfsd|thor -v {orig|map|ecam} -g {cpu|gpu}" >&2
      exit 1 ;;
  esac
done

python3 trainval.py \
  --cfg "${config_path}""${config_prefix}"-"${baseline}"-"${dataset}"-"${variant}".json \
  --tag "${tag}"-"${dataset}"-"${variant}" \
  --device ${device}
