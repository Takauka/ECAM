#!/usr/bin/env sh

echo "Download extra data"

wget -O ST-extra.tar.gz https://github.com/CVML-CFU/ECAM/releases/download/v1.0.0/ST-extra.tar.gz
tar -xzf ST-extra.tar.gz
mv ST-extra/checkpoints SingularTrajectory/checkpoints
mv ST-extra/config SingularTrajectory/config
mv ST-extra/datasets SingularTrajectory/datasets
rm -rf ST-extra.tar.gz ST-extra

echo "Done"
