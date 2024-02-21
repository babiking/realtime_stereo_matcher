#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/datasets02/aistereo"

echo "[START] ETH3D dataset downloading..."
mkdir -p ${DATA_PATH}/ETH3D/two_view_training
cd ${DATA_PATH}/ETH3D/two_view_training
wget https://www.eth3d.net/data/two_view_training.7z
echo "Unzipping two_view_training.7z using p7zip (installed from environment.yaml)"
7za x two_view_training.7z
rm two_view_training.7z

mkdir -p ${DATA_PATH}/ETH3D/two_view_training_gt
cd ${DATA_PATH}/ETH3D/two_view_training_gt
wget https://www.eth3d.net/data/two_view_training_gt.7z
echo "Unzipping two_view_training_gt.7z using p7zip (installed from environment.yaml)"
7za x two_view_training_gt.7z
rm two_view_training_gt.7z

mkdir -p ${DATA_PATH}/ETH3D/two_view_testing
cd ${DATA_PATH}/ETH3D/two_view_testing
wget https://www.eth3d.net/data/two_view_test.7z
echo "Unzipping two_view_test.7z using p7zip (installed from environment.yaml)"
7za x two_view_test.7z
rm two_view_test.7z
cd ${WORK_PATH}
echo "[FINISH] ETH3D dataset downloaded."