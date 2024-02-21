#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/datasets02/aistereo"

echo "[START] KITTI scene-flow-2015 dataset downloading..."
mkdir -p ${DATA_PATH}/KITTI
cd ${DATA_PATH}/KITTI
wget -t 0 https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
unzip data_scene_flow.zip
rm data_scene_flow.zip
echo "[FINISH] KITTI scene-flow-2015 dataset downloaded."
cd ${WORK_PATH}