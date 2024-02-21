#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/datasets02/aistereo"

echo "[START] SceneFlow/FlyingThings3D dataset downloading..."
# rm ~/.config/transmission/resume/torrentyouwanttorestart.resume
mkdir -p ${DATA_PATH}/SceneFlow/FlyingThings3D
cd ${DATA_PATH}/SceneFlow/FlyingThings3D
wget -t 0 https://academictorrents.com/download/20afbe18b5d1754b75deeefe4c2c74b17c9ea792.torrent
transmission-cli 20afbe18b5d1754b75deeefe4c2c74b17c9ea792.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D

wget -t 0 https://academictorrents.com/download/3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent
transmission-cli 3221ff49a08f5e6749f24958c1f76248fe9cb884.torrent -w ${DATA_PATH}/SceneFlow/FlyingThings3D
echo "[FINISH] SceneFlow/FlyingThings3D dataset downloaded."
cd ${WORK_PATH}