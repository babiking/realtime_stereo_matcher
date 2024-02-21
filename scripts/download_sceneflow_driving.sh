#!/bin/bash

WORK_PATH=`pwd`
DATA_PATH="/datasets02/aistereo"

echo "[START] SceneFlow/Driving dataset downloading..."
mkdir -p ${DATA_PATH}/SceneFlow/Driving
cd ${DATA_PATH}/SceneFlow/Driving
wget -t 0 https://academictorrents.com/download/ea392433e3dfcb4b83dcd3300dfa9b9919ef8e1f.torrent
transmission-cli ea392433e3dfcb4b83dcd3300dfa9b9919ef8e1f.torrent -w ${DATA_PATH}/SceneFlow/Driving

wget -t 0 https://academictorrents.com/download/1d642a371312d193ae4523e089bf127917294175.torrent
transmission-cli 1d642a371312d193ae4523e089bf127917294175.torrent -w ${DATA_PATH}/SceneFlow/Driving
echo "[FINISH] SceneFlow/Driving dataset downloaded."
cd ${WORK_PATH}