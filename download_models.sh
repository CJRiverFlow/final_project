#! /bin/bash
#Note, execute from project folder.

MODELSDIR="$(pwd)/models/"

if [ ! -d "$MODELSDIR" ] 
then
    mkdir models
    echo "Created path: '$MODELSDIR'"
fi

echo "Downloading models to: '$MODELSDIR'"

cd /opt/intel/openvino/deployment_tools/tools/model_downloader

#Models needed in the project
./downloader.py --name face-detection-adas-binary-0001 --output_dir $MODELSDIR 
./downloader.py --name head-pose-estimation-adas-0001 --output_dir $MODELSDIR
./downloader.py --name landmarks-regression-retail-0009 --output_dir $MODELSDIR
./downloader.py --name gaze-estimation-adas-0002 --output_dir $MODELSDIR

