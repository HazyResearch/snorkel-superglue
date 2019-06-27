#!/bin/bash

DATA_DIR=$1

echo "Downloading primary SuperGLUE tasks."
python download_superglue_data.py --data_dir $DATA_DIR --tasks all
echo "Downloading and unzipping SWAG."
wget -P $DATA_DIR https://www.dropbox.com/s/cklxrrzisd3zzuh/SWAG.zip?dl=1
mv ${DATA_DIR}/SWAG.zip?dl=1 ${DATA_DIR}/SWAG.zip
unzip -o ${DATA_DIR}/SWAG.zip -d ${DATA_DIR}/SWAG
rm ${DATA_DIR}/SWAG.zip
echo "Done."
