#!/bin/bash

#https://www.kaggle.com/datasets/dennisfj/asl-citizen-processed

if [ $1 == "asl-citizen-processed" ]
then
    DATASET_NAME="asl-citizen-processed"
    DATASET_OWNER="dennisfj"
    DATASET_ID="asl-citizen-processed"
elif [ $1 == "wlasl" ]
then
    DATASET_NAME="wlasl-gesture-videos"
    DATASET_OWNER="dennisfj"
    DATASET_ID="wlasl-gesture-videos"
elif [ $1 == "wlasl-processed" ]
then
    DATASET_NAME="wlasl-processed"
    DATASET_OWNER="dennisfj"
    DATASET_ID="wlasl-processed"
else
    DATASET_NAME="$1"
    DATASET_OWNER="dennisfj"  # TODO: replace with your kaggle username
    DATASET_ID="$1"
    #echo "ERROR: Invalid dataset name: \"$1\". Expected either \"asl-citizen\" , \"wlasl-processed\", or \"wlasl\""
    #exit 1
fi

function download() {
    echo "Downloading $DATASET_NAME to data/$DATASET_NAME"
    kaggle datasets download -d $DATASET_OWNER/$DATASET_ID -p data/$DATASET_NAME
   
}

function clean () {
    if [ $DATASET_NAME == "asl-citizen-processed" ]
    then
        mv data/$DATASET_NAME/asl_citizen_processed/train data/$DATASET_NAME/train
        mv data/$DATASET_NAME/asl_citizen_processed/test data/$DATASET_NAME/test
        mv data/$DATASET_NAME/asl_citizen_processed/valid data/$DATASET_NAME/valid
        mv data/$DATASET_NAME/asl_citizen_processed/config.csv data/$DATASET_NAME/config.csv
        mv data/$DATASET_NAME/asl_citizen_processed/glosses.csv data/$DATASET_NAME/glosses.csv
        mv data/$DATASET_NAME/asl_citizen_processed/label_map.csv data/$DATASET_NAME/label_map.csv
        mv data/$DATASET_NAME/asl_citizen_processed/mp4_examples data/$DATASET_NAME/mp4_examples
        
        rmdir data/$DATASET_NAME/asl-citizen-processed
    elif [ $DATASET_NAME == "wlasl-gesture-videos" ]
    then
        mv data/$DATASET_NAME/asl_glosses/* data/$DATASET_NAME/
        
        rmdir data/$DATASET_NAME/asl_glosses
    elif [ $DATASET_NAME == "wlasl-processed" ]
    then
        mv data/$DATASET_NAME/asl_glosses_hard_processed/* data/$DATASET_NAME/
        
        rmdir data/$DATASET_NAME/asl_glosses_hard_processed
    fi
    elif [ $DATASET_NAME == "$1" ]
        mv data/$DATASET_NAME/$DATASET_NAME/* data/$DATASET_NAME/
        
        rmdir data/$DATASET_NAME/$DATASET_NAME
    fi

}

echo "WARNING: Your Python virtual environment should be enabled while running this script!"
echo "Current Python Environment: $(which python)"

if [ -z "${KAGGLE_CONFIG_DIR}" ]; then
    export KAGGLE_CONFIG_DIR=secrets
fi

if [ -d "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "ERROR: Cannot find kaggle.json in $KAGGLE_CONFIG_DIR"
    exit 1
fi

if ! [ -d "data" ]; then
    echo "ERROR: Cannot find data directory in current working directory $(pwd). Make sure to run this script in the root directory of the repository"
    exit 1
fi

# install the kaggle package
pip show kaggle &>/dev/null
if [ $? != 0 ]; then
    echo "Installing the kaggle package in Python environment"
    pip install -q kaggle
fi

count=$(ls data/$DATASET_NAME | wc -l)

# download and extract the dataset
if ! [ -d data/$DATASET_NAME ] || [ $count = "0" ]; then
    mkdir -p data/$DATASET_NAME
    download

    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/$DATASET_ID.zip -d data/$DATASET_NAME
    #clean
elif [ $count = "1" ] && [ -f data/$DATASET_NAME/$DATASET_ID.zip ]
then
    echo "Extracting $DATASET_NAME - this may take some time!"
    unzip data/$DATASET_NAME/$DATASET_ID.zip -d data/$DATASET_NAME
    clean
else
    echo "Using cached version of $DATASET_NAME"
fi
