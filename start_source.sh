#!/bin/bash

#Create a virtualenv if it doesnt exist
if [ ! -d "./env" ]; then
        virtualenv -p python3 env
fi

#Load the virtualenv created
source env/bin/activate

#Install the required python modules
pip install -r requirements.txt

export HERVAL_FROM_SOURCE=1
export HERVAL_DATA_FOLDER='../'

cd output.source

python ./start_herval.py

