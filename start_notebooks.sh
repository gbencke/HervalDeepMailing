#!/bin/bash

#Create a virtualenv if it doesnt exist
if [ ! -d "./env" ]; then
        virtualenv -p python3 env
fi

#Load the virtualenv created
source env/bin/activate

#Install the required python modules
pip install -r requirements

#Start the jupyter lab...
cd notebook
jupyter lab --ip='*' --port=8080
