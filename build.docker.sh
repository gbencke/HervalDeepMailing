#!/bin/bash

if [ "$EUID" -ne 0 ]
then echo "Please run as root"
        exit
fi

sudo docker build -t zanc/hervaldeepmailing:1 .
