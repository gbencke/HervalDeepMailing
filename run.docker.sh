#!/bin/bash

if [ "$EUID" -ne 0 ]
then echo "Please run as root"
        exit
fi

sudo docker run --name hervaldeepmailing -v /home/gbencke/git/111.herval-deep-mailing/data:/home/hervaldeepmailing/data zanc/hervaldeepmailing:1
