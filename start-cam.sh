#!/bin/sh

rpicam-vid --listen -o "tcp://127.0.0.1:54088" -n \
    --width 256 --height 256 --framerate 15 \
    --vflip 1 \
    --inline --codec h264 -t 0 \
