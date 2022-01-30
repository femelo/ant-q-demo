#!/bin/bash
ffmpeg -r 24 -f image2 -i figures/ant-q-%04d.png -vcodec libx264 -crf 22 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  -pix_fmt yuv420p $1.mp4