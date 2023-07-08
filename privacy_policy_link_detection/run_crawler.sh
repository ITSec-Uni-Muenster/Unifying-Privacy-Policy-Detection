#!/bin/bash
lscpu
free -h
killall Xvfb
killall python
killall firefox-bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openwpm
which python
Xvfb :99 -screen 0 1980x1020x24 &
export DISPLAY=:99
python demo_privacy_policy_download.py
killall Xvfb
killall python
killall firefox-bin
