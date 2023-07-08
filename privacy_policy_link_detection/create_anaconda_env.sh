#!/bin/bash
sudo apt update 
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
sudo apt-get install firefox xvfb wget git make git-all libgtk-3-dev npm -y
git clone https://github.com/openwpm/OpenWPM.git
cd OpenWPM
chmod +x ./install.sh
./install.sh
conda activate openwpm
which python
pip install ndjson tranco chardet urljoin