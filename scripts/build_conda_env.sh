#!/bin/bash

conda create -n mz python=3.12 -y

conda run -n mz pip install -r requirements.txt

git clone https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo/
apt-get install swig cmake ffmpeg
pip install -e .