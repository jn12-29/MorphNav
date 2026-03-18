#!/bin/bash

conda create -n mz python=3.12 -y

conda run -n mz pip install -r requirements.txt

git clone https://github.com/jn12-29/rl-baselines3-zoo.git
cd rl-baselines3-zoo/
apt-get install swig cmake ffmpeg
pip install -e .