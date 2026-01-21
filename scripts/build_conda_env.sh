#!/bin/bash

conda create -n mz python=3.12 -y

conda run -n mz pip install -r requirements.txt