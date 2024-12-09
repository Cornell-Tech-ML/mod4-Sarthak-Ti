#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=100G

source ~/.bashrc

cd /data1/lesliec/sarthak/mod4-Sarthak-Ti/

pixi run python project/run_sentiment.py