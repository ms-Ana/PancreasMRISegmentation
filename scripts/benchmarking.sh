#!/bin/bash
#SBATCH --job-name=pancreas_benchmark       
#SBATCH --output=logs/bench_%j.log         
#SBATCH --error=logs/bench_%j.err         
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --partition=owner_fb12
#SBATCH --mem-per-cpu=6G                        
#SBATCH --time=08:00:00                    

mkdir -p logs

CONFIG_FILE=${1}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

pdm run benchmark --config "$CONFIG_FILE"