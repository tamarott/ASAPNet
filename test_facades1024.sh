#!/bin/bash
python3 test.py --name facades1024 --dataroot ./datasets/facadesHR --dataset_mode facades --load_size 1024 --crop_size 1024 --batchSize 1  --gpu_ids 0 --phase test --reflection_pad
