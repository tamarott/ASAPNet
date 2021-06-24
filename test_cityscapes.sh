#!/bin/bash
python3 test.py --name cityscapes --dataroot .fatasets/cityscapesHR/ --dataset_mode cityscapes --load_size 512 --crop_size 512 --batchSize 1 --gpu_ids 0 --phase test