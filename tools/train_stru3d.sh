#!/usr/bin/env bash

python main.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --epochs=500 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_stru3d \
               --angles_loss_coef=1 \
               --raster_loss_coef=1 \
               --uniform_loss_coef=0 \
               --output_dir= \
               --batch_size=40 \
               