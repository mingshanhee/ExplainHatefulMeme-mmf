export MMF_DATA_DIR=/opt/datasets/mmf

MODEL=hateful_memes

#!/bin/bash
CUDA_VISIBLE_DEVICES=0 mmf_run \
    dataset=hateful_memes \
    model=visual_bert \
    config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
    checkpoint.resume_file=visual_bert.finetuned.hateful_memes.from_coco \
    checkpoint.resume_pretrained=False \
    run_type=val \
    training.batch_size=1 \
    env.captum_dir=./captum_outputs/${MODEL}/visual_bert_coco \
    model_config.visual_bert.output_attentions=True \

