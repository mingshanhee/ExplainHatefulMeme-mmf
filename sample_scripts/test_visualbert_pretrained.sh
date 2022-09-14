export MMF_DATA_DIR=/opt/datasets/mmf

MODEL=hateful_memes

#!/bin/bash
CUDA_VISIBLE_DEVICES=0 mmf_run \
    dataset=hateful_memes \
    model=vilbert \
    config=projects/hateful_memes/configs/vilbert/from_cc.yaml \
    checkpoint.resume_file=vilbert.finetuned.hateful_memes.from_cc \
    checkpoint.resume_pretrained=False \
    run_type=val \
    training.batch_size=1 \
    env.captum_dir=./captum_outputs/${MODEL}/vilbert_cc \
    model_config.vilbert.output_attentions=True \

