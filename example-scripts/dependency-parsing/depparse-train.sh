#!/bin/bash
# Name of the job
#SBATCH -J gal-dep-parse
# time: 10 hours
#SBATCH --time=10:0:0
# Number of GPU
#SBATCH --gres=gpu:rtx_4090:1
# Start your application
python3 login.py
python -u -m supar.cmds.dep.biaffine train -d 0 -c dep-biaffine-xlmr -p dep/gal/pt/model \
    --train gl_ctg-ud-train.conllu  \
    --dev gl_ctg-ud-dev.conllu  \
    --test gl_ctg-ud-test.conllu  \
    --encoder=bert  \
    --bert=homersimpson/subsec-xlm-roberta-portuguese-30k  \
    --lr=2e-5  \
    --lr-rate=20  \
    --seed=86\
    --batch-size=16  \
    --epochs=4  \
    --update-steps=1