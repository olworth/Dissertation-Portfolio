#!/bin/bash
#SBATCH -J evalquick
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:rtx_6000_ada:1
python -u -m supar.cmds.dep.biaffine evaluate -d 0 -p /mnt/nas_home/och26/dep/gal/pt/model --data gl_ctg-ud-test.conllu --tree