import yaml
import argparse
from audioldm_eval import EvaluationHelper
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "-g",
    "--generation_result_path",
    type=str,
    required=True,
    help="Audio sampling rate during evaluation",
)

args = parser.parse_args()

folder = args.generation_result_path

for subfolder in os.listdir(folder):
    path = os.path.join(folder, subfolder)
    if not os.path.isdir(path):
        continue
    elif not len(os.listdir(path)) == 964:
        continue
    elif os.path.exists(path + ".json"):
        continue
    else:
        print("Evaluating %s" % subfolder)
        cmd = (
            "python3 /mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/audioldm_eval/test/test.py -g %s -t /mnt/fast/nobackup/users/hl01486/datasets/audiocaps_test_subset/0"
            % path
        )
        print(cmd)
        os.system(cmd)
