import yaml
import argparse
from audiogen_eval import EvaluationHelper
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--generation_result_path",
    type=str,
    required=False,
    help="Audio sampling rate during evaluation",
    default="/mnt/fast/datasets/audio/audioset/2million_audioset_wav/balanced_train_segments",
)
parser.add_argument(
    "-t",
    "--target_audio_path",
    type=str,
    required=False,
    help="Audio sampling rate during evaluation",
    default="/mnt/fast/datasets/audio/audioset/2million_audioset_wav/eval_segments",
)
parser.add_argument(
    "-sr",
    "--sampling_rate",
    type=int,
    required=False,
    help="Audio sampling rate during evaluation",
    default=16000,
)
parser.add_argument(
    "-l",
    "--limit_num",
    type=int,
    required=False,
    help="Audio clip numbers limit for evaluation",
    default=500,
)
parser.add_argument(
    "--same_name",
    action='store_true'
)
args = parser.parse_args()

device = torch.device(f"cuda:{0}")

evaluator = EvaluationHelper(args.sampling_rate, device)

metrics = evaluator.main(args.generation_result_path, args.target_audio_path, limit_num=args.limit_num, same_name=args.same_name)
print(metrics)
