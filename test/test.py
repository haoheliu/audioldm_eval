import yaml
import argparse
from audioldm_eval import EvaluationHelper
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "-g",
    "--generation_result_path",
    type=str,
    required=False,
    help="Audio sampling rate during evaluation",
    default="./test/test_fad/test1",
)

parser.add_argument(
    "-t",
    "--target_audio_path",
    type=str,
    required=False,
    help="Audio sampling rate during evaluation",
    default="./test/test_fad/test2",
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
    default=None,
)

args = parser.parse_args()

assert args.sampling_rate == 16000, "We only support 16000Hz sampling rate currently"

device = torch.device(f"cuda:{0}")

evaluator = EvaluationHelper(args.sampling_rate, device)

metrics = evaluator.main(
    args.generation_result_path,
    args.target_audio_path,
    limit_num=args.limit_num,
)

print(metrics)
