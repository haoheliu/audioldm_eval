import torch
from audioldm_eval import EvaluationHelper, EvaluationHelperParallel
import torch.multiprocessing as mp

device = torch.device(f"cuda:{0}")

generation_result_path = "example/paired"
# generation_result_path = "example/unpaired"
target_audio_path = "example/reference"

## Single GPU

evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)

## Multiple GPUs

if __name__ == '__main__':    
    evaluator = EvaluationHelperParallel(16000, 2)
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )