# Audio Generation Evaluation

This tool box is for the evaluation of audio generation model. 

We hope this toolbox can unify the evaluation of audio generation model for easier comparisons between different methods.

## Quick Start

First, prepare the environment
```shell
git clone git@github.com:haoheliu/audioldm_eval.git
cd audioldm_eval
pip install -e .
```

Second, generate test dataset by
```python
python3 gen_test_file.py
```

Finally, perform a test run
```
python3 test.py
```

## Evaluation metrics
We have the following metrics in this toolbox:
- FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model.
- FAD: Frechet audio distance.
- ISc: Inception Score.
- KID: Kernel Inception Score.
- KL: KL divergence (softmax over logits)
- KL_Sigmoid: KL divergence (sigmoid over logits)
- PSNR: Peak Signal Noise Ratio
- SSIM: Structural similarity index measure
- LSD: Log-spectral distance

The evaluation function will accept the paths to two folders as main parameters. 
1. If two folder have files with same name, the evaluation will run in paired mode.
2. If two folder have different numbers of files or files with different name, the evaluation will run in unpaired mode.

These metrics will only be calculated in paried mode: KL, KL_Sigmoid, PSNR, SSIM, LSD, otherwise these metrics will return minus one.


## Example

```python
import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

generation_result_path = "example/paired"
target_audio_path = "example/reference"

# Initialize a helper instance
evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)
```