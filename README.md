# Audio Generation Evaluation

This toolbox aims to unify audio generation model evaluation for easier future comparison.

## Quick Start

First, prepare the environment
```shell
git clone git@github.com:haoheliu/audioldm_eval.git
cd audioldm_eval
pip install -e .
```

Second, generate test dataset by
```shell
python3 gen_test_file.py
```

Finally, perform a test run. A result for reference is attached [here](https://github.com/haoheliu/audioldm_eval/blob/main/example/paired_ref.json).
```shell
python3 test.py # Evaluate and save the json file to disk (example/paired.json)
```

## Evaluation metrics
We have the following metrics in this toolbox: 

- FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model.
- FAD: Frechet audio distance.
- ISc: Inception score.
- KID: Kernel inception score.
- KL: KL divergence (softmax over logits)
- KL_Sigmoid: KL divergence (sigmoid over logits)
- PSNR: Peak signal noise ratio
- SSIM: Structural similarity index measure
- LSD: Log-spectral distance

The evaluation function will accept the paths of two folders as main parameters. 
1. If two folder have **files with same name and same numbers of files**, the evaluation will run in **paired mode**.
2. If two folder have **different numbers of files or files with different name**, the evaluation will run in **unpaired mode**.

**These metrics will only be calculated in paried mode**: KL, KL_Sigmoid, PSNR, SSIM, LSD. 
In the unpaired mode, these metrics will return minus one.


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
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)
```
