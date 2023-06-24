# Audio Generation Evaluation

This toolbox aims to unify audio generation model evaluation for easier future comparison.

## Quick Start

First, prepare the environment
```shell
pip install git+https://github.com/haoheliu/audioldm_eval
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

- Recommanded:
  - FAD: Frechet audio distance
  - ISc: Inception score
- Other for references:
  - FD: Frechet distance, realized by PANNs, a state-of-the-art audio classification model
  - KID: Kernel inception score
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

## Evaluation on AudioCaps and AudioSet

The AudioCaps test set consists of audio files with multiple text annotations. To evaluate the performance of AudioLDM, we randomly selected one annotation per audio file, which can be found in the [accompanying json file](https://github.com/haoheliu/audioldm_eval/tree/c9e936ea538c4db7e971d9528a2d2eb4edac975d/example/AudioCaps).

Given the size of the AudioSet evaluation set with approximately 20,000 audio files, it may be impractical for audio generative models to perform evaluation on the entire set. As a result, we randomly selected 2,000 audio files for evaluation, with the corresponding annotations available in a [json file](https://github.com/haoheliu/audioldm_eval/tree/c9e936ea538c4db7e971d9528a2d2eb4edac975d/example/AudioSet).

For more information on our evaluation process, please refer to [our paper](https://arxiv.org/abs/2301.12503).

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

## Note

- Update on 24 June 2023: 
  - **Issues on model evaluation:** I found the PANNs based Frechet Distance and KL score is not as robust as FAD sometimes. For example, when the generation are all silent audio, the FAD and KL still indicate model perform very well, while FAD and Inception Score (IS) can still reflect the model true bad performance. Sometimes the resample method on audio can significantly affect the FD (+-30) and KL (+-0.4) performance as well.
    - To address this issue, in another branch of this repo ([passt_replace_panns](https://github.com/haoheliu/audioldm_eval/tree/passt_replace_panns)), I change the PANNs model to Passt, which I found to be more robust to resample method and other trival mismatches.

  - **Update on code:** The calculation of FAD is slow. Now, after each calculation of a folder, the code will save the FAD feature into an .npy file for later reference. 

## TODO

- [ ] Add pretrained AudioLDM model.
- [ ] Add CLAP score

## Cite this repo

If you found this tool useful, please consider citing
```bibtex
@article{liu2023audioldm,
  title={AudioLDM: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2301.12503},
  year={2023}
}
```

## Reference

> https://github.com/toshas/torch-fidelity

> https://github.com/v-iashin/SpecVQGAN 
