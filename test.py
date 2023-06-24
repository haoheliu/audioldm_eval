import torch
from audioldm_eval import EvaluationHelper

device = torch.device(f"cuda:{0}")

# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k"
generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k_16k"
# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k_resp_librosa_16k"
# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k_resp_ta_16k"
# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k_resp_sox_simple__16k"
# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_32k_resp_sox_16k"

target_audio_path = "/mnt/bn/lqhaoheliu/project/audioldm_eval/audio/target/audiocaps"
# target_audio_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/testset_data/audiocaps_16k"
# target_audio_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/testset_data/audiocaps_32k_16k_sox"

evaluator = EvaluationHelper(32000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)

# generation_result_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/latent_diffusion/22_is_audiomae_helpful_or_not/2023_06_20_clap_1000_steps_32k/val_247520_16k"
# # generation_result_path = "example/unpaired"
# target_audio_path = "/mnt/bn/lqhaoheliu/project/audioldm_eval/audio/target/audiocaps_16k"

# evaluator = EvaluationHelper(16000, device)

# # Perform evaluation, result will be print out and saved as json
# metrics = evaluator.main(
#     generation_result_path,
#     target_audio_path,
# )

# generation_result_path = "/mnt/bn/lqhaoheliu/project/audioldm_eval/audio/output/2023_05_20_audiomae_crossattn_audiocaps_val_185640"
# # generation_result_path = "example/unpaired"
# target_audio_path = "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/testset_data/audiocaps_16k"

# evaluator = EvaluationHelper(16000, device)

# # Perform evaluation, result will be print out and saved as json
# metrics = evaluator.main(
#     generation_result_path,
#     target_audio_path,
# )
