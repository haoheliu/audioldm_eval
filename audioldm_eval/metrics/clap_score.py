import torch
import numpy as np

from torch.utils.data import DataLoader
from audioldm_eval.clap_score.model import CLAPAudioEmbeddingClassifierFreev2
from audioldm_eval.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset

from tqdm import tqdm

def calculate_clap_sore(generate_files_path, filename_to_text_mapping):
    print("Calculate the clap score of: ", generate_files_path)
    outputloader = DataLoader(
        WaveDataset(
            generate_files_path,
            32000,
            limit_num=None,
        ),
        batch_size=1,
        sampler=None,
        num_workers=4,
    )

    clap = CLAPAudioEmbeddingClassifierFreev2(pretrained_path="/mnt/bn/lqhaoheliu/exps/checkpoints/audioldm/2023_04_07_audioldm_clap_v2_yusong/music_speech_audioset_epoch_15_esc_89.98.pt",
                                                sampling_rate=32000,
                                                embed_mode="audio",
                                                amodel = "HTSAT-base").cuda()
    clap.eval()
    individual_similarity = []

    with torch.no_grad():
        for waveform, filename in tqdm(outputloader):
            similarity = clap.cos_similarity(torch.FloatTensor(waveform).squeeze(1), filename_to_text_mapping[filename[0]])
            individual_similarity.append(similarity.detach().cpu().item())

    return np.mean(individual_similarity)
