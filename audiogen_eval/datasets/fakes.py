import sys
import librosa
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.datasets import DatasetFolder

sys.path.insert(0, ".")  # nopep8
from train import instantiate_from_config
from feature_extraction.extract_mel_spectrogram import TRANSFORMS
import random


def read_wav(path, folder_name="melspec_10s_22050hz", length=22050 * 10):
    wav, _ = librosa.load(path, sr=None)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[: len(wav)] = wav
    else:
        y = wav[:length]
    if folder_name == "melspec_10s_22050hz":
        mel_spec = TRANSFORMS(y)
    else:
        raise NotImplementedError
    return mel_spec


class FakesFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        assert isinstance(extensions, str)
        if loader is None:
            loader = self.sample_from_path
        # print('transform ',transform)
        # print('target_transform ',target_transform)
        # print(root)
        # assert 1==2
        # print('is_valid_file ',is_valid_file) # 全是None
        # assert 1==2
        if target_transform is None:
            target_transform = self.target_transform
        super().__init__(
            root,
            loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # super().__getitem__(index) returns (sample, target) – we need only sample
        path, target = self.samples[index]
        # print(path)
        # print('target ',target)
        # comenting out target and label, because those are asigned by folder name, not original labels
        image = super().__getitem__(index)[0]
        # print('image ',image.shape)
        # assert 1==2
        return {
            # 'target': target,
            # 'label': self.idx_to_class[target],
            "file_path_": path,
            "image": image,
        }

    @staticmethod
    def sample_from_path(path):
        if path.endswith(".wav"):
            return read_wav(path)
        else:
            return np.load(path)

    @staticmethod
    def target_transform(target):
        return None


class FakesFolder_mask(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        assert isinstance(extensions, str)
        if loader is None:
            loader = self.sample_from_path
        # print('transform ',transform)
        # print('target_transform ',target_transform)
        # print(root)
        # assert 1==2
        # print('is_valid_file ',is_valid_file) # 全是None
        # assert 1==2
        if target_transform is None:
            target_transform = self.target_transform
        super().__init__(
            root,
            loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # super().__getitem__(index) returns (sample, target) – we need only sample
        path, target = self.samples[index]
        # print(path)
        # print('target ',target)
        # comenting out target and label, because those are asigned by folder name, not original labels
        image = super().__getitem__(index)[0]
        mask = np.ones((80, 860))
        mask_rate = 0.5
        for i in range(int(860 * mask_rate)):
            t = random.randint(1, 859)
            mask[:, t] = 0.0
        image = image * mask
        # print('image ',image.shape)
        # assert 1==2
        return {
            # 'target': target,
            # 'label': self.idx_to_class[target],
            "file_path_": path,
            "image": image,
        }

    @staticmethod
    def sample_from_path(path):
        if path.endswith(".wav"):
            return read_wav(path)
        else:
            return np.load(path)

    @staticmethod
    def target_transform(target):
        return None


if __name__ == "__main__":
    # input1:
    # target: evaluation.datasets.fakes.FakesFolder
    # params:
    #     root: ./logs/2021-06-03T09-34-10_vggsound_resampleframes_transformer/samples_2021-06-11T20-55-54/VGGSound_test
    #     extensions: .npy
    #     transform:
    #     - target: evaluation.datasets.transforms.GetInputFromBatchByKey
    #       params:
    #         input_key: image
    #     - target: specvqgan.modules.losses.vggishish.transforms.StandardNormalizeAudio
    #         params:
    #         specs_dir: ./data/vggsound/melspec_10s_22050hz
    #         cache_path: ./specvqgan/modules/losses/vggishish/data/
    from omegaconf import OmegaConf

    # cfg = OmegaConf.load('./evaluation/configs/eval_melception_vas.yaml')
    # cfg.input1.params.root = './logs/2021-06-09T15-17-18_vas_resampleframes_transformer/samples_2021-06-14T10-43-53/VAS_validation'
    cfg = OmegaConf.load("./evaluation/configs/eval_melception_vggsound.yaml")
    cfg.input1.params.root = "./logs/2021-06-03T09-34-10_vggsound_resampleframes_transformer/samples_2021-06-11T20-55-54/VGGSound_test"
    data = instantiate_from_config(cfg.input1)
    print(len(data))
    print(data[0])
