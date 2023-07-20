
import torch
import torchaudio
import torch.nn.functional as F

import torch.nn as nn
from audioldm_eval.clap_score.clap.open_clip import create_model
from audioldm_eval.clap_score.clap.training.data import get_audio_features
from transformers import RobertaTokenizer, AutoTokenizer, T5EncoderModel, MT5EncoderModel

class CLAPAudioEmbeddingClassifierFreev2(nn.Module):
    def __init__(
        self,
        pretrained_path,
        sampling_rate=16000,
        embed_mode="audio",
        amodel="HTSAT-base",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=True,
    ):
        super().__init__()
        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )
        audio_cfg = self.model_cfg["audio_cfg"]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            win_length=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.unconditional_token = None
        self.model.eval()

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]
        return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.size(-1)
        for i in range(waveform.size(0)):
            mute_size = int(
                self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion))
            )
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()

    def build_unconditional_emb(self):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]

    def forward(self, batch):
        # If you want this conditioner to be unconditional, set self.unconditional_prob = 1.0
        # If you want this conditioner to be fully conditional, set self.unconditional_prob = 0.0
        if self.model.training == True and not self.training_mode:
            print(
                "The pretrained CLAP model should always be in eval mode. Reloading model just in case you change the parameters."
            )
            self.model, self.model_cfg = create_model(
                self.amodel,
                self.tmodel,
                self.pretrained,
                precision=self.precision,
                device="cuda",
                enable_fusion=self.enable_fusion,
                fusion_type=self.fusion_type,
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        if(self.unconditional_token is None):
            self.build_unconditional_emb()

        # if(self.training_mode):
        #     assert self.model.training == True
        # else:
        #     assert self.model.training == False

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        if self.embed_mode == "audio":
            with torch.no_grad():
                # assert (
                #     self.sampling_rate == 16000
                # ), "We only support 16000 sampling rate"
                
                # if self.random_mute:
                #     batch = self._random_mute(batch)
                # batch: [bs, 1, t-samples]
                if(self.sampling_rate != 48000):
                    batch = torchaudio.functional.resample(
                        batch, orig_freq=self.sampling_rate, new_freq=48000
                    )

                audio_data = batch.squeeze(1)
                mel = self.mel_transform(audio_data)
                audio_dict = get_audio_features(
                    audio_data,
                    mel, 
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict)
        elif self.embed_mode == "text":
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                
                if(isinstance(batch, str) or (isinstance(batch, list) and len(batch) == 1)):
                    for key in text_data.keys():
                        text_data[key] = text_data[key].unsqueeze(0)
                        
                embed = self.model.get_text_embedding(text_data)

        embed = embed.unsqueeze(1)
        for i in range(embed.size(0)):
            if self.make_decision(self.unconditional_prob):
                embed[i] = self.unconditional_token
        # embed = torch.randn((batch.size(0), 1, 512)).type_as(batch)
        return embed.detach()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
