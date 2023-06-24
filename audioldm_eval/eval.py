import os
from audioldm_eval.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset
import numpy as np
import argparse
import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from audioldm_eval.feature_extractors.panns import Cnn14
from audioldm_eval.audio.tools import save_pickle, load_pickle, write_json, load_json
from ssr_eval.metrics import AudioMetrics
import audioldm_eval.audio as Audio
# from hear21passt.base import get_basic_model,get_model_passt

class EvaluationHelper:
    def __init__(self, sampling_rate, device, backbone="cnn14") -> None:

        self.device = device
        self.backbone = backbone
        self.sampling_rate = sampling_rate
        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        
        # self.passt_model = get_basic_model(mode="logits")
        # self.passt_model.eval()
        # self.passt_model.to(self.device)

        # self.lsd_metric = AudioMetrics(self.sampling_rate)
        self.frechet.model = self.frechet.model.to(device)

        features_list = ["2048", "logits"]
        if self.sampling_rate == 16000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
        elif self.sampling_rate == 32000:
            self.mel_model = Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

        if self.sampling_rate == 16000:
            self._stft = Audio.TacotronSTFT(512, 160, 512, 64, 16000, 50, 8000)
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(1024, 320, 1024, 64, 32000, 50, 14000)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

        self.mel_model.eval()
        self.mel_model.to(self.device)
        self.fbin_mean, self.fbin_std = None, None

    def main(
        self,
        generate_files_path,
        groundtruth_path,
        limit_num=None,
    ):
        print("Generted files", generate_files_path)
        print("Target files", groundtruth_path)

        self.file_init_check(generate_files_path)
        self.file_init_check(groundtruth_path)

        same_name = self.get_filename_intersection_ratio(
            generate_files_path, groundtruth_path, limit_num=limit_num
        )

        metrics = self.calculate_metrics(generate_files_path, groundtruth_path, same_name, limit_num, recalculate=True) # 

        return metrics

    def file_init_check(self, dir):
        assert os.path.exists(dir), "The path does not exist %s" % dir
        assert len(os.listdir(dir)) > 1, "There is no files in %s" % dir

    def get_filename_intersection_ratio(
        self, dir1, dir2, threshold=0.99, limit_num=None
    ):
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)

        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}

        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())

        intersect_keys = keyset1.intersection(keyset2)
        if (
            len(intersect_keys) / len(keyset1) > threshold
            and len(intersect_keys) / len(keyset2) > threshold
        ):
            print(
                "+Two path have %s intersection files out of total %s & %s files. Processing two folder with same_name=True"
                % (len(intersect_keys), len(keyset1), len(keyset2))
            )
            return True
        else:
            print(
                "-Two path have %s intersection files out of total %s & %s files. Processing two folder with same_name=False"
                % (len(intersect_keys), len(keyset1), len(keyset2))
            )
            return False

    def calculate_lsd(self, pairedloader, same_name=True, time_offset=160 * 7):
        if same_name == False:
            return {
                "lsd": -1,
                "ssim_stft": -1,
            }
        print("Calculating LSD using a time offset of %s ..." % time_offset)
        lsd_avg = []
        ssim_stft_avg = []
        for _, _, filename, (audio1, audio2) in tqdm(pairedloader):
            audio1 = audio1.cpu().numpy()[0, 0]
            audio2 = audio2.cpu().numpy()[0, 0]

            # If you use HIFIGAN (verified on 2023-01-12), you need seven frames' offset
            audio1 = audio1[time_offset:]

            audio1 = audio1 - np.mean(audio1)
            audio2 = audio2 - np.mean(audio2)

            audio1 = audio1 / np.max(np.abs(audio1))
            audio2 = audio2 / np.max(np.abs(audio2))

            min_len = min(audio1.shape[0], audio2.shape[0])

            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            result = self.lsd(audio1, audio2)

            lsd_avg.append(result["lsd"])
            ssim_stft_avg.append(result["ssim"])

        return {"lsd": np.mean(lsd_avg), "ssim_stft": np.mean(ssim_stft_avg)}

    def lsd(self, audio1, audio2):
        result = self.lsd_metric.evaluation(audio1, audio2, None)
        return result

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {"psnr": -1, "ssim": -1}
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=data_range))
        return {"psnr": np.mean(psnr_avg), "ssim": np.mean(ssim_avg)}

    def calculate_metrics(self, generate_files_path, groundtruth_path, same_name, limit_num=None, calculate_psnr_ssim=False, calculate_lsd=False, recalculate=False):
        # Generation, target
        torch.manual_seed(0)

        num_workers = 6

        outputloader = DataLoader(
            WaveDataset(
                generate_files_path,
                self.sampling_rate, # TODO
                # 32000,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            WaveDataset(
                groundtruth_path,
                self.sampling_rate, # TODO
                # 32000,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )

        out = {}

        print("")
        # FAD
        ######################################################################################################################
        # if(recalculate): 
        #     print("Calculate FAD score from scratch")
        # fad_score = self.frechet.score(generate_files_path, groundtruth_path, limit_num=limit_num, recalculate=recalculate)
        # out.update(fad_score)
        # print("FAD: %s" % fad_score)
        ######################################################################################################################
        
        # PANNs
        ######################################################################################################################
        featuresdict_2 = self.get_featuresdict(resultloader)
        featuresdict_1 = self.get_featuresdict(outputloader)

        metric_kl, kl_ref, paths_1 = calculate_kl(
            featuresdict_1, featuresdict_2, "logits", same_name
        )
        
        out.update(metric_kl)

        metric_isc = calculate_isc(
            featuresdict_1,
            feat_layer_name="logits",
            splits=10,
            samples_shuffle=True,
            rng_seed=2020,
        )
        out.update(metric_isc)

        if("2048" in featuresdict_1.keys() and "2048" in featuresdict_2.keys()):
            metric_fid = calculate_fid(
                featuresdict_1, featuresdict_2, feat_layer_name="2048"
            )
            out.update(metric_fid)

        # Metrics for Autoencoder
        ######################################################################################################################
        if(calculate_psnr_ssim or calculate_lsd):
            pairedloader = DataLoader(
                MelPairedDataset(
                    generate_files_path,
                    groundtruth_path,
                    self._stft,
                    self.sampling_rate,
                    self.fbin_mean,
                    self.fbin_std,
                    limit_num=limit_num,
                ),
                batch_size=1,
                sampler=None,
                num_workers=16,
            )
            
        if(calculate_lsd):
            metric_lsd = self.calculate_lsd(pairedloader, same_name=same_name)
            out.update(metric_lsd)

        if(calculate_psnr_ssim):
            metric_psnr_ssim = self.calculate_psnr_ssim(pairedloader, same_name=same_name)
            out.update(metric_psnr_ssim)

        # metric_kid = calculate_kid(
        #     featuresdict_1,
        #     featuresdict_2,
        #     feat_layer_name="2048",
        #     subsets=100,
        #     subset_size=1000,
        #     degree=3,
        #     gamma=None,
        #     coef0=1,
        #     rng_seed=2020,
        # )
        # out.update(metric_kid)

        print("\n".join((f"{k}: {v:.7f}" for k, v in out.items())))
        print("\n")
        print(limit_num)
        print(
            f'KL_Sigmoid: {out.get("kullback_leibler_divergence_sigmoid", float("nan")):8.5f};',
            f'KL: {out.get("kullback_leibler_divergence_softmax", float("nan")):8.5f};',
            f'PSNR: {out.get("psnr", float("nan")):.5f}',
            f'SSIM: {out.get("ssim", float("nan")):.5f}',
            f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
            f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
            f'({out.get("kernel_inception_distance_std", float("nan")):.5f})',
            f'FD: {out.get("frechet_distance", float("nan")):8.5f};',
            f'FAD: {out.get("frechet_audio_distance", float("nan")):.5f}',
            f'LSD: {out.get("lsd", float("nan")):.5f}',
            # f'SSIM_STFT: {out.get("ssim_stft", float("nan")):.5f}',
        )
        result = {
            "frechet_distance": out.get("frechet_distance", float("nan")),
            "frechet_audio_distance": out.get("frechet_audio_distance", float("nan")),
            "kullback_leibler_divergence_sigmoid": out.get(
                "kullback_leibler_divergence_sigmoid", float("nan")
            ),
            "kullback_leibler_divergence_softmax": out.get(
                "kullback_leibler_divergence_softmax", float("nan")
            ),
            "lsd": out.get("lsd", float("nan")),
            "psnr": out.get("psnr", float("nan")),
            "ssim": out.get("ssim", float("nan")),
            # "ssim_stft": out.get("ssim_stft", float("nan")),
            "inception_score_mean": out.get("inception_score_mean", float("nan")),
            "inception_score_std": out.get("inception_score_std", float("nan")),
            "kernel_inception_distance_mean": out.get(
                "kernel_inception_distance_mean", float("nan")
            ),
            "kernel_inception_distance_std": out.get(
                "kernel_inception_distance_std", float("nan")
            ),
        }

        json_path = os.path.join(os.path.dirname(generate_files_path), self.get_current_time()+"_"+os.path.basename(generate_files_path) + ".json")
        write_json(result, json_path)
        return result

    def get_current_time(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d-%H:%M:%S")

    def get_featuresdict(self, dataloader):
        out = None
        out_meta = None

        # transforms=StandardNormalizeAudio()
        for waveform, filename in tqdm(dataloader):
            try:
                metadict = {
                    "file_path_": filename,
                }
                waveform = waveform.squeeze(1)

                # batch = transforms(batch)
                waveform = waveform.float().to(self.device)

                # featuresdict = {}
                # with torch.no_grad():
                #     if(waveform.size(-1) >= 320000):
                #         waveform = waveform[...,:320000]
                #     else:
                #         waveform = torch.nn.functional.pad(waveform, (0,320000-waveform.size(-1)))
                #     featuresdict["logits"] = self.passt_model(waveform)

                with torch.no_grad():
                    featuresdict = self.mel_model(waveform) # "logits": [1, 527]

                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

                if out is None:
                    out = featuresdict
                else:
                    out = {k: out[k] + featuresdict[k] for k in out.keys()}

                if out_meta is None:
                    out_meta = metadict
                else:
                    out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}
            except Exception as e:
                import ipdb

                ipdb.set_trace()
                print("Classifier Inference error: ", e)
                continue

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]


if __name__ == "__main__":
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
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:{0}")

    evaluator = EvaluationHelper(args.sampling_rate, device)

    metrics = evaluator.main(
        args.generation_result_path,
        args.target_audio_path,
        limit_num=args.limit_num,
        same_name=args.same_name,
    )

    print(metrics)
