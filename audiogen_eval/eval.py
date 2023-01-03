import sys

from audiogen_eval.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset
from audiogen_eval.metrics.ndb import *
import numpy as np
import argparse
import json

import torch
from torch.utils.data import DataLoader
from audiogen_eval.feature_extractors.melception import Melception
from tqdm import tqdm
from audiogen_eval.metrics import gs
from audiogen_eval.metrics.fad import FrechetAudioDistance
from audiogen_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from audiogen_eval.feature_extractors.panns import Cnn14, Cnn14_16k

import audiogen_eval.audio as Audio

def write_json(my_dict, fname):
    print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data
    
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
            self._stft = Audio.TacotronSTFT(512,160,512,64,16000,50,8000)
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(1024,320,1024,64,32000,50,14000)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )
        
        self.mel_model.eval()
        self.mel_model.to(self.device)
        self.fbin_mean, self.fbin_std = None, None

    def main(
        self,
        o_filepath,
        resultpath,
        limit_num = None,
    ):
        self.file_init_check(o_filepath)
        self.file_init_check(resultpath)
        
        same_name=self.get_filename_intersection_ratio(o_filepath, resultpath, limit_num=limit_num)
        
        # gsm = self.getgsmscore(o_filepath, resultpath, iter_num)

        # ndb = self.getndbscore(
        #     o_filepath, resultpath, number_of_bins, evaluation_num, cache_folder
        # )

        metrics = self.calculate_metrics(o_filepath, resultpath, same_name, limit_num)
        
        return metrics
    
    def file_init_check(self, dir):
        assert os.path.exists(dir), "The path does not exist %s" % dir
        assert len(os.listdir(dir)) > 1, "There is no files in %s" % dir

    def get_filename_intersection_ratio(self, dir1, dir2, threshold=0.99, limit_num=None):
        self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
        self.datalist1 = sorted(self.datalist1)

        self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
        self.datalist2 = sorted(self.datalist2)
        
        data_dict1 = {os.path.basename(x): x for x in self.datalist1}
        data_dict2 = {os.path.basename(x): x for x in self.datalist2}
        
        keyset1 = set(data_dict1.keys())
        keyset2 = set(data_dict2.keys())
        
        intersect_keys = keyset1.intersection(keyset2)
        if(len(intersect_keys)/len(keyset1) > threshold and len(intersect_keys)/len(keyset2) > threshold):
            print("+Two path have %s intersection files out of total %s & %s files. Processing two folder with same_name=True" % (len(intersect_keys), len(keyset1), len(keyset2)))
            return True
        else:
            print("-Two path have %s intersection files out of total %s & %s files. Processing two folder with same_name=False" % (len(intersect_keys), len(keyset1), len(keyset2)))
            return False
        
    # def getndbscore(
    #     self,
    #     output,
    #     result,
    #     number_of_bins=30,
    #     evaluation_num=50,
    #     cache_folder="./results/mnist_toy_example_ndb_cache",
    # ):
    #     print("calculating the ndb score:")
    #     num_workers = 0

    #     outputloader = DataLoader(
    #         MelDataset(
    #             output,
    #             self._stft,
    #             self.sampling_rate,
    #             self.fbin_mean,
    #             self.fbin_std,
    #             augment=True,
    #         ),
    #         batch_size=1,
    #         sampler=None,
    #         num_workers=num_workers,
    #     )
    #     resultloader = DataLoader(
    #         MelDataset(
    #             result,
    #             self._stft,
    #             self.sampling_rate,
    #             self.fbin_mean,
    #             self.fbin_std,
    #         ),
    #         batch_size=1,
    #         sampler=None,
    #         num_workers=num_workers,
    #     )

    #     n_query = evaluation_num
    #     train_samples = load_npy_data(outputloader)

    #     mnist_ndb = NDB(
    #         training_data=train_samples,
    #         number_of_bins=number_of_bins,
    #         z_threshold=None,
    #         whitening=False,
    #         cache_folder=cache_folder,
    #     )

    #     result_samples = load_npy_data(resultloader)
    #     results = mnist_ndb.evaluate(
    #         self.sample_from(result_samples, n_query), "generated result"
    #     )
    #     plt.figure()
    #     mnist_ndb.plot_results()

    # def getgsmscore(self, output, result, iter_num=40):
    #     num_workers = 0

    #     print("calculating the gsm score:")

    #     outputloader = DataLoader(
    #         MelDataset(
    #             output,
    #             self._stft,
    #             self.sampling_rate,
    #             self.fbin_mean,
    #             self.fbin_std,
    #             augment=True,
    #         ),
    #         batch_size=1,
    #         sampler=None,
    #         num_workers=num_workers,
    #     )
    #     resultloader = DataLoader(
    #         MelDataset(
    #             result,
    #             self._stft,
    #             self.sampling_rate,
    #             self.fbin_mean,
    #             self.fbin_std,
    #         ),
    #         batch_size=1,
    #         sampler=None,
    #         num_workers=num_workers,
    #     )

    #     x_train = load_npy_data(outputloader)

    #     x_1 = x_train
    #     newshape = int(x_1.shape[1] / 8)
    #     x_1 = np.reshape(x_1, (-1, newshape))
    #     rlts = gs.rlts(x_1, gamma=1.0 / 128, n=iter_num)
    #     mrlt = np.mean(rlts, axis=0)

    #     gs.fancy_plot(mrlt, label="MRLT of data_1", color="C0")
    #     plt.xlim([0, 30])
    #     plt.legend()

    #     x_train = load_npy_data(resultloader)

    #     x_1 = x_train
    #     x_1 = np.reshape(x_1, (-1, newshape))
    #     rlts = gs.rlts(x_1, gamma=1.0 / 128, n=iter_num)

    #     mrlt = np.mean(rlts, axis=0)

    #     gs.fancy_plot(mrlt, label="MRLT of data_2", color="orange")
    #     plt.xlim([0, 30])
    #     plt.legend()
    #     plt.show()

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if(same_name == False):
            return {
                "psnr": -1,
                "ssim": -1
            }
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if(np.isinf(psnrval)):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            ssim_avg.append(ssim(mel_gen, mel_target))
        return {
            "psnr": np.mean(psnr_avg),
            "ssim": np.mean(ssim_avg)
        }

    def calculate_metrics(self, output, result, same_name, limit_num=None):
        # Generation, target
        torch.manual_seed(0)
        
        num_workers = 0

        outputloader = DataLoader(
            WaveDataset(
                output,
                self.sampling_rate,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )
        
        resultloader = DataLoader(
            WaveDataset(
                result,
                self.sampling_rate,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )
        
        pairedloader = DataLoader(
            MelPairedDataset(
                output,
                result,
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

        out = {}
        print("Extracting features from %s." % result)
        featuresdict_2 = self.get_featuresdict(resultloader)
        print("Extracting features from %s." % output)
        featuresdict_1 = self.get_featuresdict(outputloader)

        # if cfg.have_kl:
        metric_psnr_ssim = self.calculate_psnr_ssim(pairedloader, same_name=same_name)
        out.update(metric_psnr_ssim)
        
        metric_kl = calculate_kl(featuresdict_1, featuresdict_2, "logits", same_name)
        out.update(metric_kl)

        metric_isc = calculate_isc(
            featuresdict_1,
            feat_layer_name="logits",
            splits=10,
            samples_shuffle=True,
            rng_seed=2020,
        )
        out.update(metric_isc)

        metric_fid = calculate_fid(
            featuresdict_1, featuresdict_2, feat_layer_name="2048"
        )
        out.update(metric_fid)
        
        # Gen, target
        fad_score = self.frechet.score(output, result, limit_num=limit_num)
        out.update(fad_score)

        metric_kid = calculate_kid(
            featuresdict_1,
            featuresdict_2,
            feat_layer_name="2048",
            subsets=100,
            subset_size=1000,
            degree=3,
            gamma=None,
            coef0=1,
            rng_seed=2020,
        )
        out.update(metric_kid)

        print("\n".join((f"{k}: {v:.7f}" for k, v in out.items())))
        print("\n")
        print(limit_num)
        print(
            f'KL_Sigmoid: {out.get("kullback_leibler_divergence_sigmoid", float("nan")):8.5f};',
            f'KL_Softmax: {out.get("kullback_leibler_divergence_softmax", float("nan")):8.5f};',
            f'PSNR: {out.get("psnr", float("nan")):.5f}',
            f'SSIM: {out.get("ssim", float("nan")):.5f}',
            f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
            f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
            f'({out.get("kernel_inception_distance_std", float("nan")):.5f})',
            f'FID: {out.get("frechet_inception_distance", float("nan")):8.5f};',
            f'FAD: {out.get("frechet_audio_distance", float("nan")):.5f}',
        )
        result = {
            "kullback_leibler_divergence_sigmoid": out.get(
                "kullback_leibler_divergence_sigmoid", float("nan")
            ),
            "kullback_leibler_divergence_softmax": out.get(
                "kullback_leibler_divergence_softmax", float("nan")
            ),
            "psnr": out.get(
                "psnr", float("nan")
            ),
            "ssim": out.get(
                "ssim", float("nan")
            ),
            "inception_score_mean": out.get("inception_score_mean", float("nan")),
            "inception_score_std": out.get("inception_score_std", float("nan")),
            "kernel_inception_distance_mean": out.get(
                "kernel_inception_distance_mean", float("nan")
            ),
            "kernel_inception_distance_std": out.get(
                "kernel_inception_distance_std", float("nan")
            ),
            "frechet_inception_distance": out.get(
                "frechet_inception_distance", float("nan")
            ),
            "frechet_audio_distance": out.get(
                "frechet_audio_distance", float("nan")
            ),
        }
        json_path=output+".json"
        write_json(result, json_path)
        return result

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

                with torch.no_grad():
                    featuresdict = self.mel_model(waveform)

                # featuresdict = self.mel_model.convert_features_tuple_to_dict(features)
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
                import ipdb; ipdb.set_trace()
                print("PANNs Inference error: ", e)
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
    from audiogen_eval import EvaluationHelper
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
