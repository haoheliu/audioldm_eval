import os
import torch
import datetime
import numpy as np
import torch.multiprocessing as mp
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, AutoModel, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from audioldm_eval.datasets.load_mel import WaveDataset
from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from audioldm_eval.feature_extractors.panns import Cnn14
from audioldm_eval.audio.tools import save_pickle, load_pickle, write_json, load_json
from tqdm import tqdm

class EvaluationHelperParallel:
    def __init__(self, sampling_rate, num_gpus, batch_size=1, backbone="mert") -> None:
        self.sampling_rate = sampling_rate
        self.num_gpus = num_gpus
        self.backbone = backbone
        self.batch_size = batch_size

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup(self):
        dist.destroy_process_group()

    def init_models(self, rank):
        self.device = torch.device(f"cuda:{rank}")
        self.frechet = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)
        self.frechet.model = self.frechet.model.to(self.device)

        features_list = ["2048", "logits"]
        
        if self.backbone == "mert":
            self.mel_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
            self.target_sample_rate = self.processor.sampling_rate
            self.resampler = T.Resample(orig_freq=self.sampling_rate, new_freq=self.target_sample_rate).to(self.device)
            
        elif self.backbone == "cnn14":
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
                raise ValueError("We only support the evaluation on 16kHz and 32kHz sampling rate.")
        
        else:
            raise ValueError("Backbone not supported")

        self.mel_model = DDP(self.mel_model.to(self.device), device_ids=[rank])
        self.mel_model.eval()

    def main(self, generate_files_path, groundtruth_path, limit_num=None):
        mp.spawn(self.run, args=(generate_files_path, groundtruth_path, limit_num), nprocs=self.num_gpus, join=True)
    
    def get_featuresdict(self, rank, dataloader):
        out = None
        out_meta = {"file_path_": []}

        for waveform, filename in dataloader:
            try:
                waveform = waveform.squeeze(1).float().to(self.device)

                with torch.no_grad():
                    if self.backbone == "mert":
                        waveform = self.resampler(waveform[0])
                        mert_input = self.processor(waveform, sampling_rate=self.target_sample_rate, return_tensors="pt").to(self.device)
                        mert_output = self.mel_model(**mert_input, output_hidden_states=True)
                        time_reduced_hidden_states = torch.stack(mert_output.hidden_states).squeeze().mean(dim=1)
                        featuresdict = {"2048": time_reduced_hidden_states, "logits": time_reduced_hidden_states}
                    elif self.backbone == "cnn14":
                        featuresdict = self.mel_model(waveform)

                featuresdict = {k: v for k, v in featuresdict.items()}

                if out is None:
                    out = featuresdict
                else:
                    out = {k: torch.cat([out[k], featuresdict[k]], dim=0) for k in out.keys()}
                
                out_meta["file_path_"].extend(filename)
            except Exception as e:
                print(f"Classifier Inference error on rank {rank}: ", e)
                continue

        return out, out_meta

    def gather_features(self, featuresdict, metadict):
        all_features = {}
        for k, v in featuresdict.items():
            gathered = [torch.zeros_like(v) for _ in range(self.num_gpus)]
            dist.all_gather(gathered, v)
            all_features[k] = torch.cat(gathered, dim=0)

        all_meta = {}
        for k, v in metadict.items():
            gathered = [None for _ in range(self.num_gpus)]
            dist.all_gather_object(gathered, v)
            all_meta[k] = sum(gathered, [])

        return {**all_features, **all_meta}

    def run(self, rank, generate_files_path, groundtruth_path, limit_num):
        self.setup(rank, self.num_gpus)
        self.init_models(rank)

        same_name = self.get_filename_intersection_ratio(generate_files_path, groundtruth_path, limit_num=limit_num)

        metrics = self.calculate_metrics(rank, generate_files_path, groundtruth_path, same_name, limit_num) # recalculate = True

        if rank == 0:
            print("\n".join((f"{k}: {v:.7f}" for k, v in metrics.items())))
            json_path = os.path.join(os.path.dirname(generate_files_path), f"{self.get_current_time()}_{os.path.basename(generate_files_path)}.json")
            write_json(metrics, json_path)

        self.cleanup()

    def calculate_metrics(self, rank, generate_files_path, groundtruth_path, same_name, limit_num=None, calculate_psnr_ssim=False, calculate_lsd=False, recalculate=False):
        torch.manual_seed(0)
        num_workers = 6

        output_dataset = WaveDataset(generate_files_path, self.sampling_rate, limit_num=limit_num)
        result_dataset = WaveDataset(groundtruth_path, self.sampling_rate, limit_num=limit_num)

        output_sampler = DistributedSampler(output_dataset, num_replicas=self.num_gpus, rank=rank)
        result_sampler = DistributedSampler(result_dataset, num_replicas=self.num_gpus, rank=rank)

        outputloader = DataLoader(
            output_dataset,
            batch_size=self.batch_size,
            sampler=output_sampler,
            num_workers=num_workers,
        )

        resultloader = DataLoader(
            result_dataset,
            batch_size=self.batch_size,
            sampler=result_sampler,
            num_workers=num_workers,
        )

        out = {}
        if rank == 0:
            if(recalculate): 
                print("Calculate FAD score from scratch")
            fad_score = self.frechet.score(generate_files_path, groundtruth_path, limit_num=limit_num, recalculate=recalculate)
            out.update(fad_score)
            print("FAD: %s" % fad_score)
            
        cache_path = generate_files_path + "classifier_logits_feature_cache.pkl"
        if os.path.exists(cache_path) and not recalculate:
            print("reload", cache_path)
            all_featuresdict_1 = load_pickle(cache_path)
        else:
            print(f"Extracting features from {generate_files_path}.")
            featuresdict_1, metadict_1 = self.get_featuresdict(rank, outputloader)
            all_featuresdict_1 = self.gather_features(featuresdict_1, metadict_1)
            if rank == 0: 
                save_pickle(all_featuresdict_1, cache_path)

        cache_path = groundtruth_path + "classifier_logits_feature_cache.pkl"
        if os.path.exists(cache_path) and not recalculate:
            print("reload", cache_path)
            all_featuresdict_2 = load_pickle(cache_path)
        else:
            print(f"Extracting features from {groundtruth_path}.")
            featuresdict_2, metadict_2 = self.get_featuresdict(rank, resultloader)
            all_featuresdict_2 = self.gather_features(featuresdict_2, metadict_2)
            if rank == 0:  
                save_pickle(all_featuresdict_2, cache_path)

        if rank == 0:
            for k, v in all_featuresdict_1.items():
                if isinstance(v, torch.Tensor):
                    all_featuresdict_1[k] = v.cpu()
            for k, v in all_featuresdict_2.items():
                if isinstance(v, torch.Tensor):
                    all_featuresdict_2[k] = v.cpu()
                
            metric_kl, _, _ = calculate_kl(all_featuresdict_1, all_featuresdict_2, "logits", same_name)
            out.update(metric_kl)

            metric_isc = calculate_isc(all_featuresdict_1, feat_layer_name="logits", splits=10, samples_shuffle=True, rng_seed=2020)
            out.update(metric_isc)

            if "2048" in all_featuresdict_1.keys() and "2048" in all_featuresdict_2.keys():
                metric_fid = calculate_fid(all_featuresdict_1, all_featuresdict_2, feat_layer_name="2048")
                out.update(metric_fid)
                
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
                    batch_size=self.batch_size,
                    sampler=None,
                    num_workers=16,
                )
                
            if(calculate_lsd):
                metric_lsd = self.calculate_lsd(pairedloader, same_name=same_name)
                out.update(metric_lsd)

            if(calculate_psnr_ssim):
                metric_psnr_ssim = self.calculate_psnr_ssim(pairedloader, same_name=same_name)
                out.update(metric_psnr_ssim)

        dist.barrier()
        return out
    
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
    
    def get_current_time(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d-%H:%M:%S")
    
    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[: samples.shape[0]], :]