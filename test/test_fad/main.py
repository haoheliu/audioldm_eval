from audiogen_eval.metrics.fad import FrechetAudioDistance

frechet = FrechetAudioDistance(
     use_pca=False, 
     use_activation=False,
     verbose=True,
     audio_load_worker=1
)

frechet.model = frechet.model.cuda()
# 0.47
# fad_score = frechet.score("/mnt/fast/nobackup/scratch4weeks/hl01486/exps/audio_generation/stablediffusion/autoencoderkl16k/audioverse/2022-12-07-kl-f4-ch128-no-time-stride-num_res_blocks-2-klweight-1_8_128_4.5e-05_v1.0.2/autoencoder_result/1030000/fbank_vocoder_gt_wave", "/mnt/fast/datasets/audio/audioset/2million_audioset_wav/eval_segments")
fad_score = frechet.score("/mnt/fast/nobackup/scratch4weeks/hl01486/exps/audio_generation/stablediffusion/autoencoderkl16k/audioverse/2022-12-07-kl-f4-ch128-no-time-stride-num_res_blocks-2-klweight-1_8_128_4.5e-05_v1.0.2/autoencoder_result/1030000/fbank_wav_prediction", "/mnt/fast/datasets/audio/audioset/2million_audioset_wav/eval_segments")
print(fad_score)

# fad_score = 12.739781528181275

# fad_score = frechet.score("background/", "test2/")
# print(fad_score)
# fad_score = 4.981572090467181