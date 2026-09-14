# Copyright 2025 ETH Zurich
# Project contributors:
#   Fan Wu (ETH Zurich)
#   Kaicheng Zhao (RWTH Aachen University)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
import torch

# pre-set parameters
window_length = 2048  # STFT window length
hop_length = 512  # STFT hop length
vad_threshold = 1e-3  # energy threshold for checking silence
fade_samples = 512  # crossfade region

# input/output path
input_dir = r"path\to\input\audios"
output_dir_audio = r"path\to\output\vad_segments" # audio after VAD
output_dir_mel_full = r"path\to\output\raw_melspectrogram" # spectrogram of raw audios
output_dir_mel_filtered = r"path\to\output\melspectromgram_with_silence_box" # spectrogram with silence boxes
output_dir_mel_no_silence = r"path\to\output\no_silence_mel_spectrograms" # spectrogram without silence
os.makedirs(output_dir_audio, exist_ok=True)
os.makedirs(output_dir_mel_full, exist_ok=True)
os.makedirs(output_dir_mel_filtered, exist_ok=True)
os.makedirs(output_dir_mel_no_silence, exist_ok=True)

# record files which completely dropped
no_segments_files = []
no_mel_files = []

def energy_based_vad(audio, frame_length=2048, hop_length=512, threshold=vad_threshold):
    # RMS computing for each frame
    rms_energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_length]**2))
        for i in range(0, len(audio) - frame_length + 1, hop_length)
    ])
    vad_segments = []
    is_speech = False
    start = 0

    for i, e in enumerate(rms_energy):
        if e > threshold and not is_speech:
            is_speech = True
            start = i * hop_length
        elif e < threshold and is_speech:
            is_speech = False
            end = i * hop_length
            vad_segments.append((start, end))
    
    if is_speech:
        vad_segments.append((start, len(audio)))
    return vad_segments

def apply_fade(audio, fade_samples):
    """
    Fade in/out
    """
    if len(audio) < fade_samples:
        fade_in = np.linspace(0, 1, len(audio) // 2)
        fade_out = np.linspace(1, 0, len(audio) // 2)
        audio[:len(fade_in)] *= fade_in
        audio[-len(fade_out):] *= fade_out
    else:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    return audio

def apply_crossfade(segments, fade_samples):
    """
    Crossdade for connections
    """
    if not segments:
        return np.array([])

    output_audio = segments[0]

    for i in range(1, len(segments)):
        previous_segment = output_audio[-fade_samples:]
        current_segment = segments[i][:fade_samples]
        
        crossfade = np.linspace(1, 0, fade_samples) * previous_segment + np.linspace(0, 1, fade_samples) * current_segment
        output_audio = np.concatenate([output_audio[:-fade_samples], crossfade, segments[i][fade_samples:]])

    return output_audio

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        
        waveform, sr = torchaudio.load(file_path)
        audio_np = waveform.numpy().flatten()
        
        mel_transform = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=window_length,
            hop_length=hop_length,
            n_mels=128
        )
        
        mel_spectrogram = mel_transform(waveform)
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Full Mel Spectrogram of {filename}')
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        
        full_mel_output_path = os.path.join(output_dir_mel_full, f"{os.path.splitext(filename)[0]}_full_mel.png")
        plt.savefig(full_mel_output_path)
        plt.close()
        
        print(f"Saved full Mel spectrogram for {filename} at {full_mel_output_path}")
        
        segments = energy_based_vad(audio_np)
        
        mask = np.zeros(len(audio_np), dtype=bool)
        for start, end in segments:
            mask[start:end] = True

        filtered_waveform = waveform[:, mask]

        if filtered_waveform.numel() > 0:
            mel_spectrogram_filtered = mel_transform(filtered_waveform)
            mel_spectrogram_filtered_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram_filtered)

            plt.figure(figsize=(10, 4))
            plt.imshow(mel_spectrogram_filtered_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Filtered Mel Spectrogram (Masked) of {filename}')
            plt.xlabel("Time")
            plt.ylabel("Mel Frequency")

            filtered_mel_output_path = os.path.join(output_dir_mel_filtered, f"{os.path.splitext(filename)[0]}_filtered_mel.png")
            plt.savefig(filtered_mel_output_path)
            plt.close()

            print(f"Saved filtered Mel spectrogram for {filename} at {filtered_mel_output_path}")
        else:
            print(f"No valid segments found for {filename}, skipping filtered Mel spectrogram generation.")
            no_mel_files.append(filename)

        if segments:
            faded_segments = [apply_fade(audio_np[start:end], fade_samples) for start, end in segments]
            concatenated_audio = apply_crossfade(faded_segments, fade_samples)

            concatenated_audio_path = os.path.join(output_dir_audio, f"{os.path.splitext(filename)[0]}_concatenated_crossfade.wav")
            sf.write(concatenated_audio_path, concatenated_audio, sr)

            print(f"Saved concatenated audio with crossfade for {filename} at {concatenated_audio_path}")

            concatenated_waveform = torch.from_numpy(concatenated_audio).unsqueeze(0).to(torch.float32)
            mel_spectrogram_no_silence = mel_transform(concatenated_waveform)
            mel_spectrogram_no_silence_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram_no_silence)

            plt.figure(figsize=(10, 4))
            plt.imshow(mel_spectrogram_no_silence_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram without Silence and with Crossfade of {filename}')
            plt.xlabel("Time")
            plt.ylabel("Mel Frequency")

            no_silence_mel_output_path = os.path.join(output_dir_mel_no_silence, f"{os.path.splitext(filename)[0]}_no_silence_crossfade_mel.png")
            plt.savefig(no_silence_mel_output_path)
            plt.close()

            print(f"Saved Mel spectrogram without silence and with crossfade for {filename} at {no_silence_mel_output_path}")
        else:
            print(f"No segments found for {filename}, skipping concatenation.")
            no_segments_files.append(filename)

if no_segments_files:
    print("\nFiles with no speech segments detected:")
    for fname in no_segments_files:
        print(fname)

