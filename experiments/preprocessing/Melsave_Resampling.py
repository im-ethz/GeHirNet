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

import os
import numpy as np
import torchaudio
import torchaudio.transforms as T

# Fixed setting for 44100Hz
fixed_parameters_44100 = {'window_length': 1808, 'hop_length': 452}

# input/output paths
input_dirs = {
    "ALS": r"path\to\ALS",
    "Covid-19": r"path\to\Covid-19",
    "Dysphonie": r"path\to\Dysphonie",
    "Laryngitis": r"path\to\Laryngitis",
    "Parkinson": r"path\to\Parkinson",
    "Rekurrensparese": r"path\to\Rekurrensparese",
    "HC": r"path\to\HC"
}
output_dirs = {
    "ALS": r"path\to\Mel_NPY_Output\ALS",
    "Covid-19": r"path\to\Mel_NPY_Output\Covid-19",
    "Dysphonie": r"path\to\Mel_NPY_Output\Dysphonie",
    "Laryngitis": r"path\to\Mel_NPY_Output\Laryngitis",
    "Parkinson": r"path\to\Mel_NPY_Output\Parkinson",
    "Rekurrensparese": r"path\to\Mel_NPY_Output\Rekurrensparese",
    "HC": r"path\to\Mel_NPY_Output\HC"
}

for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# define abnormal document list
unsaved_files = []
invalid_mel_shapes = []

# Dynamical modification for hop/window length
def calculate_dynamic_parameters(sample_rate):
    # modification base：45000Hz
    base_sample_rate = 45000
    base_window_length = 1848
    base_hop_length = 462

    # scale computing
    scale = sample_rate / base_sample_rate
    window_length = int(base_window_length * scale)
    hop_length = int(base_hop_length * scale)
    
    return window_length, hop_length

def generate_and_save_mel(file_path, output_dir):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"Processing: {file_path}, Sample Rate: {sample_rate}, Shape: {waveform.shape}")
        
        # 44100 or not
        if sample_rate == 44100:
            # use set parameters
            window_length = fixed_parameters_44100['window_length']
            hop_length = fixed_parameters_44100['hop_length']
        elif 40000 <= sample_rate <= 50000 and (sample_rate - 40000) % 125 == 0:
            # dynamic modification
            window_length, hop_length = calculate_dynamic_parameters(sample_rate)
        else:
            print(f"[SKIP] Unsupported sample rate ({sample_rate}) for file: {file_path}")
            unsaved_files.append(file_path)  # record abnormal info
            return
        
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_length,
            hop_length=hop_length,
            n_mels=128
        )(waveform)

        log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)

        if log_mel_spectrogram.size(2) != 98:
            print(f"[INVALID SHAPE] File: {file_path}, Shape: {log_mel_spectrogram.size()}")
            invalid_mel_shapes.append((file_path, log_mel_spectrogram.size()))
            return

        # save .npy
        file_name = os.path.splitext(os.path.basename(file_path))[0] + ".npy"
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, log_mel_spectrogram.numpy())

        # print npy size, window/hop length
        print(f"[SAVED] {output_path}, Shape: {log_mel_spectrogram.size()}, Window Length: {window_length}, Hop Length: {hop_length}")

    except Exception as e:
        print(f"[ERROR] Failed to process file: {file_path}. Error: {e}")
        unsaved_files.append(file_path) 

for category, input_dir in input_dirs.items():
    output_dir = output_dirs[category]
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            generate_and_save_mel(file_path, output_dir)

# Check abnormal documents during processing
print("\nSummary of unsaved files:")
if unsaved_files:
    for unsaved_file in unsaved_files:
        print(unsaved_file)
else:
    print("All files were successfully saved.")

print("\nSummary of invalid Mel spectrogram shapes:")
if invalid_mel_shapes:
    for file_path, shape in invalid_mel_shapes:
        print(f"File: {file_path}, Shape: {shape}")
else:
    print("All Mel spectrograms have a valid shape (128, 98).")
