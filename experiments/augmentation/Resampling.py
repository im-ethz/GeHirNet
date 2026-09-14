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
import soundfile as sf
import pandas as pd
import numpy as np
import torchaudio

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# working path
folder_path = r'path\to\documents'
vvpd_records_path = os.path.join(folder_path, "train_set.xlsx")

# Pathology folders
pathology_folders = ["ALS", "Covid-19", "Dysphonie", "Laryngitis", "Parkinson", "Rekurrensparese", "HC"]

# Initialize a dictionary to store the total duration for each pathology
total_durations = {pathology: 0 for pathology in pathology_folders}
file_info = []

# Load VVPD Table Records
vvpd_records = pd.read_excel(vvpd_records_path)

# Step 1: Traverse each pathology folder and calculate total duration
for pathology in pathology_folders:
    pathology_path = os.path.join(folder_path, pathology)
    if os.path.exists(pathology_path):
        for file_name in os.listdir(pathology_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(pathology_path, file_name)
                try:
                    with sf.SoundFile(file_path) as audio_file:
                        duration = len(audio_file) / audio_file.samplerate
                        total_durations[pathology] += duration

                        # Extract segment_filename and Group from VVPD Records
                        row = vvpd_records[vvpd_records['segment_filename'] == file_name]
                        group = row['Group'].iloc[0] if not row.empty else 'Unknown'

                        file_info.append({
                            'file_name': file_name,
                            'pathology': pathology,
                            'duration_seconds': duration,
                            'samplerate': audio_file.samplerate,
                            'augmentation_method': 'original',
                            'segment_filename': file_name,
                            'Group': group
                        })
                except Exception as e:
                    print(f"Could not read file {file_path}, error: {e}")

# Save initial total duration information
output_durations = os.path.join(folder_path, 'total_durations_by_pathology.txt')
with open(output_durations, 'w') as f:
    f.write("Initial total durations:\n")
    for pathology, total_duration in total_durations.items():
        f.write(f"{pathology}: {total_duration:.2f} seconds\n")

print(f"\nInitial total durations have been saved to {output_durations}")

# Find the longest pathology duration among all groups
max_duration = max(total_durations.values())

# Step 2: Perform data augmentation for pathologies with insufficient duration
for pathology, total_duration in total_durations.items():
    if total_duration < max_duration:
        pathology_path = os.path.join(folder_path, pathology)
        files = [f for f in os.listdir(pathology_path) if f.endswith('.wav')]

        # Dictionary to track used resampling rates for each file
        used_resampling_rates = {file_name: set() for file_name in files}

        # Generate valid resampling rates (40000 to 50000 with step 125, excluding 44100, 48000, 50000) to avoid repeatable sampling rates
        valid_resampling_rates = [rate for rate in range(40000, 50001, 125) if rate not in [44100, 48000, 50000]]

        while total_duration < max_duration:
            # Randomly select a file for augmentation
            file_name = np.random.choice(files)
            file_path = os.path.join(pathology_path, file_name)

            try:
                # Skip files if all valid resampling rates are already used
                available_rates = [rate for rate in valid_resampling_rates if rate not in used_resampling_rates[file_name]]
                if not available_rates:
                    print(f"All resampling rates used for {file_name}. Skipping.")
                    continue

                # Choose a new sampling rate from available rates
                new_samplerate = np.random.choice(available_rates)

                # Read audio data
                waveform, original_samplerate = torchaudio.load(file_path)

                # Skip audio with insufficient length
                if waveform.size(1) <= 1:
                    print(f"Skipping enhancement for {file_name} due to insufficient data length.")
                    continue

                # Apply torchaudio resample with anti-alias filter
                resampler = torchaudio.transforms.Resample(orig_freq=original_samplerate, new_freq=new_samplerate)
                resampled_waveform = resampler(waveform)

                # Generate new file name and save resampled audio
                new_file_name = f"{os.path.splitext(file_name)[0]}_{new_samplerate}hz_resampled.wav"
                new_file_path = os.path.join(pathology_path, new_file_name)

                # Write the resampled file
                torchaudio.save(new_file_path, resampled_waveform, new_samplerate)
                new_duration = resampled_waveform.size(1) / new_samplerate  # Update duration with new sample rate
                augmentation_method = f'resampled_to_{new_samplerate}hz'

                # Update total duration
                total_duration += new_duration
                total_durations[pathology] += new_duration

                # Extract Group from VVPD Records for the augmented file
                row = vvpd_records[vvpd_records['segment_filename'] == file_name]
                group = row['Group'].iloc[0] if not row.empty else 'Unknown'

                # Record the augmented file information
                file_info.append({
                    'file_name': new_file_name,
                    'pathology': pathology,
                    'duration_seconds': new_duration,
                    'samplerate': new_samplerate,
                    'augmentation_method': augmentation_method,
                    'segment_filename': file_name,
                    'Group': group
                })

                # Record used sampling rate
                used_resampling_rates[file_name].add(new_samplerate)

                print(f"Applied {augmentation_method} to {file_name}.")

            except Exception as e:
                print(f"Could not enhance file {file_path}, error: {e}")

# Update DataFrame and save
df = pd.DataFrame(file_info)
output_file_all = os.path.join(folder_path, 'audio_file_durations_augmented_with_group.xlsx')
df.to_excel(output_file_all, index=False)

# Print final total durations for each Pathology
print("\nFinal total durations for each Pathology:")
for pathology, total_duration in total_durations.items():
    print(f"{pathology}: {total_duration:.2f} seconds")

# Save final total duration information to the same file
with open(output_durations, 'a') as f:
    f.write("\nFinal total durations:\n")
    for pathology, total_duration in total_durations.items():
        f.write(f"{pathology}: {total_duration:.2f} seconds\n")

print(f"\nFinal total durations have been saved to {output_durations}")
print(f"All audio file information (including augmented samples) has been saved to {output_file_all}")
