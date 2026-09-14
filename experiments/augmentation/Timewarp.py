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
from itertools import permutations

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Crossfade range (32 samples)
crossfade_samples = 32

# Folder paths
folder_path = r'path\to\audios'
vvpd_records_path = r'path\to\train_set.xlsx'

# Pathology folders
pathology_folders = ["ALS", "Covid-19", "Dysphonie", "Laryngitis", "Parkinson", "Rekurrensparese", "HC"]

# Initialize a dictionary to store the total duration for each pathology
total_durations = {pathology: 0 for pathology in pathology_folders}
file_info = []

# Load VVPD Records
vvpd_records = pd.read_excel(vvpd_records_path)

# Step 1: Traverse all subfolders and files to calculate total duration
for subfolder in total_durations.keys():
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.exists(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(subfolder_path, file_name)
                try:
                    with sf.SoundFile(file_path) as audio_file:
                        duration = len(audio_file) / audio_file.samplerate
                        total_durations[subfolder] += duration

                        # Extract ID from original_file_name (before `_segment_`)
                        if '_segment_' in file_name:
                            extracted_id = file_name.split('_segment_')[0]
                        else:
                            extracted_id = file_name.split('.')[0]  # Fallback to entire name without extension

                        # Find matching record in VVPD Records
                        record = vvpd_records[vvpd_records['segment_filename'] == file_name]

                        # Get Pathology, Dataset, and Group values
                        pathology = record.iloc[0]['Pathology'] if not record.empty else 'Unknown'
                        dataset = record.iloc[0]['Dataset'] if not record.empty else 'Unknown'
                        group = record.iloc[0]['Group'] if not record.empty else subfolder

                        # Add original file info to file_info list
                        file_info.append({
                            'segment_filename': file_name,
                            'original_file_name': file_name,  # Original file name
                            'Group': group,  # Use Group from the original file
                            'duration_seconds': duration,
                            'samplerate': audio_file.samplerate,
                            'augmentation_method': 'original',
                            'Pathology': pathology,
                            'Dataset': dataset
                        })
                except Exception as e:
                    print(f"Could not read file {file_path}, error: {e}")

# Save initial total duration information
output_durations = os.path.join(folder_path, 'total_durations.txt')
with open(output_durations, 'w') as f:
    f.write("Initial total durations:\n")
    for subfolder, total_duration in total_durations.items():
        f.write(f"{subfolder}: {total_duration:.2f} seconds\n")

print(f"\nInitial total durations have been saved to {output_durations}")

# Find the longest folder duration
max_duration = max(total_durations.values())

# Function to apply crossfade between segments
def crossfade_segments(segments, fade_samples=64):
    """
    Apply crossfade when concatenating segments.
    """
    output = segments[0]  # Start with the first segment
    for i in range(1, len(segments)):
        # Extract the overlapping regions
        previous_end = output[-fade_samples:]
        current_start = segments[i][:fade_samples]
        
        # Create crossfade
        crossfade = np.linspace(1, 0, fade_samples) * previous_end + np.linspace(0, 1, fade_samples) * current_start
        
        # Combine the segments with the crossfade
        output = np.concatenate([output[:-fade_samples], crossfade, segments[i][fade_samples:]])
    return output

# Step 2: Perform data augmentation using time warping for folders with insufficient duration
used_permutations = {}  # Dictionary to track used permutations for each file

for subfolder, total_duration in total_durations.items():
    if total_duration < max_duration:
        subfolder_path = os.path.join(folder_path, subfolder)
        files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]

        while total_duration < max_duration:
            # Randomly select a file for augmentation
            file_name = np.random.choice(files)
            file_path = os.path.join(subfolder_path, file_name)

            try:
                # Read audio data
                data, samplerate = sf.read(file_path)

                # Skip audio with insufficient length
                if len(data) <= 5:
                    print(f"Skipping enhancement for {file_name} due to insufficient data length.")
                    continue

                # Time warping: ensure the audio is long enough
                num_segments = 5
                segment_length = len(data) // num_segments
                if segment_length == 0:
                    print(f"Skipping time warp for {file_name} due to insufficient length.")
                    continue

                # Split data into segments and calculate all unique permutations
                segments = [data[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
                unique_permutations = list(permutations(range(num_segments)))

                # Initialize used_permutations entry if not present
                if file_name not in used_permutations:
                    used_permutations[file_name] = set()

                # Filter out already used permutations
                available_permutations = [
                    perm for perm in unique_permutations
                    if perm not in used_permutations[file_name]
                ]

                # Stop if no new permutations are available
                if not available_permutations:
                    print(f"No new permutations available for {file_name}. Skipping further enhancement.")
                    break

                # Randomly choose a new permutation
                chosen_permutation = np.random.choice(len(available_permutations))
                permuted_segments = [segments[i] for i in available_permutations[chosen_permutation]]

                # Record the chosen permutation
                used_permutations[file_name].add(available_permutations[chosen_permutation])

                # Apply crossfade between segments
                warped_data = crossfade_segments(permuted_segments, fade_samples=crossfade_samples)

                # Generate new file name with the order of segments
                permutation_order = ''.join(str(i + 1) for i in available_permutations[chosen_permutation])
                new_file_name = f"{file_name.split('.')[0]}_timewarp_{permutation_order}.wav"
                new_file_path = os.path.join(subfolder_path, new_file_name)

                # Write the warped audio data
                sf.write(new_file_path, warped_data, samplerate)
                new_duration = len(warped_data) / samplerate

                # Update total duration
                total_duration += new_duration
                total_durations[subfolder] += new_duration

                # Retrieve original file info to copy Pathology, Dataset, and Group
                original_file_info = next((item for item in file_info if item['segment_filename'] == file_name), None)
                if original_file_info:
                    pathology = original_file_info['Pathology']
                    dataset = original_file_info['Dataset']
                    group = original_file_info['Group']
                else:
                    pathology = 'Unknown'
                    dataset = 'Unknown'
                    group = subfolder

                # Record the augmented file information
                file_info.append({
                    'segment_filename': new_file_name,
                    'original_file_name': file_name,
                    'Group': group,
                    'duration_seconds': new_duration,
                    'samplerate': samplerate,
                    'augmentation_method': f'time_warp_{permutation_order}',
                    'Pathology': pathology,
                    'Dataset': dataset
                })

                print(f"Applied time warp with crossfade (order: {permutation_order}) to {file_name}.")

            except Exception as e:
                print(f"Could not enhance file {file_path}, error: {e}")

# Save combined DataFrame
df = pd.DataFrame(file_info)
output_file_all = os.path.join(folder_path, 'audio_file_durations_augmented_combined.xlsx')
df.to_excel(output_file_all, index=False)

# Save final total durations to a file
final_durations_path = os.path.join(folder_path, 'final_total_durations.txt')
with open(final_durations_path, 'w') as f:
    f.write("Final total durations after balancing:\n")
    for subfolder, total_duration in total_durations.items():
        f.write(f"{subfolder}: {total_duration:.2f} seconds\n")

print(f"\nFinal file saved: {output_file_all}")
print(f"\nFinal total durations have been saved to {final_durations_path}")
