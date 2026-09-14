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
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

# Paths and parameters
audio_dirs = [
    r'path\to\complete_audios',
]
output_base_dir = r'path\to\output\pieces'
os.makedirs(output_base_dir, exist_ok=True)
metadata_path = r'path\to\output\VVPD_Records_cleaned.csv'
segment_duration = 1.0  # 1 sec
hop_duration = 0.4  # 0.4 sec

metadata = pd.read_csv(metadata_path)

# some complicated operations for ingtegrating data in table
metadata['Clean_ID'] = metadata.apply(
    lambda row: str(row['ID']).lstrip('0') if row['Dataset'] == 'ALS' else str(row['ID']),
    axis=1
)
metadata['Clean_ID'] = metadata['Clean_ID'].str.strip()
if 'Group' not in metadata.columns:
    metadata['Group'] = metadata['Dataset'].str.strip()
segment_metadata = []

# iterate through directories and process wav files
for audio_dir in audio_dirs:
    folder = os.path.basename(audio_dir)
    output_dir = os.path.join(output_base_dir, f'{folder}_segments')
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            try:
                # load audio file
                y, sr = librosa.load(filepath, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                clean_filename = os.path.splitext(filename)[0].split('_concatenated_crossfade')[0].strip()
                if clean_filename.isdigit(): 
                    clean_filename = clean_filename.lstrip('0')
                original_metadata = metadata[
                    (metadata['Clean_ID'] == clean_filename) & 
                    (metadata['Group'] == folder)
                ]
                if original_metadata.empty:
                    print(f"No metadata found for: {filename} in group: {folder}")
                    continue
                num_segments = int(np.ceil((duration - segment_duration) / hop_duration)) + 1
                for i in range(num_segments):
                    start_time = i * hop_duration
                    end_time = start_time + segment_duration

                    # Handle the case where end_time exceeds the duration
                    if end_time > duration:
                        segment = np.pad(y[int(start_time * sr):], 
                                         (0, int((end_time - duration) * sr)), 
                                         mode='wrap')
                    else:
                        segment = y[int(start_time * sr):int(end_time * sr)]

                    segment_group_suffix = f"_{folder}" if original_metadata['Dataset'].iloc[0] == 'ALS' else ""
                    segment_filename = f"{clean_filename}_segment{segment_group_suffix}_{i + 1}.wav"
                    segment_filepath = os.path.join(output_dir, segment_filename)
                    sf.write(segment_filepath, segment, sr)

                    record = original_metadata.iloc[0].to_dict()
                    record.update({
                        'segment_filename': segment_filename,
                        'segment_duration_seconds': segment_duration,
                        'samplerate': sr,
                        'segment_index': i + 1,
                        'Group': folder 
                    })
                    segment_metadata.append(record)

            except Exception as e:
                print(f"Error processing {filename} in {folder}: {e}")

segment_metadata_df = pd.DataFrame(segment_metadata)
segment_metadata_df['NPY'] = segment_metadata_df['segment_filename'].str.replace('.wav', '.npy', regex=False)
metadata_output_path = os.path.join(output_base_dir, 'segment_metadata.xlsx')
segment_metadata_df.to_excel(metadata_output_path, index=False)

print("Segmentation complete for all folders. Metadata saved at:", metadata_output_path)
