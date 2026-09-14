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
import soundfile as sf

def min_max_normalize_audio(input_file, output_file):

    data, samplerate = sf.read(input_file)
    
    # Check if the audio is mono or multi-channel
    if len(data.shape) > 1:  # Multi-channel audio
        normalized_data = []
        for channel in data.T:
            min_val = np.min(channel)
            max_val = np.max(channel)
            normalized_channel = (channel - min_val) / (max_val - min_val)
            normalized_data.append(normalized_channel)
        normalized_data = np.array(normalized_data).T
    else:  # Mono audio
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
    
    # save
    sf.write(output_file, normalized_data, samplerate)
    print(f"Audio {input_file} normalized and save to {output_file}")

input_dir = r'path\to\input_audios'  
output_dir = r'path\to\output_audios' 

for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        min_max_normalize_audio(input_path, output_path)
