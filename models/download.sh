#!/bin/sh
set -ex

# TinyStyler
mkdir tinystyler
#gdown FILL IN (EMNLP wifi slow) --fuzzy -O tinystyler/model_weights.pt

# TinyStyler_Sim
mkdir tinystyler_sim
gdown https://drive.google.com/file/d/19gJJLoD3up4Nvkt_oY4XezgiNNyFD6Yh/view?usp=drive_link --fuzzy -O tinystyler_sim/model_weights.pt

# TinyStyler_Recon
mkdir tinystyler_recon
gdown --folder https://drive.google.com/drive/folders/14RVtTTed8T1qWeq9LsbOltPzwcB6QLU4?usp=sharing --fuzzy -O tinystyler_recon
