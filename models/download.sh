#!/bin/sh
set -x

# TinyStyler
mkdir tinystyler
gdown https://drive.google.com/file/d/17_nUvgnUVAdL6dmTQL0Nx6fKN7d1giY3/view?usp=sharing --fuzzy -O tinystyler/model_weights.pt

# TinyStyler_Sim
mkdir tinystyler_sim
gdown https://drive.google.com/file/d/19gJJLoD3up4Nvkt_oY4XezgiNNyFD6Yh/view?usp=sharing --fuzzy -O tinystyler_sim/model_weights.pt

# TinyStyler_Recon
mkdir tinystyler_recon
gdown --folder https://drive.google.com/drive/folders/14RVtTTed8T1qWeq9LsbOltPzwcB6QLU4?usp=sharing --fuzzy -O tinystyler_recon
