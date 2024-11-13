#!/bin/sh
set -ex

MODEL_DOWNLOAD_DIR="$(dirname "$0")" # Feel free to change this to your preferred location

gdown https://drive.google.com/drive/folders/1ThlK2oeBBaclWGEX5eb9fJeaZ4Oo53z9?usp=sharing --fuzzy -O ${MODEL_DOWNLOAD_DIR}
