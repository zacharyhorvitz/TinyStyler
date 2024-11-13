#!/bin/sh
set -ex

MODEL_DOWNLOAD_DIR="$(dirname "$0")" # Feel free to change this to your preferred location

gdown https://drive.google.com/file/d/13lLj2fK8IycK_EYPJeO7TEmx9knFe0Cv/view?usp=sharing --fuzzy -O ${MODEL_DOWNLOAD_DIR}
