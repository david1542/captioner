import argparse
import os.path
import shutil
from pathlib import Path

import requests
from zipfile import ZipFile
from io import BytesIO

from config.paths import DATA_PATH, FLICKER_TEXT_PATH, FLICKER_IMAGES_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
from config.urls import CAPTIONS_URL, IMAGES_URL


def download_zip(url: str, output_path: Path):
    response = requests.get(url, stream=True)
    zip_file = ZipFile(BytesIO(response.content))
    zip_file.extractall(output_path)


def download_data(clean=False):
    if clean and os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        shutil.rmtree(PROCESSED_DATA_PATH)
        os.mkdir(PROCESSED_DATA_PATH)

    download_zip(CAPTIONS_URL, FLICKER_TEXT_PATH)
    download_zip(IMAGES_URL, RAW_DATA_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-data', action='store_true')

    args = parser.parse_args()
    download_data(clean=args.clean_data)
