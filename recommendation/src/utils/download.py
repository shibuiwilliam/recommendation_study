import os
import shutil

import httpx
from tqdm import tqdm

URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"


def download():
    target_directory = os.getenv("TARGET_DIRECTORY", "data/")
    target_file = os.path.join(target_directory, "ml-10m.zip")

    if not os.path.exists(target_file):
        with open(target_file, "wb") as download_file:
            with httpx.stream("GET", URL) as response:
                total = int(response.headers["Content-Length"])
                with tqdm(
                    total=total,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                ) as progress:
                    downloaded_bytes = response.num_bytes_downloaded
                    for chunk in response.iter_bytes():
                        download_file.write(chunk)
                        progress.update(response.num_bytes_downloaded - downloaded_bytes)
                        downloaded_bytes = response.num_bytes_downloaded

    shutil.unpack_archive(target_file, target_directory)
