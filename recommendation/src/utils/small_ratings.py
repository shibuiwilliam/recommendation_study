import os
from random import random

from src.utils.logger import configure_logger

logger = configure_logger(__name__)


def make_small_ratings(rate: float = 0.1):
    target_directory = os.getenv("TARGET_DIRECTORY", "data/ml-10M100K")
    original_file = os.path.join(target_directory, "ratings.dat")
    target_file = os.path.join(target_directory, f"small_rating_{rate}.dat")
    logger.info(
        f"""
make small rating file:
    rate: {rate}
    output file: {target_file}
    """
    )

    with open(original_file, "r") as f:
        lines = f.readlines()

    selected = []
    for l in lines:
        if random() < rate:
            selected.append(l)

    with open(target_file, "w") as f:
        f.write("".join(selected))

    logger.info("done")
