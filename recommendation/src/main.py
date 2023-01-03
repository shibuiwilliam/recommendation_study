from typing import Any, Dict

import click
from src.algorithms.random_recommender import RandomRecommender
from src.utils import download
from src.utils.logger import configure_logger

logger = configure_logger(__name__)


@click.group()
def cli():
    pass


@click.command()
def download_command():
    logger.info("download")
    download.download()


@click.group()
@click.option(
    "--num_users",
    "num_users",
    type=int,
    default=1000,
)
@click.option(
    "--num_test_items",
    "num_test_items",
    type=int,
    default=5,
)
@click.option(
    "--top_k",
    "top_k",
    type=int,
    default=10,
)
@click.pass_context
def recommend(
    ctx,
    num_users: int,
    num_test_items: int,
    top_k: int,
):
    ctx.obj = dict(
        num_users=num_users,
        num_test_items=num_test_items,
        top_k=top_k,
    )


@click.command()
@click.pass_obj
def random_recommend(obj: Dict[str, Any]):
    logger.info("random recommendation")
    random_recommender = RandomRecommender(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    random_recommender.run_sample(k=obj.get("top_k", 10))
    logger.info("done random recommendation")


if __name__ == "__main__":
    logger.info("#### START ####")
    cli.add_command(download_command)
    recommend.add_command(random_recommend)
    cli.add_command(recommend)
    cli()
    logger.info("#### COMPLETED ####")
