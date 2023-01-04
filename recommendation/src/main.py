from typing import Any, Dict

import click
from src.algorithms.association_recommender import AssociationRecommender
from src.algorithms.popularity_recommender import PopularityRecommender
from src.algorithms.random_recommender import RandomRecommender
from src.algorithms.regression_recommendation import RegressionRecommendation
from src.algorithms.umcf_recommender import UMCFRecommender
from src.utils import download, small_ratings
from src.utils.logger import configure_logger

logger = configure_logger(__name__)


@click.group()
def cli():
    pass


@click.command()
def download_command():
    logger.info("download")
    download.download()


@click.command()
@click.option(
    "--rate",
    "rate",
    type=float,
    default=0.1,
)
def small_rating_command(rate: float = 0.1):
    logger.info("select ratings")
    small_ratings.make_small_ratings(rate=rate)


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
    recommender = RandomRecommender(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    recommender.run_sample(k=obj.get("top_k", 10))
    logger.info("done random recommendation")


@click.command()
@click.pass_obj
@click.option(
    "--minimum_num_rating",
    "minimum_num_rating",
    type=int,
    default=200,
)
def popularity_recommend(
    obj: Dict[str, Any],
    minimum_num_rating: int,
):
    logger.info("popularity recommendation")
    recommender = PopularityRecommender(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    recommender.run_sample(
        k=obj.get("top_k", 10),
        minimum_num_rating=minimum_num_rating,
    )
    logger.info("done popularity recommendation")


@click.command()
@click.pass_obj
@click.option(
    "--min_support",
    "min_support",
    type=float,
    default=0.1,
)
@click.option(
    "--min_threshold",
    "min_threshold",
    type=float,
    default=1.0,
)
def association_recommend(
    obj: Dict[str, Any],
    min_support: float,
    min_threshold: float,
):
    logger.info("association recommendation")
    recommender = AssociationRecommender(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    recommender.run_sample(
        k=obj.get("top_k", 10),
        min_support=min_support,
        min_threshold=min_threshold,
    )
    logger.info("done association recommendation")


@click.command()
@click.pass_obj
def umcf_recommend(obj: Dict[str, Any]):
    logger.info("umcf recommendation")
    recommender = UMCFRecommender(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    recommender.run_sample(
        k=obj.get("top_k", 10),
        top_k=obj.get("top_k", 10),
    )
    logger.info("done umcf recommendation")


@click.command()
@click.pass_obj
def regression_recommend(obj: Dict[str, Any]):
    logger.info("regression recommendation")
    recommender = RegressionRecommendation(
        num_users=obj.get("num_users", 1000),
        num_test_items=obj.get("num_test_items", 5),
    )
    recommender.run_sample(
        k=obj.get("top_k", 10),
        top_k=obj.get("top_k", 10),
    )
    logger.info("done regression recommendation")


if __name__ == "__main__":
    cli.add_command(download_command)
    cli.add_command(small_rating_command)
    recommend.add_command(random_recommend)
    recommend.add_command(popularity_recommend)
    recommend.add_command(association_recommend)
    recommend.add_command(umcf_recommend)
    recommend.add_command(regression_recommend)
    cli.add_command(recommend)
    cli()
