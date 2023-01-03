import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    test_user2items: Dict[int, List[int]]
    item_content: pd.DataFrame


@dataclass(frozen=True)
class RecommendResult:
    rating: pd.DataFrame
    user2items: Dict[int, List[int]]


class DataLoader(object):
    def __init__(
        self,
        num_users: int = 1000,
        num_test_items: int = 5,
        data_path: str = "../../data/ml-10M100K/",
    ):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )
        return Dataset(
            train=movielens_train,
            test=movielens_test,
            test_user2items=movielens_test_user2items,
            item_content=movie_content,
        )

    def _split_data(
        self,
        movielens: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(
            ascending=False,
            method="first",
        )
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        return movielens_train, movielens_test

    def _load(self) -> (pd.DataFrame, pd.DataFrame):
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            names=m_cols,
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))

        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"),
            names=t_cols,
            sep="::",
            engine="python",
        )

        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        movies = movies.merge(movie_tags, on="movie_id", how="left")

        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            names=r_cols,
            sep="::",
            engine="python",
        )

        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        movielens_ratings = ratings.merge(
            movies,
            on="movie_id",
        )

        return movielens_ratings, movies
