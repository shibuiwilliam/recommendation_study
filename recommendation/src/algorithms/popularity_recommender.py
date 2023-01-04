from collections import defaultdict

import numpy as np
from src.algorithms.base_recommender import BaseRecommender
from src.models.dataset import Dataset, RecommendResult


class PopularityRecommender(BaseRecommender):
    def __init__(
        self,
        num_users: int = 1000,
        num_test_items: int = 5,
        data_path: str = "data/ml-10M100K/",
    ):
        super().__init__(
            num_users=num_users,
            num_test_items=num_test_items,
            data_path=data_path,
        )
        np.random.seed(0)
        self.logger.info("initialized popularity recommender")

    def train(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)

        movie_stats = dataset.train.groupby("movie_id").agg({"rating": [np.size, np.mean]})
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating
        self.movies_sorted_by_rating = (
            movie_stats[atleast_flg]
            .sort_values(
                by=("rating", "mean"),
                ascending=False,
            )
            .index.tolist()
        )

    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        self.logger.info("start recommendation")
        self.train(
            dataset=dataset,
            **kwargs,
        )

        user_watched_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        pred_user2items = defaultdict(list)
        for user_id in dataset.train.user_id.unique():
            for movie_id in self.movies_sorted_by_rating:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating": np.mean})
        movie_rating_predict = dataset.test.merge(
            movie_rating_average,
            on="movie_id",
            how="left",
            suffixes=("_test", "_pred"),
        ).fillna(0)

        recommendation = RecommendResult(
            rating=movie_rating_predict.rating_pred,
            user2items=pred_user2items,
        )
        self.logger.info("done recommendation")
        return recommendation
