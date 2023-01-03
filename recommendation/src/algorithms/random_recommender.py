from collections import defaultdict

import numpy as np
from src.algorithms.base_recommender import BaseRecommender
from src.models.dataset import Dataset, RecommendResult


class RandomRecommender(BaseRecommender):
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
        self.logger.info("initialized random recommender")

    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        self.logger.info("start recommendation")
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        pred_matrix = np.random.uniform(
            0.5,
            5.0,
            (len(unique_user_ids), len(unique_movie_ids)),
        )
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            if i % 100 == 0:
                self.logger.info(f"at {i} row")
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        pred_user2items = defaultdict(list)

        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for i, user_id in enumerate(unique_user_ids):
            if i % 100 == 0:
                self.logger.info(f"at {i} user")
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for j, movie_index in enumerate(movie_indexes):
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        recommendation = RecommendResult(
            rating=movie_rating_predict.rating_pred,
            user2items=pred_user2items,
        )
        self.logger.info("done recommendation")
        return recommendation
