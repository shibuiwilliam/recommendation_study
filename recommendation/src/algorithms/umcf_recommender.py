from collections import defaultdict
from typing import List

import numpy as np
from src.algorithms.base_recommender import BaseRecommender
from src.models.dataset import Dataset, RecommendResult
from surprise import Dataset as SurpriseDataset
from surprise import KNNWithMeans, Reader
from surprise.trainset import Trainset


class UMCFRecommender(BaseRecommender):
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
        self.knn: KNNWithMeans = None
        self.data_train: Trainset = None

        np.random.seed(0)
        self.logger.info("initialized umcf recommender")

    def train(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        reader = Reader(rating_scale=(0.5, 5))
        self.data_train = SurpriseDataset.load_from_df(
            dataset.train[["user_id", "movie_id", "rating"]],
            reader,
        ).build_full_trainset()

        sim_options = {
            "name": "pearson",
            "user_based": True,
        }
        self.knn = KNNWithMeans(
            k=30,
            min_k=1,
            sim_options=sim_options,
        )
        self.knn.fit(self.data_train)

    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        self.logger.info("start recommendation")

        top_k = kwargs.get("top_k", 10)

        self.train(
            dataset=dataset,
            **kwargs,
        )

        user_movie_matrix = dataset.train.pivot(
            index="user_id",
            columns="movie_id",
            values="rating",
        )
        user_id2index = dict(
            zip(
                user_movie_matrix.index,
                range(len(user_movie_matrix.index)),
            )
        )
        movie_id2index = dict(
            zip(
                user_movie_matrix.columns,
                range(len(user_movie_matrix.columns)),
            )
        )

        data_test = self.data_train.build_anti_testset(None)
        predictions = self.knn.test(data_test)

        def get_top_n(
            preds: List,
            n=10,
        ):
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in preds:
                top_n[uid].append((iid, est))

            for uid, user_ratings in top_n.items():
                user_ratings.sort(
                    key=lambda x: x[1],  # type: ignore
                    reverse=True,
                )
                top_n[uid] = [d[0] for d in user_ratings[:n]]

            return top_n

        pred_user2items = get_top_n(
            preds=predictions,
            n=top_k,
        )

        average_score = dataset.train.rating.mean()
        pred_results = []
        for _, row in dataset.test.iterrows():
            if row["user_id"] not in user_id2index or row["movie_id"] not in movie_id2index:
                pred_results.append(average_score)
                continue
            pred_score = self.knn.predict(
                uid=row["user_id"],
                iid=row["movie_id"],
            ).est
            pred_results.append(pred_score)
        dataset.test["rating_pred"] = pred_results

        recommendation = RecommendResult(
            rating=dataset.test.rating_pred,
            user2items=pred_user2items,
        )

        self.logger.info("done recommendation")
        return recommendation
