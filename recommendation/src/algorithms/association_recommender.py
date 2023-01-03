from collections import Counter, defaultdict

import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from src.algorithms.base_recommender import BaseRecommender
from src.models.dataset import Dataset, RecommendResult


class AssociationRecommender(BaseRecommender):
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
        self.logger.info("initialized association recommender")

    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        self.logger.info("start recommendation")
        min_support = kwargs.get("min_support", 0.1)
        min_threshold = kwargs.get("min_threshold", 1)

        user_movie_matrix = dataset.train.pivot(
            index="user_id",
            columns="movie_id",
            values="rating",
        )

        user_movie_matrix[user_movie_matrix < 4] = 0
        user_movie_matrix[user_movie_matrix.isnull()] = 0
        user_movie_matrix[user_movie_matrix >= 4] = 1

        freq_movies = apriori(
            user_movie_matrix,
            min_support=min_support,
            use_colnames=True,
        )

        rules = association_rules(
            freq_movies,
            metric="lift",
            min_threshold=min_threshold,
        )

        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]

        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1

            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])
            counter = Counter(consequent_movies)
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        recommendation = RecommendResult(
            rating=dataset.test.rating,
            user2items=pred_user2items,
        )
        self.logger.info("done recommendation")
        return recommendation
