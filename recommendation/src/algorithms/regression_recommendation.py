import itertools
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.algorithms.base_recommender import BaseRecommender
from src.models.dataset import Dataset, RecommendResult


class RegressionRecommendation(BaseRecommender):
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
        self.reg: RandomForestRegressor = None
        np.random.seed(0)
        self.logger.info("initialized regression recommender")

    def train(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        self.reg = RandomForestRegressor(
            n_jobs=-1,
            random_state=0,
        )
        self.reg.fit(
            self.train_x.values,
            self.train_y,
        )

    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        self.logger.info("start recommendation")

        user_movie_matrix = dataset.train.pivot(
            index="user_id",
            columns="movie_id",
            values="rating",
        )

        self.train_x = dataset.train[["user_id", "movie_id"]]
        self.train_y = dataset.train.rating.values

        test_x = dataset.test[["user_id", "movie_id"]]
        train_all_x = user_movie_matrix.stack(dropna=False).reset_index()[["user_id", "movie_id"]]
        train_all_key = train_all_x.copy()

        aggregators = ["min", "max", "mean"]
        user_features = dataset.train.groupby("user_id").rating.agg(aggregators).to_dict()
        movie_features = dataset.train.groupby("movie_id").rating.agg(aggregators).to_dict()
        for agg in aggregators:
            self.train_x[f"u_{agg}"] = self.train_x["user_id"].map(user_features[agg])
            test_x[f"u_{agg}"] = test_x["user_id"].map(user_features[agg])
            train_all_x[f"u_{agg}"] = train_all_x["user_id"].map(user_features[agg])
            self.train_x[f"m_{agg}"] = self.train_x["movie_id"].map(movie_features[agg])
            test_x[f"m_{agg}"] = test_x["movie_id"].map(movie_features[agg])
            train_all_x[f"m_{agg}"] = train_all_x["movie_id"].map(movie_features[agg])

        average_rating = self.train_y.mean()
        test_x.fillna(average_rating, inplace=True)

        movie_genres = dataset.item_content[["movie_id", "genre"]]
        genres = set(list(itertools.chain(*movie_genres.genre)))
        for genre in genres:
            movie_genres[f"is_{genre}"] = movie_genres.genre.apply(lambda x: genre in x)
        movie_genres.drop("genre", axis=1, inplace=True)

        self.train_x = self.train_x.merge(
            movie_genres,
            on="movie_id",
        ).drop(columns=["user_id", "movie_id"])
        test_x = test_x.merge(
            movie_genres,
            on="movie_id",
        ).drop(columns=["user_id", "movie_id"])
        train_all_x = train_all_x.merge(
            movie_genres,
            on="movie_id",
        ).drop(columns=["user_id", "movie_id"])

        self.train(
            dataset=dataset,
            **kwargs,
        )

        test_pred = self.reg.predict(test_x.values)

        test_x["rating_pred"] = test_pred

        train_all_pred = self.reg.predict(train_all_x.values)

        train_all_key["rating_pred"] = train_all_pred
        pred_matrix = train_all_key.pivot(
            index="user_id",
            columns="movie_id",
            values="rating_pred",
        )

        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            movie_indexes = np.argsort(-pred_matrix.loc[user_id, :]).values  # type: ignore
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in (user_evaluated_movies[user_id]):
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        recommendation = RecommendResult(
            rating=test_x.rating_pred,
            user2items=pred_user2items,
        )

        self.logger.info("done recommendation")
        return recommendation
