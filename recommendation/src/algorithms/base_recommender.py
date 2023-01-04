from abc import ABC, abstractmethod

from src.models.dataset import DataLoader, Dataset, RecommendResult
from src.models.metrics import MetricCalculator
from src.utils.logger import configure_logger


class BaseRecommender(ABC):
    def __init__(
        self,
        num_users: int = 1000,
        num_test_items: int = 5,
        data_path: str = "data/ml-10M100K/",
    ):
        self.logger = configure_logger(__name__)
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path
        self.data_loader = DataLoader(
            num_users=self.num_users,
            num_test_items=self.num_test_items,
            data_path=self.data_path,
        )
        self.metric_calculator = MetricCalculator()
        self.logger.info("initialized base recommender")

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> RecommendResult:
        raise NotImplementedError

    def run_sample(
        self,
        k: int = 10,
        **kwargs,
    ) -> None:
        movielens = self.data_loader.load()
        recommend_result = self.recommend(
            dataset=movielens,
            **kwargs,
        )
        metrics = self.metric_calculator.calculate(
            true_rating=movielens.test.rating.tolist(),
            pred_rating=recommend_result.rating.tolist(),
            true_user2items=movielens.test_user2items,
            pred_user2items=recommend_result.user2items,
            k=k,
        )
        self.logger.info(
            f"""
RESULT:
    RMSE: {metrics.rmse:.3f}
    PRECISION@{k}: {metrics.precision_at_k.precision:.3f}
    RECALL@{k}: {metrics.recall_at_k.recall:.3f}
        """
        )
