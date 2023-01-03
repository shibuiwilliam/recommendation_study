from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_squared_error
from src.utils.logger import configure_logger


@dataclass(frozen=True)
class PrecisionAtK:
    precision: float
    k: int


@dataclass(frozen=True)
class RecallAtK:
    recall: float
    k: int


@dataclass(frozen=True)
class Metrics:
    rmse: float
    precision_at_k: PrecisionAtK
    recall_at_k: RecallAtK

    def __repr__(self):
        return f"""rmse={self.rmse:.3f}
Precision@K={self.precision_at_k.precision:.3f}
Recall@K={self.recall_at_k.recall:.3f}"""


class MetricCalculator(object):
    def __init__(self):
        self.logger = configure_logger(__name__)
        self.logger.info("initialized metric calculator")

    def calculate(
        self,
        true_rating: List[float],
        pred_rating: List[float],
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        rmse = self.calculate_rmse(
            true_rating=true_rating,
            pred_rating=pred_rating,
        )
        precision_at_k = self.calculate_precision_at_k(
            true_user2items=true_user2items,
            pred_user2items=pred_user2items,
            k=k,
        )
        recall_at_k = self.calculate_recall_at_k(
            true_user2items=true_user2items,
            pred_user2items=pred_user2items,
            k=k,
        )
        return Metrics(
            rmse=rmse,
            precision_at_k=PrecisionAtK(
                precision=precision_at_k,
                k=k,
            ),
            recall_at_k=RecallAtK(
                recall=recall_at_k,
                k=k,
            ),
        )

    def precision_at_k(
        self,
        true_items: List[int],
        pred_items: List[int],
        k: int,
    ) -> float:
        if k < 1:
            raise ValueError

        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    def recall_at_k(
        self,
        true_items: List[int],
        pred_items: List[int],
        k: int,
    ) -> float:
        if k < 1:
            raise ValueError

        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
        return r_at_k

    def calculate_rmse(
        self,
        true_rating: List[float],
        pred_rating: List[float],
    ) -> float:
        return np.sqrt(mean_squared_error(true_rating, pred_rating))

    def calculate_recall_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        if k < 1:
            raise ValueError
        scores = []
        for user_id in true_user2items.keys():
            r_at_k = self.recall_at_k(
                true_items=true_user2items[user_id],
                pred_items=pred_user2items[user_id],
                k=k,
            )
            scores.append(r_at_k)
        return np.mean(scores)

    def calculate_precision_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        if k < 1:
            raise ValueError
        scores = []
        for user_id in true_user2items.keys():
            p_at_k = self.precision_at_k(
                true_items=true_user2items[user_id],
                pred_items=pred_user2items[user_id],
                k=k,
            )
            scores.append(p_at_k)
        return np.mean(scores)
