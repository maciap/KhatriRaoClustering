"""
Khatri-Rao K-Means wrapper

Expected modules:
- kr_k_means_time_efficient  -> class KrKMeans
- kr_k_means_space_efficient -> class KrKMeansV2
- kr_k_means_p_sets          -> class KrKMeansP

Dataset loading utilities (expected):
- scripts/run_experiments_utils.py providing:
    - load_data(dataset: str) -> (X, L)
    - closest_factors(n_labels: int) -> (h1, h2)

Defaults:
- impl = "space"
- standardize = False
- for 2-set impls: if h1/h2 omitted, use closest_factors(n_clusters)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from html import parser
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from KathriRaokMeans.kr_k_means import KrKMeans as KrKMeansTE
from KathriRaokMeans.kr_k_means_space_efficient import KrKMeans as KrKMeansSE
from KathriRaokMeans.kr_k_means_p_sets import KrKMeans as KrKMeansPsets

from scripts.run_experiments_utils import load_data, closest_factors


@dataclass
class FitResult:
    params: Any
    loss: float
    idxs_all: np.ndarray


class KRKMeansWrapper:
    def __init__(
        self,
        impl: str = "space",
        *,
        operator: str = "product",
        standardize: bool = True,
    ):
        impl = impl.lower().strip()
        if impl not in {"space", "time", "p"}:
            raise ValueError("impl must be one of: 'space', 'time', 'p'")
        if operator not in {"product", "sum"}:
            raise ValueError("operator must be 'product' or 'sum'")

        self.impl = impl
        self.operator = operator
        self.standardize = bool(standardize)

        self.model = None
        self.X = None
        self.L = None

    def load_dataset(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        X, L = load_data(dataset)
        self.X = np.asarray(X)
        self.L = np.asarray(L)
        return self.X, self.L

    @staticmethod
    def _infer_h1_h2_from_labels(L: np.ndarray) -> Tuple[int, int]:
        n_clusters = int(np.unique(L).shape[0])
        h1, h2 = closest_factors(n_clusters)
        return int(h1), int(h2)

    @staticmethod
    def _parse_h_list(h_list: Union[str, Sequence[int]]) -> List[int]:
        if isinstance(h_list, str):
            s = h_list.strip()
            if not s:
                return []
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        return [int(x) for x in h_list]

    def build(
        self,
        *,
        dataset: str,
        h1: Optional[int] = None,
        h2: Optional[int] = None,
        h_list: Optional[Union[str, Sequence[int]]] = None,
    ) -> Any:
        X, L = self.load_dataset(dataset)

        if self.impl in {"time", "space"}:
            if h1 is None or h2 is None:
                h1_inf, h2_inf = self._infer_h1_h2_from_labels(L)
                h1 = h1_inf if h1 is None else int(h1)
                h2 = h2_inf if h2 is None else int(h2)

            if self.impl == "time":
                self.model = KrKMeansTE(
                    X,
                    int(h1),
                    int(h2),
                    standardize=self.standardize,
                    operator=self.operator,
                )
            else:
                self.model = KrKMeansSE(
                    X,
                    int(h1),
                    int(h2),
                    standardize=self.standardize,
                    operator=self.operator,
                )

        else:
            if h_list is None:
                raise ValueError("For impl='p', provide h_list (e.g. '6,6,6')")
            h_list_parsed = self._parse_h_list(h_list)
            if len(h_list_parsed) == 0:
                raise ValueError("h_list must contain at least one integer")

            self.model = KrKMeansPsets(
                X,
                h_list_parsed,
                standardize=self.standardize,
                operator=self.operator,
            )

        return self.model

    def fit(
        self,
        *,
        n_iter: int,
        th_movement: float = 1e-4,
        verbose: Union[bool, int] = False,
        init_type: Union[str, list] = "random",
    ) -> FitResult:
        if self.model is None:
            raise RuntimeError("Call build(...) before fit(...)")

        v = 2 if verbose is True else (int(verbose) if isinstance(verbose, int) else 0)
        params, loss, idxs_all = self.model.fit(
            n_iter=n_iter,
            th_movement=th_movement,
            verbose=v,
            init_type=init_type,
        )
        return FitResult(params=params, loss=float(loss), idxs_all=np.asarray(idxs_all))

    def predict(self, new_data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call build(...) before predict(...)")
        return np.asarray(self.model.predict(new_data)).flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Khatri-Rao K-Means with a selectable implementation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--impl", type=str, default="space", choices=["space", "time", "p"])
    parser.add_argument("--operator", type=str, default="product", choices=["product", "sum"])

    parser.add_argument("--no-standardize",dest="standardize",action="store_false",help="Disable feature standardization",)
    parser.set_defaults(standardize=True)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--th_movement", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--init_type", type=str, default="random")

    parser.add_argument("--h1", type=int, default=None)
    parser.add_argument("--h2", type=int, default=None)
    parser.add_argument("--h_list", type=str, default="")

    args = parser.parse_args()

    wrapper = KRKMeansWrapper(
        impl=args.impl,
        operator=args.operator,
        standardize=bool(args.standardize),
    )

    if args.impl in {"space", "time"}:
        wrapper.build(dataset=args.dataset, h1=args.h1, h2=args.h2)
    else:
        wrapper.build(dataset=args.dataset, h_list=args.h_list)

    out = wrapper.fit(
        n_iter=args.n_iter,
        th_movement=args.th_movement,
        verbose=args.verbose,
        init_type=args.init_type,
    )

    print(f"Terminated. final loss: {out.loss}")


if __name__ == "__main__":
    main()
