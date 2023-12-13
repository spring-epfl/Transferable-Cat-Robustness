import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import numpy as np
import pandas as pd
from dotmap import DotMap
from tqdm import autonotebook as tqdm

from exp.utils import clf_pgd_attack
from src.search import generalized_best_first_search
from src.search import IterLimitExceededError
from src.search import submod_greedy
from src.transformations import TransformationGenerator
from src.utils.counter import ExpansionCounter
from src.utils.hash import fast_hash

criteria = {
    "delta_score": lambda c: c.delta_score,
    "score": lambda c: c.score / c.cost,
}

# it is better to use decorators here
def conf_heuristic(x, es):
    if isinstance(x, list):
        conf = (
            1
            - es.run_params.clf.predict_proba(np.array(x))[
                :, es.run_params.target_class
            ]
        )
    else:
        conf = (
            1
            - es.run_params.clf.predict_proba(np.array([x]))[
                0, es.run_params.target_class
            ]
        )
    return conf


def rand_heuristic(x):
    if isinstance(x, list):
        return [np.random.rand() for _ in x]
    else:
        return np.random.rand()


class ExperimentSuite:
    def __init__(
        self,
        clf,
        X_test,
        y_test,
        target_class,
        spec,
        iter_lim=None,
        cost_bound=None,
        gain_col=None,
        dataset=None,
    ):
        self.clf = clf
        self.X_test = X_test
        self.y_test = y_test
        self.target_class = target_class
        self.spec = spec
        self.iter_lim = iter_lim
        self.cost_bound = cost_bound
        self.gain_col = gain_col
        self.dataset = dataset

    def run(self, attack_config):
        # Create a pool of initial examples.
        preds = self.clf.predict(self.X_test)
        is_source = self.y_test != self.target_class
        initial_pool = self.X_test[is_source]

        results = pd.DataFrame()
        it = tqdm.tqdm(list(initial_pool.iterrows()))
        total_gain = 0
        j = 0
        gcr = 0
        for i, x in it:
            if isinstance(self.cost_bound, str):
                tokens = self.cost_bound.split("-")
                if len(tokens) > 1:
                    gain_col, thresh = tokens
                    cost_bound = float(x[gain_col]) - float(thresh)
                else:
                    gain_col = tokens[0]
                    cost_bound = x[gain_col]
            else:
                cost_bound = self.cost_bound

            attack = Attack(
                attack_config,
                clf=self.clf,
                x=x,
                spec=self.spec,
                target_class=self.target_class,
                iter_lim=self.iter_lim,
                cost_bound=cost_bound,
                dataset=self.dataset,
            )

            start_time = time.time()
            if self.gain_col is not None:
                gain = x[self.gain_col]
            else:
                gain = 0
            try:
                with ExpansionCounter().as_default() as counter:
                    adv_x, cost = attack.run()

                if adv_x is not None and cost != 0:
                    j += 1
                    #print(x.compare(adv_x))
                    if (gain - cost) > 0:
                        total_gain += gain - cost
                    gcr = total_gain / j
                    it.set_description(
                        f"Found! {cost:.1f}, Gain: {gain:.2f}, ratio: {gcr:.2f}"
                    )
                elif adv_x is None:
                    it.set_description(
                        f"Not found,   Gain: {gain:.2f}, ratio: {gcr:.2f}"
                    )

            except IterLimitExceededError:
                adv_x = None
                cost = None
                it.set_description(f"Fail,   Gain: {gain:.2f}, ratio: {gcr:.2f}")

            time_passed = time.time() - start_time
            if attack is None:
                count = pgd_steps
            else:
                count = counter.count

            results = pd.concat([results,
                pd.DataFrame.from_records([dict(
                    attack=attack_config.name,
                    orig_index=i,
                    x=x,
                    adv_x=adv_x,
                    cost=cost,
                    time=time_passed,
                    num_expansions=count,
                )])]
                #ignore_index=True,
            )

        return results


@dataclass
class AttackConfig:
    name: str
    algo: str = "bfs"
    scoring: str = None
    heuristic: str = None
    beam_size: int = None
    kwargs: dict = field(default_factory=dict)


class Attack:
    def __init__(self, config, **run_params):
        self.config = replace(config)
        self.run_params = DotMap(run_params)
        self._search_args = self._get_search_args()

    def _get_search_args(self):
        eps = 10e-10

        if self.config.algo is None:
            self.config.algo = "bfs"

        # Implementations of the scoring functions.
        scoring_funcs = {
            "hc_ratio": lambda c, h, _: (1 - h) / (c + eps),
            "delta_hc_ratio": lambda c, h_new, h_old: ((1 - h_new) - (1 - h_old))
            / (c + eps),
            "ps": lambda c, h, _: (self.run_params.cost_bound - c) / (h + eps),
            "greedy": lambda c, h, _: -h,
            "a_star": lambda c, h, _: -(c + h),
        }

        # Set up for the linear heuristic.
        if "cost_coef" not in self.config.kwargs:
            self.config.kwargs["cost_coef"] = 1
        if "cost_min_step_value" not in self.config.kwargs:
            self.config.kwargs["cost_min_step_value"] = 0.0

        # Implementations of the heuristic functions.
        heuristic_funcs = {
            "zero": lambda _: 0,
            "constant": lambda _: 1,
            "random": rand_heuristic,
            "linear": (
                lambda x: max(
                    self.config.kwargs["cost_coef"]
                    * (
                        self.run_params.clf.predict_proba([x])[
                            0, self.run_params.target_class
                        ]
                        - 0.5
                    )
                    / (np.linalg.norm(self.run_params.clf.coef_[0], np.inf))
                    - self.config.kwargs["cost_min_step_value"],
                    0,
                )
            ),
            "confidence": lambda x: conf_heuristic(x, self),
        }

        hash_fn = lambda x: fast_hash(np.array(x.values, dtype=np.float32))
        goal_fn = (
            lambda x: self.run_params.clf.predict(np.array([x]))[0]
            == self.run_params.target_class
        )
        expand_fn = TransformationGenerator(self.run_params.spec)

        return dict(
            start_node=self.run_params.x,
            spec=self.run_params.spec,
            target_class=self.run_params.target_class,
            goal_fn=goal_fn,
            hash_fn=hash_fn,
            expand_fn=expand_fn,
            heuristic_fn=heuristic_funcs[self.config.heuristic or "confidence"],
            score_fn=scoring_funcs[self.config.scoring or "greedy"],
            beam_size=self.config.beam_size,
            admissible=(self.config.heuristic == "linear"),
            cost_bound=self.run_params.cost_bound,
            dataset=self.run_params.dataset,
            iter_lim=self.run_params.iter_lim,
            **self.config.kwargs,
        )

    def run(self, **kwargs):
        if self.config.algo == "bfs":
            return generalized_best_first_search(**self._search_args, **kwargs)
        if self.config.algo == "pgd":
            return clf_pgd_attack(
                clf=self.run_params.clf,
                x=self.run_params.x,
                y=self.run_params.target_class,
                eps=self.run_params.cost_bound,
                **self._search_args,
                **kwargs,
            )
        if self.config.algo == "Ballet":
            return clf_pgd_attack(
                clf=self.run_params.clf,
                x=self.run_params.x,
                y=self.run_params.target_class,
                eps=self.run_params.cost_bound,
                **self._search_args,
                **kwargs,
                attack_type="Ballet"
            )
        elif self.config.algo == "greedy":
            return submod_greedy(**self._search_args, **kwargs)
