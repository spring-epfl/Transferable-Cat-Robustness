from cmath import cos
from dataclasses import dataclass
from pathlib import Path

from dotmap import DotMap

from exp import loaders
from exp.framework import AttackConfig
from src.transformations import CategoricalFeature
from src.transformations import CategoricalMatrixFeature
from src.transformations import NumFeature
from src.transformations import BinaryFeature
import string

def get_experiment_path(results_dir, model_path, experiment_name, cost_bound, label="", embs="", tr=-1):
    model_name = Path(model_path.replace(".", "_")).stem
    if embs != "":
        emb_name = Path(embs).parts[-1]
        return Path(results_dir) / f"{model_name}+{emb_name}+{tr}_{experiment_name}_{cost_bound}_{label}"
    else:
        emb_name = ""
    return Path(results_dir) / f"{model_name}_{experiment_name}_{cost_bound}_{label}"


generic_experiments = [
    AttackConfig(name="random", scoring="greedy", heuristic="random", beam_size=1),
    AttackConfig(name="pgd", algo="pgd", kwargs=dict(steps=100)),
    AttackConfig(
        name="greedy", scoring="hc_ratio", heuristic="confidence", beam_size=1,
    ),
    AttackConfig(
        name="greedy_delta",
        scoring="delta_hc_ratio",
        heuristic="confidence",
        beam_size=1,
    ),
    # AttackConfig(
    #     name="greedy_beam10", scoring="hc_ratio", heuristic="confidence", beam_size=10,
    # ),
    # AttackConfig(
    #     name="greedy_beam100",
    #     scoring="hc_ratio",
    #     heuristic="confidence",
    #     beam_size=100,
    # ),
    # AttackConfig(
    #     name="greedy_full", scoring="hc_ratio", heuristic="confidence", beam_size=None,
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_s_at_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="score",
    #         change_feature_once=False,
    #         all_transformations=True,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_s_bt_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="score",
    #         change_feature_once=False,
    #         all_transformations=False,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_at_of",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=True,
    #         all_transformations=True,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_bt_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=False,
    #         all_transformations=False,
    #     ),
    # ),
    # AttackConfig(
    #     name="submod_greedy_es_ds_at_rf",
    #     algo="greedy",
    #     kwargs=dict(
    #         early_stop=True,
    #         criterion="delta_score",
    #         change_feature_once=False,
    #         all_transformations=True,
    #     ),
    # ),
]

common_a_star_family_experiments = [
    AttackConfig(
        name="astar_subopt_beam1",
        scoring="a_star",
        heuristic="confidence",
        beam_size=1,
    ),
    AttackConfig(
        name="astar_subopt_beam10",
        scoring="a_star",
        heuristic="confidence",
        beam_size=10,
    ),
    AttackConfig(
        name="astar_subopt_beam100",
        scoring="a_star",
        heuristic="confidence",
        beam_size=100,
    ),
    # AttackConfig(name="astar_subopt", scoring="a_star", heuristic="confidence"),
    AttackConfig(name="ucs", scoring="a_star", heuristic="zero"),
]

common_ps_family_experiments = [
    AttackConfig(
        name="ps_subopt_beam1", scoring="ps", heuristic="confidence", beam_size=1,
    ),
    AttackConfig(
        name="ps_subopt_beam10", scoring="ps", heuristic="confidence", beam_size=10,
    ),
    AttackConfig(
        name="ps_subopt_beam100", scoring="ps", heuristic="confidence", beam_size=100
    ),
    # AttackConfig(name="ps_subopt", scoring="ps", heuristic="confidence",),
]


def _get_working_datasets(data_test, target_col):
    X_test = data_test.X_test
    y_test = data_test.y_test
    orig_df = data_test.orig_df
    return DotMap(
        X_test=X_test,
        y_test=y_test,
        orig_y=orig_df[target_col],
        orig_cost=data_test.cost_orig_df,
        orig_df=orig_df,
    )


def get_dataset(dataset, data_dir, mode, seed=0, same_cost=False, cat_map=False, noise='0', w_max=0.0):
    if dataset == "ieeecis":
        data = loaders.IEEECISDataset(
            data_dir,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            seed=seed,
            same_cost=same_cost,
            cat_map=cat_map,
            w_max=w_max
        )

    elif dataset == "twitter_bot":
        data = loaders.TwitterBotDataset(
            data_dir,
            seed=seed,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            same_cost=same_cost,
            cat_map=cat_map,
        )

    elif dataset == "home_credit":
        data = loaders.HomeCreditDataset(
            data_dir,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            seed=seed,
            same_cost=same_cost,
            cat_map=cat_map,
        )
    elif dataset == "credit_sim":
        data = loaders.CreditCard(
            data_dir,
            mode=mode,
            balanced=True,
            discrete_one_hot=True,
            seed=seed,
            same_cost=same_cost,
            cat_map=cat_map,
            w_max=w_max
        )
    elif dataset == "syn":
        data = loaders.Synthetic(
            data_dir, mode=mode, seed=seed, same_cost=same_cost, cat_map=cat_map, noise=noise, w_max=0.0
        )
        #data = loaders.Synthetic(data_dir, mode=mode, seed=seed, same_cost=same_cost,)
    elif dataset == "lending_club":
        data = loaders.LendingClub(
            data_dir, mode=mode, seed=seed, same_cost=same_cost, cat_map=cat_map, w_max=w_max
        )
    elif dataset == "bank_fraud":
        data = loaders.BankFraud(
            data_dir, mode=mode, seed=seed, same_cost=same_cost, cat_map=cat_map, w_max=w_max
        )
    return data


def _get_working_datasets(data_test, target_col):
    X_test = data_test.X_test
    y_test = data_test.y_test
    orig_df = data_test.orig_df
    cat_map = data_test.cat_map
    return DotMap(
        X_test=X_test,
        y_test=y_test,
        orig_y=orig_df[target_col],
        orig_cost=data_test.cost_orig_df,
        orig_df=orig_df,
        dataset=data_test,
        cat_map=cat_map
    )


@dataclass
class EvalSettings:
    target_col: str
    gain_col: str
    spec: list
    target_class: int
    experiments: list
    working_datasets: DotMap

def matrix2dict(dataset, costs, name=None):

    cm = {}
    num_cat = len(costs)
    if name is None:
        values = string.ascii_lowercase[0:num_cat]
    else:
        values = []
        cols = dataset.orig_df.columns
        for c in cols:
            if name in c:
                values.append(c.replace(name,"")[1:])
        #print(values)

    for i, value1 in enumerate(values):
        cl = {}
        for j, value2 in enumerate(values):
            cl[value2] = costs[i, j]
        cm[value1] = cl
    #print(costs)
    return cm



def setup_dataset_eval(dataset, data_dir, seed=0, noise="0"):

    if dataset == "syn":
        target_col = "target"
        gain_col = "gain"
        target_class = 0
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True, noise=noise)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = []
        costs = data_test.costs
        for i in range(50):
            cm = matrix2dict(dataset, costs["cat" + str(i)])
            cat_feat  = CategoricalMatrixFeature(name="cat" + str(i), cost_matrix=cm)
            spec.append(cat_feat)

        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
            AttackConfig(name="pgd_400", algo="pgd", kwargs=dict(steps=400)),
            AttackConfig(name="pgd_20", algo="pgd", kwargs=dict(steps=20)),
        ]

    if dataset == "ieeecis":
        target_col = "isFraud"
        gain_col = "TransactionAmt"
        target_class = 0
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = [
            # Product type
            # CategoricalFeature(name="ProductCD", cost=1),
            # Card brand and type
            #CategoricalFeature(name="card_type", cost=20),
            # Receiver email domain
            # CategoricalFeature(name="R_emaildomain", cost=0.2),
            # Payee email domain
            #CategoricalFeature(name="P_emaildomain", cost=0.2),
            CategoricalMatrixFeature(name="P_emaildomain",
                cost_matrix=matrix2dict(data_test, data_test.costs["P_emaildomain"], name="P_emaildomain")),
            CategoricalMatrixFeature(name="card_type",
                cost_matrix=matrix2dict(data_test, data_test.costs["card_type"], name="card_type")),
            # Payment device
            CategoricalMatrixFeature(name="DeviceType",
                cost_matrix=matrix2dict(data_test, data_test.costs["DeviceType"], name="DeviceType")),
        ]
        experiments = (
            generic_experiments
            + common_ps_family_experiments
            + common_a_star_family_experiments
        ) + [
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_delta_beam100",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=100,
            ),
            AttackConfig(name="pgd_1k", algo="pgd", kwargs=dict(steps=1000)),
            AttackConfig(name="pgd_10", algo="pgd", kwargs=dict(steps=10)),
            AttackConfig(name="pgd_1", algo="pgd", kwargs=dict(steps=1)),
            AttackConfig(name="pgd_100", algo="pgd", kwargs=dict(steps=100)),
            AttackConfig(name="Ballet", algo="Ballet", kwargs=dict(steps=1000)),
        ]

    if dataset == "ieeecis-gain":
        target_col = "isFraud"
        gain_col = "TransactionAmt"
        target_class = 0
        data_test = get_dataset("ieeecis", data_dir, mode="test", seed=seed, cat_map=True)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = [
            # Product type
            # CategoricalFeature(name="ProductCD", cost=1),
            # Card brand and type
            NumFeature(name="TransactionAmt", inc_cost=0, integer=True).infer_range(
                working_datasets.orig_df, bins=30,
            ),
            CategoricalFeature(name="card_type", cost=20),
            # Receiver email domain
            # CategoricalFeature(name="R_emaildomain", cost=0.2),
            # Payee email domain
            CategoricalFeature(name="P_emaildomain", cost=0.2),
            # Payment device
            CategoricalFeature(name="DeviceType", cost=0.1),
        ]
        experiments = (
            generic_experiments
            + common_ps_family_experiments
            + common_a_star_family_experiments
        ) + [
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_delta_beam100",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=100,
            ),
            AttackConfig(name="pgd_1k", algo="pgd", kwargs=dict(steps=1000)),
        ]

    if dataset == "twitter_bot":
        target_col = "is_bot"
        gain_col = "value"
        target_class = 0
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        working_datasets = _get_working_datasets(data_test, target_col)
        spec = [
            NumFeature(name="user_tweeted", inc_cost=2, integer=True).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="user_replied", inc_cost=2, integer=True).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="likes_per_tweet", inc_cost=0.025).infer_range(
                working_datasets.orig_df, bins=10,
            ),
            NumFeature(name="retweets_per_tweet", inc_cost=0.025).infer_range(
                working_datasets.orig_df, bins=10
            ),
        ]
        ps_family_experiments = common_ps_family_experiments + [
            AttackConfig(
                name="astar_opt",
                scoring="a_star",
                heuristic="linear",
                kwargs=dict(cost_coef=0.5, cost_min_step_value=0.01),
            ),
        ]
        a_star_family_experiments = common_a_star_family_experiments + [
            AttackConfig(
                name="ps_opt",
                scoring="ps",
                heuristic="linear",
                kwargs=dict(cost_coef=0.5, cost_min_step_value=0.025),
            )
        ]
        experiments = (
            generic_experiments + ps_family_experiments + a_star_family_experiments
        ) + [
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_delta_beam100",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=100,
            ),
            AttackConfig(name="pgd_1k", algo="pgd", kwargs=dict(steps=1000)),
        ]

    if dataset == "home_credit":
        target_col = "TARGET"
        gain_col = "AMT_CREDIT"
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        target_class = 1
        working_datasets = _get_working_datasets(data_test, target_col)
        df = working_datasets.orig_df
        spec = [
            CategoricalFeature("NAME_CONTRACT_TYPE", 0.1),
            CategoricalFeature("NAME_TYPE_SUITE", 0.1),
            BinaryFeature("FLAG_EMAIL", cost=0.1),
            CategoricalFeature("WEEKDAY_APPR_PROCESS_START", 0.1),
            CategoricalFeature("HOUR_APPR_PROCESS_START", 0.1),
            BinaryFeature("FLAG_MOBIL", cost=10),
            BinaryFeature("FLAG_EMP_PHONE", cost=10),
            BinaryFeature("FLAG_WORK_PHONE", cost=10),
            BinaryFeature("FLAG_CONT_MOBILE", cost=10),
            BinaryFeature("FLAG_OWN_CAR", cost=100),
            BinaryFeature("FLAG_OWN_REALTY", cost=100),

            BinaryFeature("REG_REGION_NOT_LIVE_REGION", cost=100),
            BinaryFeature("REG_REGION_NOT_WORK_REGION", cost=100),
            BinaryFeature("LIVE_REGION_NOT_WORK_REGION", cost=100),
            BinaryFeature("REG_CITY_NOT_LIVE_CITY", cost=100),
            BinaryFeature("REG_CITY_NOT_WORK_CITY", cost=100),
            BinaryFeature("LIVE_CITY_NOT_WORK_CITY", cost=100),

            CategoricalFeature("NAME_INCOME_TYPE", 100),
            CategoricalFeature("cluster_days_employed", 100),
            CategoricalFeature("NAME_HOUSING_TYPE", 100),
            CategoricalFeature("OCCUPATION_TYPE", 100),
            CategoricalFeature("ORGANIZATION_TYPE", 100),
            CategoricalFeature("NAME_FAMILY_STATUS", 1000),
            CategoricalFeature("NAME_EDUCATION_TYPE", 1000),
            BinaryFeature("has_children", cost=1000),
            NumFeature("AMT_INCOME_TOTAL", inc_cost=1).infer_range(df, bins=30),
            NumFeature("EXT_SOURCE_1", inc_cost=10000).infer_range(df, bins=10),
            NumFeature("EXT_SOURCE_2", inc_cost=10000).infer_range(df, bins=10),
            NumFeature("EXT_SOURCE_3", inc_cost=10000).infer_range(df, bins=10),
        ]
        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
            AttackConfig(name="pgd_400", algo="pgd", kwargs=dict(steps=400)),
            AttackConfig(name="pgd_20", algo="pgd", kwargs=dict(steps=20)),
        ]

    if dataset == "lending_club":
        target_col = "loan_status"
        gain_col = "loan_amnt"
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        target_class = 1
        working_datasets = _get_working_datasets(data_test, target_col)
        #df = working_datasets.orig_df
        spec = [
            CategoricalMatrixFeature(name="annual_inc",
                cost_matrix=matrix2dict(data_test, data_test.costs["annual_inc"], name="annual_inc")),
            CategoricalMatrixFeature(name="application_type",
                cost_matrix=matrix2dict(data_test, data_test.costs["application_type"], name="application_type")),
            CategoricalMatrixFeature(name="verification_status",
                cost_matrix=matrix2dict(data_test, data_test.costs["verification_status"], name="verification_status")),
            CategoricalMatrixFeature(name="pub_rec_bankruptcies",
                cost_matrix=matrix2dict(data_test, data_test.costs["pub_rec_bankruptcies"], name="pub_rec_bankruptcies")),
            #CategoricalMatrixFeature(name="zip_code",
            #    cost_matrix=matrix2dict(data_test, data_test.costs["zip_code"], name="zip_code")),
        ]

        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(name="pgd_400", algo="pgd", kwargs=dict(steps=400)),
            AttackConfig(name="pgd_20", algo="pgd", kwargs=dict(steps=20)),
        ]
    
    if dataset == "bank_fraud":
        target_col = "fraud_bool"
        gain_col = "proposed_credit_limit"
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        target_class = 0
        working_datasets = _get_working_datasets(data_test, target_col)
        #df = working_datasets.orig_df
        spec = [
            CategoricalMatrixFeature(name="income",
                cost_matrix=matrix2dict(data_test, data_test.costs["income"], name="income")),
            CategoricalMatrixFeature(name="zip_count_4w",
                cost_matrix=matrix2dict(data_test, data_test.costs["zip_count_4w"], name="zip_count_4w")),
            CategoricalMatrixFeature(name="foreign_request",
                cost_matrix=matrix2dict(data_test, data_test.costs["foreign_request"], name="foreign_request")),
            CategoricalMatrixFeature(name="keep_alive_session",
                cost_matrix=matrix2dict(data_test, data_test.costs["keep_alive_session"], name="keep_alive_session")),
            CategoricalMatrixFeature(name="source",
                cost_matrix=matrix2dict(data_test, data_test.costs["source"], name="source")),
            CategoricalMatrixFeature(name="device_os",
                cost_matrix=matrix2dict(data_test, data_test.costs["device_os"], name="device_os")),
            CategoricalMatrixFeature(name="email_is_free",
                cost_matrix=matrix2dict(data_test, data_test.costs["email_is_free"], name="email_is_free")),
            CategoricalMatrixFeature(name="phone_home_valid",
                cost_matrix=matrix2dict(data_test, data_test.costs["phone_home_valid"], name="phone_home_valid")),
            CategoricalMatrixFeature(name="phone_mobile_valid",
                cost_matrix=matrix2dict(data_test, data_test.costs["phone_mobile_valid"], name="phone_mobile_valid")),
            CategoricalMatrixFeature(name="device_distinct_emails_8w",
                cost_matrix=matrix2dict(data_test, data_test.costs["device_distinct_emails_8w"], name="device_distinct_emails_8w")),
                
        ]

        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
            AttackConfig(name="pgd_400", algo="pgd", kwargs=dict(steps=400)),
            AttackConfig(name="pgd_20", algo="pgd", kwargs=dict(steps=20)),
        ]

    if dataset == "credit_sim":
        target_col = "Fraud"
        gain_col = "Amount"
        data_test = get_dataset(dataset, data_dir, mode="test", seed=seed, cat_map=True)
        target_class = 0
        working_datasets = _get_working_datasets(data_test, target_col)
        #df = working_datasets.orig_df
        spec = [
            CategoricalMatrixFeature(name="Merchant City",
                cost_matrix=matrix2dict(data_test, data_test.costs["Merchant City"], name="Merchant City")),
            CategoricalMatrixFeature(name="card_brand",
                cost_matrix=matrix2dict(data_test, data_test.costs["card_brand"], name="card_brand")),
            CategoricalMatrixFeature(name="card_type",
                cost_matrix=matrix2dict(data_test, data_test.costs["card_type"], name="card_type")),

        ]

        experiments = [
            AttackConfig(
                name="random", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy", scoring="greedy", heuristic="random", beam_size=1
            ),
            AttackConfig(
                name="greedy_delta",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=1,
            ),
            AttackConfig(
                name="greedy_delta_beam10",
                scoring="delta_hc_ratio",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="astar_subopt_beam10",
                scoring="a_star",
                heuristic="confidence",
                beam_size=10,
            ),
            AttackConfig(
                name="greedy_beam10", scoring="hc_ratio", heuristic="confidence", beam_size=10,
            ),
            AttackConfig(
                name="ps_opt",
                scoring="ps",
                heuristic="linear",
                kwargs=dict(cost_coef=0.5, cost_min_step_value=0.025),
            ),
            AttackConfig(name="pgd_400", algo="pgd", kwargs=dict(steps=400)),
            AttackConfig(name="pgd_20", algo="pgd", kwargs=dict(steps=20)),
        ]

    return EvalSettings(
        target_col=target_col,
        gain_col=gain_col,
        spec=spec,
        target_class=target_class,
        working_datasets=_get_working_datasets(data_test, target_col),
        experiments=experiments,
    )
