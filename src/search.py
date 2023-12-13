import depq
from dotmap import DotMap
import numpy as np

def _get_optimal_path(predecessors, start_node, node, hash_fn):
    """Reconstruct the optimal path from the start to the current node."""

    path = [node]
    hashed_start = hash_fn(start_node)

    # Backtrack until starting node is found
    while hash_fn(node) != hashed_start:
        node = predecessors[hash_fn(node)]
        path.append(node)
    path.reverse()

    return path


class IterLimitExceededError(Exception):
    pass


def generalized_best_first_search(
    start_node,
    expand_fn,
    goal_fn,
    cost_bound=None,
    heuristic_fn=None,
    score_fn=None,
    hash_fn=None,
    iter_lim=None,
    beam_size=None,
    return_path=False,
    admissible=False,
    **kwargs,
):
    """
    Generalized best-first search.

    Returns the tuple (cost, target_node) if return_path is set to False.
    Otherwise, returns the target node, the costs of nodes expanded by the
    algorithm, and the optimal path from the initial node to the target node.

    :param start_node: Initial node.
    :param expand_fn: Returns an iterable of tuples (neighbour, cost).
    :param goal_fn: Returns True if the current node is the target node.
    :param cost_bound: Upper bound on costs.
    :param heuristic_fn: Returns an estimate of the cost to the target
            node. By default, is a constant 0.
    :param score_fn: Score function to combine the cost and the heuristic.
    :param hash_fn: Hash function for nodes. By default equals the
            identity function f(x) = x.
    :param iter_lim: Maximum number of iterations to try.
    :param beam_size: Beam size. Turns the search into beam search.
    :param return_path: Whether to return the optimal path from the
            initial node to the target node. By default equals False.
    :param admissible: For cost-bounded search, whether the heuristic is admissible.
    """

    # Define default heuristic and hash functions if none given
    if heuristic_fn is None:
        heuristic_fn = lambda _: 1

    if score_fn is None:
        if cost_bound is not None:
            # PS (Potential Search) scoring function.
            score_fn = lambda c, h, _: (cost_bound - c) / h

        else:
            # Greedy best-first search scoring function.
            score_fn = lambda c, h, _: -h

    if hash_fn is None:
        hash_fn = lambda x: x

    iter_count = 0

    # Define data structures to hold intermediate information.
    path_costs = {}
    predecessors = {}
    reverse_hashes = {}
    heuristic_values = {}
    open_set = depq.DEPQ(maxlen=beam_size)
    closed_set = set()

    # Add the starting node; score equal to heuristic.
    hashed_start = hash_fn(start_node)
    path_costs[hashed_start] = 0
    heuristic_values[hashed_start] = init_heuristic_value = heuristic_fn(start_node)
    score = score_fn(0, init_heuristic_value, init_heuristic_value)
    open_set.insert(hashed_start, priority=score)
    reverse_hashes[hashed_start] = start_node

    # Iterate until a goal node is found, open set is empty
    # or iteration limit has been reached.
    while len(open_set):
        if (iter_lim is not None) and (iter_count >= iter_lim):
            raise IterLimitExceededError()

        # Retrieve the node with the highest score.
        hashed_node, current_score = open_set.popfirst()
        node = reverse_hashes[hashed_node]

        # print(f"> current: {current_score=}")

        # Check if the current node is a goal node.
        if goal_fn(node):
            if return_path:
                optimal_path = _get_optimal_path(
                    predecessors, start_node, node, hash_fn
                )
                return node, path_costs, optimal_path
            else:
                return (node, path_costs[hashed_node])
        closed_set.add(hashed_node)

        # Iterate through all neighbours of the current node
        neighbours = [x for x, _ in expand_fn(node)]
        h_vals = heuristic_fn(neighbours)
        for i, n in enumerate(neighbours):
            heuristic_values[hash_fn(n)] = h_vals[i]
        for neighbour, cost in expand_fn(node):
            hashed_neighbour = hash_fn(neighbour)
            if hashed_neighbour in closed_set:
                continue

            # Compute tentative path cost from the start node to the neighbour.
            tentative_cost = path_costs[hashed_node] + cost
            # print(f"candidate {tentative_cost=}")

            # Skip if the tentative path cost is larger or equal than the
            # recorded one (if the latter exists).
            if hashed_neighbour in path_costs and (
                tentative_cost >= path_costs[hashed_neighbour]
            ):
                continue
            heuristic_value = heuristic_values[hashed_neighbour]
            parent_heuristic_value = heuristic_values.get(hashed_node) or heuristic_fn(
                node
            )

            # Additional skipping conditions for Potential Search.
            if cost_bound is not None:
                # Skip if the node is estimated to exceed the bound, for an admissible heuristic.
                if admissible and (tentative_cost + heuristic_value >= cost_bound):
                    continue
                # If the heuristic is not admissible, skip if already exceeding the cost.
                elif not admissible and (tentative_cost >= cost_bound):
                    continue

            # Record new path cost for the neighbour, the predecessor, and add
            # to open set.
            path_costs[hashed_neighbour] = tentative_cost
            score = score_fn(tentative_cost, heuristic_value, parent_heuristic_value)
            open_set.insert(hashed_neighbour, priority=score)

            reverse_hashes[hashed_neighbour] = neighbour
            if return_path:
                predecessors[hashed_neighbour] = node

            # print(f"added {tentative_cost=} {score=} {heuristic_value=}")

        iter_count += 1

    # Goal node is unreachable.
    if return_path:
        return None, path_costs, None
    else:
        return None, None


def a_star_search(
    start_node,
    expand_fn,
    goal_fn,
    heuristic_fn=None,
    hash_fn=None,
    iter_lim=None,
    beam_size=None,
    return_path=False,
):
    """
    Generalized A* search with no beam size limit.

    See :py:func:`generalized_a_star_search`.
    """

    return generalized_best_first_search(
        start_node=start_node,
        expand_fn=expand_fn,
        goal_fn=goal_fn,
        score_fn=lambda c, h: -(c + h),
        heuristic_fn=heuristic_fn,
        hash_fn=hash_fn,
        iter_lim=iter_lim,
        return_path=return_path,
        beam_size=None,
    )


def submod_greedy(
    start_node,
    spec,
    heuristic_fn,
    goal_fn,
    cost_bound=None,
    change_feature_once=False,
    criterion="score",
    all_transformations=True,
    **kwargs,
):
    """
    Greedy algorithm inspired by submodular optimization algorithms.
    """
    score_fn = lambda x: 1 - heuristic_fn(x)
    feature_pool = list(range(len(spec)))
    current_candidate = start_node
    current_cost = 0
    init_score = current_score = score_fn(start_node)
    # print(f"! new {current_score=}")
    while True:

        # Possibly stop early.
        if kwargs.get("early_stop") and goal_fn(current_candidate):
            return current_candidate, current_cost

        candidates = []
        # print("> attempt")
        # print(f"{feature_pool=}.")
        for feature_index in feature_pool:
            # print(spec[feature_index].name)
            best_candidate = find_best_feature_value(
                current_candidate,
                spec[feature_index],
                score_fn=score_fn,
                current_cost=current_cost,
                current_score=current_score,
                cost_bound=cost_bound,
                criterion=criterion,
                all_transformations=all_transformations,
            )
            if not best_candidate:
                continue

            # print(
            #     f"best: {best_candidate.cost=} {best_candidate.delta_score=} {best_candidate.score=}"
            # )

            if best_candidate.score > current_score:
                candidates.append(
                    DotMap(
                        feature_index=feature_index,
                        x_prime=best_candidate.x_prime,
                        cost=best_candidate.cost,
                        delta_score=best_candidate.delta_score,
                        score=best_candidate.score,
                    )
                )

        # print("< end attempt")
        prev_cost = current_cost
        if candidates:
            best_candidate_data = max(candidates, key=criteria[criterion])
            current_candidate = best_candidate_data.x_prime
            current_score = best_candidate_data.score
            current_cost += best_candidate_data.cost
            if change_feature_once:
                feature_pool.remove(best_candidate_data.feature_index)
            # print(f"! new {current_score=}")

        if current_cost == prev_cost:
            break

    if goal_fn(current_candidate):
        return current_candidate, current_cost

    else:
        return None, None


def find_best_feature_value(
    x,
    spec,
    score_fn,
    current_cost,
    current_score,
    cost_bound,
    criterion="delta_score",
    all_transformations=True,
):
    best_delta_score = 0
    best_score = current_score
    best_cost = 0
    best_example = x
    candidates = []

    for example, cost in spec.get_example_transformations(
        x, all_transformations=all_transformations
    ):
        if cost_bound is not None and current_cost + cost > cost_bound:
            continue

        score = score_fn(example)
        delta_score = (score - current_score) / cost
        candidate_data = DotMap(
            x_prime=example, cost=cost, delta_score=delta_score, score=score
        )
        # crit_value = criteria[criterion](candidate_data)
        # print(f"{cost=} {score=} {crit_value=}")
        candidates.append(candidate_data)

    if candidates:
        return max(candidates, key=criteria[criterion])
