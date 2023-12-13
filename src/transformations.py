import bisect

import attr
import numpy as np
import pandas as pd

from src.utils.counter import ExpansionCounter


@attr.s(auto_attribs=True)
class CategoricalFeature:
    name: str
    cost: int = 1
    separator: str = "_"
    _col_names: list = None
    _values: list = None

    def get_column_name(self, value: str):
        return f"{self.name}{self.separator}{value}"

    def get_value_from_col(self, col_name: str):
        n = len(self.name.split(self.separator))  # Make it better pls
        return self.separator.join(col_name.split(self.separator)[n:])

    def get_col_names(self):
        assert self._values is not None
        if self._col_names is None:
            self._col_names = [self.get_column_name(value) for value in self._values]
        return self._col_names

    def infer_range(self, x: pd.Series):
        if self._values is None:
            self._values = [
                self.get_value_from_col(col)
                for col in x.index
                if col.startswith(self.name + self.separator)
            ]
        return self._values

    def get_example_value(self, x: pd.Series):
        # x[self.get_col_names()] = x[self.get_col_names()].apply(pd.to_numeric, errors='coerce')
        # print(x[self.get_col_names()])
        col_names = x[self.get_col_names()].astype(np.float64)
        orig_col = col_names.idxmax()
        # print(self.get_value_from_col(orig_col))
        return self.get_value_from_col(orig_col)

    def get_cost(self, x, x_prim):
        orig_value = self.get_example_value(x)
        new_value = self.get_example_value(x_prim)
        return self.cost if orig_value != new_value else 0

    def get_example_transformations(self, x: pd.Series, all_transformations=True):
        assert all_transformations

        if self._values is None:
            self._values = self.infer_range(x)
        cost = self.cost

        counter = ExpansionCounter.get_default()
        counter.increment()

        orig_value = self.get_example_value(x)
        orig_col = self.get_column_name(orig_value)
        for value in self._values:
            if value != orig_value:
                copy = x.copy()
                copy[orig_col] = 0
                copy[self.get_column_name(value)] = 1
                yield copy, cost

@attr.s(auto_attribs=True)
class CategoricalMatrixFeature:
    name: str
    cost_matrix: dict = None
    separator: str = "_"
    _col_names: list = None
    _values: list = None

    def get_column_name(self, value: str):
        return f"{self.name}{self.separator}{value}"

    def get_value_from_col(self, col_name: str):
        n = len(self.name.split(self.separator))  # Make it better pls
        return self.separator.join(col_name.split(self.separator)[n:])

    def get_col_names(self):
        assert self._values is not None
        if self._col_names is None:
            self._col_names = [self.get_column_name(value) for value in self._values]
        return self._col_names

    def infer_range(self, x: pd.Series):
        if self._values is None:
            self._values = [
                self.get_value_from_col(col)
                for col in x.index
                if col.startswith(self.name + self.separator)
            ]
            #print(self._values)
        return self._values

    def get_example_value(self, x: pd.Series):
        # x[self.get_col_names()] = x[self.get_col_names()].apply(pd.to_numeric, errors='coerce')
        #print(self.get_col_names())
        col_names = x[self.get_col_names()].astype(np.float64)
        #print(x[self.get_col_names()].astype(np.float64))
        orig_col = col_names.idxmax()
        # print(self.get_value_from_col(orig_col))
        return self.get_value_from_col(orig_col)

    def get_cost(self, x, x_prim):
        orig_value = self.get_example_value(x)
        new_value = self.get_example_value(x_prim)
        if orig_value == new_value:
            return 0
        cost = self.cost_matrix[orig_value][new_value]
        return cost

    def get_example_transformations(self, x: pd.Series, all_transformations=True):
        assert all_transformations

        if self._values is None:
            self._values = self.infer_range(x)

        counter = ExpansionCounter.get_default()
        counter.increment()

        orig_value = self.get_example_value(x)
        orig_col = self.get_column_name(orig_value)
        for value in self._values:
            if value != orig_value:
                copy = x.copy()
                copy[orig_col] = 0
                copy[self.get_column_name(value)] = 1
                #print(self.cost_matrix[orig_value][value], orig_value, value)
                yield copy, self.cost_matrix[orig_value][value]


@attr.s(auto_attribs=True)
class BinaryFeature:
    name: str
    values: list = None
    cost: int = 1

    def get_column_name(self):
        return f"{self.name}"

    def get_example_value(self, x: pd.Series):
        #return x[self.get_column_name()].astype(bool)
        return bool(x[self.get_column_name()])

    def get_cost(self, x, x_prim):
        orig_value = self.get_example_value(x)
        new_value = self.get_example_value(x_prim)
        return self.cost if orig_value != new_value else 0

    def get_example_transformations(self, x: pd.Series, all_transformations=True):
        assert all_transformations

        if self.values is None:
            values = [0, 1]
        else:
            values = self.values

        counter = ExpansionCounter.get_default()
        counter.increment()

        orig_value = self.get_example_value(x)
        for value in values:
            if value != orig_value:
                copy = x.copy()
                copy[self.get_column_name()] = value
                yield copy, self.cost


EPS = 10e-5


@attr.s(auto_attribs=True)
class NumFeature:
    name: str
    inc_cost: float = None
    dec_cost: float = None
    integer: bool = False

    def infer_range(self, data, bins):
        self._bin_edges = np.histogram_bin_edges(data[self.name], bins=bins)
        return self

    def get_max_val(self):
        return self._bin_edges[-1]

    def get_min_val(self):
        return self._bin_edges[0]

    def get_example_value(self, x):
        orig_value = x.loc[self.name]
        return orig_value

    def get_cost(self, x, x_prim):
        orig_value = self.get_example_value(x)
        current_bin = bisect.bisect(self._bin_edges, orig_value)
        new_value = self.get_example_value(x_prim)
        new_bin = bisect.bisect(self._bin_edges, new_value)
        if new_bin > current_bin:
            cost = self.inc_cost
        elif new_bin < current_bin:
            cost = self.dec_cost
        else:
            cost = 0
        return self._compute_cost(orig_value, new_value, cost)

    def _compute_cost(self, orig_value, new_value, cost):
        return np.abs(orig_value - new_value) * cost

    def get_example_transformations(self, x: pd.Series, all_transformations=False):
        # print(self.name)
        counter = ExpansionCounter.get_default()
        counter.increment()

        orig_value = self.get_example_value(x)
        orig_bin = bisect.bisect(self._bin_edges, orig_value)
        current_bin = orig_bin

        # Move down one bin.
        while current_bin > 1 and self.dec_cost is not None:
            copy = x.copy()
            new_value = self._bin_edges[current_bin - 1] - EPS
            if self.integer:
                new_value = int(new_value)
            copy.loc[self.name] = new_value
            yield copy, self._compute_cost(orig_value, new_value, self.dec_cost)

            if all_transformations:
                current_bin -= 1
            else:
                break

        current_bin = orig_bin

        # Move up one bin.
        while current_bin < len(self._bin_edges) - 1 and self.inc_cost is not None:
            copy = x.copy()
            new_value = self._bin_edges[current_bin + 1] - EPS
            if self.integer:
                new_value = int(new_value)
            copy.loc[self.name] = new_value
            yield copy, self._compute_cost(orig_value, new_value, self.inc_cost)

            if all_transformations:
                current_bin += 1
            else:
                break


class TransformationGenerator:
    def __init__(self, feature_specs: list):
        self.feature_specs = feature_specs

    def __call__(self, x: pd.Series):
        for spec in self.feature_specs:
            for transformation in spec.get_example_transformations(x):
                yield transformation


def recompute_cost(spec, x, adv_x):
    cost = 0
    for feature_spec in spec:
        cost += feature_spec.get_cost(x, adv_x)
    return cost
