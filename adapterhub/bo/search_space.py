import copy

import ConfigSpace as CS
from ConfigSpace.util import get_random_neighbor, get_one_exchange_neighbourhood
import ConfigSpace.hyperparameters as CSH
import numpy as np
import networkx as nx
from typing import Optional, Union, Tuple, List, Dict, Any
from sklearn.preprocessing import OneHotEncoder
import torch
from adapterhub.bo.extractor import get_extractor
from adapterhub.bo.utils import parse_graphs, collate_graphs_adj
from collections import OrderedDict
from botorch.utils.transforms import normalize


class AdapterSearchSpace:
    def __init__(self,
                 is_large: bool = False,
                 seed: Optional[int] = None,
                 ):
        self.seed = seed

        self.cs = None
        self.one_hot_encoders = {}
        self.conditional_encoder = None
        self.is_large = is_large
        self.create_search_space()
        self.dim = len(self.cs)

    def create_search_space(self):
        self.cs = CS.ConfigurationSpace(seed=self.seed)
        # architectural parameters
        params = [
            # encode Boolean as integer with 0, 1 only
            CSH.UniformIntegerHyperparameter(
                "log2_reduction_factor", 0, 10, default_value=1),

            CSH.UniformIntegerHyperparameter(
                "log2_reduction_prefix", 0, 10, default_value=1),

            CSH.UniformIntegerHyperparameter(
                "log2_reduction_serial", 0, 10, default_value=1),

            CSH.UniformIntegerHyperparameter(
                "leave_out_0", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_1", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_2", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_3", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_4", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_5", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_6", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_7", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_8", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_9", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_10", 0, 1, default_value=0),
            CSH.UniformIntegerHyperparameter(
                "leave_out_11", 0, 1, default_value=0),
        ]
        if self.is_large:
            params = [
                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_factor", 0, 10, default_value=1),

                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_prefix", 0, 10, default_value=1),

                CSH.UniformIntegerHyperparameter(
                    "log2_reduction_serial", 0, 10, default_value=1),

                CSH.UniformIntegerHyperparameter(
                    "leave_out_0", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_1", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_2", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_3", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_4", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_5", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_6", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_7", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_8", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_9", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_10", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_11", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_12", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_13", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_14", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_15", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_16", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_17", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_18", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_19", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_20", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_21", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_22", 0, 1, default_value=0),
                CSH.UniformIntegerHyperparameter(
                    "leave_out_23", 0, 1, default_value=0),
            ]
        self.cs.add_hyperparameters(params)

        # specify the conditional and categorical parameters
        self._conditional_params = [
            # "is_parallel", "ln_after", "original_ln_before", "residual_before_ln", "adapter_residual_before_ln"
        ]
        self._categorical_params = [
            # "non_linearity"
        ]

        self._conditional_dim_indices_orig = [
            self.cs.get_idx_by_hyperparameter_name(param) for param in self._conditional_params
        ]
        self._categorical_dim_indices_orig = [
            self.cs.get_idx_by_hyperparameter_name(param) for param in self._categorical_params
        ]
        # define an one-hot encoder for each categorical parameter
        for param in self._categorical_params:
            enc = OneHotEncoder()
            one_hot_encoding = enc.fit_transform(
                [[s] for s in self.cs[param].choices]).toarray()
            self.one_hot_encoders[param] = (
                list(self.cs[param].choices), one_hot_encoding)

        # define a single graph encoder to handle the conditional parameters
        # self.conditional_encoder = get_extractor("wl", options={"h": 1})
        # self._enumerate_and_cache_conditional_encodings(
        #     self.conditional_encoder)

    def _enumerate_and_cache_conditional_encodings(self, extractor):
        """
        The encoder of the conditional variables (a Weifeiler-Lehman feature extractor) behaves differently in train
            and eval modes. To properly function, we need to "fit" it on the features. Since we only have 5 binary
            conditional variables we are going to simply enumerate all of them (2^5 = 32 possibilities)

            Note that this process doesn't require output and thus is unsupervised.
            For an alternative problem where it is impossible to enumerate all the possible options, do not use
            this method.
        Returns:

        """
        len_params = len(self._conditional_params)
        total_permutes = 2 ** len_params
        # convert all of them into binary numbers
        all_permute_binary = np.stack([
            np.binary_repr(i, width=len_params)
            for i in np.arange(total_permutes)])

        Xs, adj_ts = [], []
        for i, config in enumerate(all_permute_binary):
            config_dict = {self._conditional_params[i]: int(
                config[i]) for i in range(len(config))}
            _, X, adj_t = self._get_graph_repr_conditional_params(config_dict)
            Xs.append(X)
            adj_ts.append(adj_t)
        # this two functions pad the graph adjs and features so that they are of the same size
        Xs, adj_ts = parse_graphs(adj_mats=adj_ts, Xs=Xs)
        Xs_collated, adj_collated, batch = collate_graphs_adj(Xs, adj_ts)
        extractor.train()
        output = extractor(Xs_collated, adj_collated, batch)
        # cache the feature - output pairs for later use.

        # conditional_encoded is a 5-bit string; conditional_decoded is the WL encoding
        self._all_conditional_strs = all_permute_binary
        # normalize output by its empirical bounds so that features lie in [0, 1]
        empirical_bounds = torch.stack([output.min(0)[0], output.max(0)[0]])
        self._all_conditional_wl_encodings = normalize(output, empirical_bounds).numpy()


    def _get_graph_repr_conditional_params(self, config: Union[CS.Configuration, Dict[str, Any]], ) \
            -> Tuple[nx.DiGraph, torch.Tensor, torch.Tensor]:
        """
        Hyperparameters `original_ln_before`, `is_parallel`, `ln_after`, `adapter_residual_before` and
            `residual_before_ln` exhibit complicated conditional relations. This def encodes this part of
            a configuration into a graph, and return the adjacency matrix and the feature matrix.

        """
        # create a graph of the part of the search space where conditional relations exist
        g = nx.DiGraph()
        # Add the "root" nodes first
        g.add_node(0, op_name="is_parallel", value=config["is_parallel"])
        g.add_node(1, op_name="ln_after", value=config["ln_after"])
        if not config["is_parallel"]:
            g.add_node(2, op_name="original_ln_before",
                       value=config["original_ln_before"])
            g.add_edge(0, 2)
            if config["original_ln_before"]:
                g.add_node(4, op_name="residual_before_ln",
                           value=config["residual_before_ln"])
                g.add_edge(2, 4)
            if config["ln_after"]:
                # adapter_residual_before is only applicable when ln_after is True and is_parallel is False
                g.add_node(3, op_name="adapter_residual_before",
                           value=config["adapter_residual_before_ln"])
                g.add_edge(1, 3)
                g.add_edge(0, 3)
        adj_mat = nx.to_numpy_array(g)
        feature_mat = np.array([data["value"]
                               for i, data in g.nodes(data=True)])

        # Now encode the graph into a latent vector representation
        adj_t = torch.from_numpy(adj_mat)
        X = torch.from_numpy(feature_mat)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return g, X, adj_t

    def _encode_conditional(self, config: CS.Configuration) -> np.ndarray:
        # get the encoding of the conditional and look up its index in self._all_conditional_strs
        config_conditional_str = "".join(
            [str(int(config[s])) for s in self._conditional_params])
        # look up index
        config_conditional_str_index = np.argwhere(
            self._all_conditional_strs == config_conditional_str)
        # find the corresponding WL encoding
        wl_encoding = self._all_conditional_wl_encodings[config_conditional_str_index].squeeze(
        )
        return wl_encoding

    def _encode_categorical(self, config: CS.Configuration) -> np.ndarray:
        """
        Create a one-hot representation of the categorical variables
        Args:
            config:

        Returns:

        """
        one_hot_vals = []
        for i, idx in enumerate(self._categorical_dim_indices_orig):
            param = self.cs.get_hyperparameter_by_idx(idx)
            val = config[param]
            # print(self.one_hot_encoders[param])
            one_hot_val_index = self.one_hot_encoders[param][0].index(val)
            one_hot_vals.append(
                self.one_hot_encoders[param][1][one_hot_val_index])
        if len(one_hot_vals) > 1:
            one_hot_vals = np.concatenate(one_hot_vals, axis=-1)
        else:
            one_hot_vals = one_hot_vals[0]
        return one_hot_vals

    def encode_config(
        self,
        config: CS.Configuration,
        use_orig: bool = False,
    ) -> np.ndarray:
        orig_array_rep = config.get_array()
        if use_orig:
            return orig_array_rep
        return orig_array_rep

    def config_to_dict(self, config: CS.Configuration) -> Dict[str, Any]:
        """Convert a config representation to a dict representation"""
        ret = config.get_dictionary()
        ret["reduction_factor"] = 2 ** ret["log2_reduction_factor"]
        del ret["log2_reduction_factor"]
        ret["reduction_prefix"] = 2 ** ret["log2_reduction_prefix"]
        del ret["log2_reduction_prefix"]
        ret["reduction_serial"] = 2 ** ret["log2_reduction_serial"]
        del ret["log2_reduction_serial"]
        if "log2_reduction_rank" in ret or "reduction_rank" in ret:
            ret["reduction_rank"] = 2 ** ret["log2_reduction_rank"]
            del ret["log2_reduction_rank"]
        return ret

    def dict_to_config(self, config_dict: Dict[str, Any]) -> CS.Configuration:
        config_dict = copy.deepcopy(config_dict)
        config_dict["log2_reduction_factor"] = int(
            np.log2(config_dict["reduction_factor"]))
        del config_dict["reduction_factor"]
        config_dict["log2_reduction_prefix"] = int(
            np.log2(config_dict["reduction_prefix"]))
        del config_dict["reduction_prefix"]
        config_dict["log2_reduction_serial"] = int(
            np.log2(config_dict["reduction_serial"]))
        del config_dict["reduction_serial"]
        if "log2_reduction_rank" in config_dict or "reduction_rank" in config_dict:
            config_dict["log2_reduction_rank"] = int(
                np.log2(config_dict["reduction_rank"]))
            del config_dict["reduction_rank"]
        return CS.Configuration(self.cs, config_dict)

    def sample_configuration(self, return_dict_repr: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Sample a single configuration randomly
        Returns: a vector representing the search space, or a 2-tuple consisting of the vector and the original
            ConfigSpace.Configuration object.

            Note that the vector array may differ from the naively flattened vector represented, as we do one-hot
            encoding on the categorical dimensions and use a latent space encoding for the conditional part of
            the search space.
        """
        is_valid = False
        # here we use simple rejection sampling to remove the edge case where neither mh_adapter
        # nor output_adapter is on

        while not is_valid:
            config = self.cs.sample_configuration()
            is_valid = self.is_valid(config)
        array_rep = self.encode_config(config)
        dict_rep = self.config_to_dict(config)

        if return_dict_repr:
            return array_rep, dict_rep
        return array_rep

    def get_neighbours(self,
                       config: Union[CS.Configuration, Dict[str, Any]],
                       stochastic: bool = True,
                       n_neighbors: Optional[int] = None,
                       return_dict_repr: bool = False,
                       seed: Optional[int] = None
                       ):
        # seed = seed or self.seed
        if isinstance(config, dict):
            config = self.dict_to_config(config)
        if stochastic:
            # assert n_neighbors is not None
            n_neighbors = n_neighbors or len(self.cs)
            configs = []
            while len(configs) < n_neighbors:
                config = get_random_neighbor(config, seed=seed)
                if self.is_valid(config):
                    configs.append(config)
        else:
            configs = list(get_one_exchange_neighbourhood(
                config, seed=seed, ))
            configs = [c for c in configs if self.is_valid(c)]
        dict_repr = [self.config_to_dict(config) for config in configs]
        array_repr = [self.encode_config(config) for config in configs]
        if return_dict_repr:
            return array_repr, dict_repr
        else:
            return array_repr

    def get_config_id(self, config: Union[CS.Configuration, Dict[str, Any]]) -> str:
        """Obtain a unique configuration ID string for the current config"""
        if isinstance(config, CS.Configuration):
            config = self.config_to_dict(config)

        # leave_out_name = config["leave_out"]
        config = OrderedDict(config)
        str_id = "_".join(list([str(i) for i in config.values()]))
        # replace the leaveout string with _ to make it more readable
        # str_id = str_id.replace(str(leave_out_name), "_".join(list([str(i) for i in leave_out_name])))
        return str_id

    # @staticmethod
    def is_valid(self, config: Union[CS.Configuration, Dict[str, Any]]):
        """
        Ensure either `mh_adapter` or `output_adapter` is True (otherwise we end up with
        tuning the head only).
        """

        # return (config["mh_adapter"] or config["output_adapter"])
        max_layer = 12
        num_layer = 12
        if self.is_large:
            max_layer = 24
            num_layer = 24
        for i in range(max_layer):
            if config[f"leave_out_{i}"]:
                num_layer -= 1
        if num_layer == 0:
            return False
        else:
            return config['log2_reduction_factor'] < 10 or config['log2_reduction_prefix'] < 10 or config['log2_reduction_serial'] < 10
                # or config['log2_reduction_rank'] < 7

if __name__ == '__main__':
    cs = AdapterSearchSpace()
    a = cs.cs.sample_configuration()
    print(cs.encode_config(a))
    # print(cs._get_graph_repr_conditional_params(a))
    print(cs.get_neighbours(a))
