import numpy as np
import h5py
import os


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(
                item,
            (np.ndarray, np.int64, np.float64, str, bytes, float, int)):
            h5file[path + "/" + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file,
                                                    path + "/" + key + "/",
                                                    item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + "/" + key + "/")
    return ans


# these two functions add and clean
# a networkx graph dict so that it will be stored in
# an hdf5 file


def fill_graph_dict(graph_dict):
    new_dict = {}

    for k, v in graph_dict.items():
        if len(v) == 0:
            # this is an empty dict
            # so fill it
            new_dict[k] = 1
        else:
            new_dict[k] = fill_graph_dict(v)
    return new_dict


def clean_graph_dict(graph_dict):
    new_dict = {}

    for k, v in graph_dict.items():
        if not isinstance(v, dict):
            # this is an empty dict
            # so fill it
            new_dict[k] = {}

        else:

            new_dict[k] = clean_graph_dict(v)
    return new_dict
