import h5py
import numpy as np
from typing import Dict, Any


def recursively_save_dict_contents_to_group(h5file, path, dic: Dict[Any, Any]):
    """
    Save dict to HDF5.

    :param h5file: HDF5 file
    :param path: Path inside file
    :param dic: Dictionary to save
    :type dic: Dict[Any, Any]
    """
    for key, item in dic.items():

        if isinstance(item, list):

            is_ok = True

            for i in item:

                if isinstance(i, str):

                    pass

                else:

                    is_ok = False

            if is_ok:

                # now create a dict

                tmp = {}

                for i in item:

                    tmp[i] = {}

                item = tmp

            else:

                raise ValueError("Cannot save %s type" % type(item))

        if isinstance(
                item,
            (np.ndarray, np.int64, np.float64, str, bytes, float, int)):

            h5file[path + "/" + key] = item

        elif isinstance(item, dict):

            if len(item) == 0:

                h5file[path + "/" + key] = "FILL_VALUE"

            recursively_save_dict_contents_to_group(h5file,
                                                    path + "/" + key + "/",
                                                    item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Load files from hdf5.

    :param h5file: HDF5 file
    :param path: Path in file
    :returns: A dictionary
    :rtype: Dict[Any, Any]
    """
    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]

            try:

                ans[key] = ans[key].decode()

                if ans[key] == "FILL_VALUE":

                    ans[key] = {}

            except:

                pass

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + "/" + key + "/")
    return ans


def fill_graph_dict(graph_dict, fill_value=1):
    """
    Fill a networkx graph dict so that it
    can be stored in an HDF5 file.
    """
    new_dict = {}

    for k, v in graph_dict.items():
        if len(v) == 0:
            # this is an empty dict
            # so fill it
            new_dict[k] = fill_value
        else:
            new_dict[k] = fill_graph_dict(v)
    return new_dict


def clean_graph_dict(graph_dict):
    """
    Clean networkx graph dict so that it
    can be stored in an HDF5 file.
    """
    new_dict = {}

    for k, v in graph_dict.items():
        if not isinstance(v, dict):
            # this is an empty dict
            # so fill it
            new_dict[k] = {}

        else:

            new_dict[k] = clean_graph_dict(v)
    return new_dict
