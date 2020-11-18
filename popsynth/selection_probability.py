import abc

import numpy as np
from nptyping import NDArray


class SelectionProbabilty(object, metaclass=abc.ABCMeta):
    def __init__(self, verbose: bool = False):
        """"""
        self._verbose = verbose  # type: bool

    @abc.abstractclassmethod
    def draw(self, size: int) -> None:

        pass

    @property
    def selection(self) -> NDArray[np.bool_]:
        return self._selection
