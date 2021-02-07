import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from asciitree import LeftAligned
from asciitree.drawing import BOX_BLANK, BOX_DOUBLE, BoxStyle


def red(x) -> str:

    return f"\x1b[31;1m{x}\x1b[0m"


def blue(x) -> str:

    return f"\x1b[34;1m{x}\x1b[0m"


def green(x) -> str:

    return f"\x1b[32;1m{x}\x1b[0m"


@dataclass
class Node(object):
    """
    Implements a simple tree/node structure where the leaves store values
    Can be converted to dict and pretty printed
    """
    name: str
    parent: Optional["Node"] = None
    value: Optional[Any] = None
    _children: dict = field(init=False)

    def __post_init__(self) -> None:
        self._children = {}

    def add_child(self, child: "Node") -> None:
        """
        add a child to this node
        """
        self._children[child.name] = child
        child._set_parent(self)

    def _set_parent(self, parent: "Node") -> None:
        self.parent = parent

    @property
    def leaves(self) -> List[str]:
        """
        The children names that are leaves i.e., are the end
        """
        return [child.name for child in self._get_children() if child.is_leaf]

    @property
    def branches(self) -> List[str]:
        """
        The children names that are leaves i.e., are not the end
        """
        return [child.name for child in self._get_children() if not child.is_leaf]

    def __getitem__(self, key):

        if key in self._children:
            if self._children[key].is_leaf:
                return self._children[key].value

            else:
                return self._children[key]

        else:
            raise RuntimeError("not one of my children")

    def __setitem__(self, key, value) -> None:

        if key in self.leaves:

            self._children[key].value = value

        else:

            raise RuntimeError("Woah, you are going to overwrite the structure")

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def is_branch(self) -> bool:
        return not self.is_leaf

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def _get_children(self) -> List["Node"]:
        """
        return the children of this node
        """
        return [child for _, child in self._children.items()]

    def get_child_names(self) -> List[str]:
        """
        get the names of the children of this node
        """

        return [child for child, _ in self._children.items()]

    def __dir__(self):

        # Get the names of the attributes of the class
        l = list(self.__class__.__dict__.keys())

        # Get all the children
        l.extend([child.name for child in self._get_children()])

        return l

    def __getattribute__(self, name):

        try:
            if name in self._children:

                if self._children[name].is_leaf:
                    return self._children[name].value

                else:
                    return self._children[name]
            else:
                return super().__getattribute__(name)

        except:

            return super().__getattribute__(name)

    def __setattr__(self, name, value):

        # We cannot change a node
        # but if the node has a value
        # attribute, we want to call that

        if "_children" in self.__dict__:
            if name in self._children:

                if self._children[name].is_leaf:
                    self._children[name].value = value

                else:

                    raise RuntimeError(" Woah, you are going to overwrite the structure")

            else:
                return super().__setattr__(name, value)
        else:

            return super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        return this node and its children as 
        a dictionary
        """

        if self.is_leaf:

            dummy: Dict[str, Dict[str, Any]] = {}
            dummy[red(self.value)] = {}

            return dummy

        else:

            out: Dict[str, Dict[str, Any]] = {}

            for k, v in self._children.items():

                if v.is_leaf:

                    text = blue(k)

                else:

                    text = green(k)

                out[text] = v.to_dict()

            return out

    def __repr__(self):

        if self.is_leaf:
            return f"{self.value}"

        else:
            tr = LeftAligned(draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))

            out_final = {}
            out_final[self.name] = self.to_dict()
            return tr(out_final)
