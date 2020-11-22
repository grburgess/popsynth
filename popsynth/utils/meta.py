from typing import Union, Dict


class Parameter(object):
    def __init__(
        self,
        default: Union[None, float] = None,
        vmin: Union[None, float] = None,
        vmax: Union[None, float] = None,
    ):

        self.name = None  # type: Union[None, str]
        self._vmin = vmin  # type: Union[None, float]
        self._vmax = vmax  # type: Union[None, float]
        self._default = default  # type: Union[None, float]

    @property
    def default(self) -> Union[None, float]:
        return self._default

    def __get__(self, obj, type=None) -> object:

        try:

            return obj._parameter_storage[self.name]

        except:
            obj._parameter_storage[self.name] = self._default

        return obj._parameter_storage[self.name]

        return obj._parameter_storage[self.name]

    def __set__(self, obj, value) -> None:
        self._is_set = True

        if self._vmin is not None:
            assert (
                value >= self._vmin
            ), f"trying to set {self.x} to a value below {self._vmin} is not allowed"

        if self._vmax is not None:
            assert (
                value <= self._vmax
            ), f"trying to set {self.name} to a value above {self._vmax} is not allowed"

        obj._parameter_storage[self.name] = value


class ParameterMeta(type):
    def __new__(mcls, name, bases, attrs, **kwargs):

        cls = super().__new__(mcls, name, bases, attrs, **kwargs)

        # Compute set of abstract method names
        abstracts = {
            name
            for name, value in attrs.items()
            if getattr(value, "__isabstractmethod__", False)
        }
        for base in bases:
            for name in getattr(base, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)
        cls.__abstractmethods__ = frozenset(abstracts)

        ### parameters

        for k, v in attrs.items():

            if isinstance(v, Parameter):
                v.name = k

        return cls

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class")
        # Check cache

        # Check the subclass hook
        ok = cls.__subclasshook__(subclass)
        if ok is not NotImplemented:
            assert isinstance(ok, bool)
            if ok:
                cls._abc_cache.add(subclass)
            else:
                cls._abc_negative_cache.add(subclass)
            return ok
        # Check if it's a direct subclass
        if cls in getattr(subclass, "__mro__", ()):
            cls._abc_cache.add(subclass)
            return True
        # Check if it's a subclass of a registered class (recursive)
        for rcls in cls._abc_registry:
            if issubclass(subclass, rcls):
                cls._abc_cache.add(subclass)
                return True
        # Check if it's a subclass of a subclass (recursive)
        for scls in cls.__subclasses__():
            if issubclass(subclass, scls):
                cls._abc_cache.add(subclass)
                return True
        # No dice; update negative cache
        cls._abc_negative_cache.add(subclass)
        return False
