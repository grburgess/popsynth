from typing import Dict, Optional, Union

from popsynth.utils.logging import setup_logger

log = setup_logger(__name__)


class Parameter(object):

    def __init__(
        self,
        default: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        free: bool = True,
    ):
        """
        Parameter base class.

        :param default: Default parameter value
        :type default: Optional[float]
        :param vmin: Minimum parameter value
        :type vmin: Optional[float]
        :param vmax: Maximum parameter value
        :type vmax: Optional[float]
        :param free: If `True`, parameter is free
        """

        self.name = None  # type: Union[None, str]
        self._vmin = vmin  # type: Optional[float]
        self._vmax = vmax  # type: Optional[float]
        self._default = default  # type: Optional[float]
        self._free = free

    @property
    def default(self) -> Optional[float]:
        return self._default

    def __get__(self, obj, type=None) -> object:

        try:

            if obj._parameter_storage[self.name] is None:

                obj._parameter_storage[self.name] = self._default

            assert (obj._parameter_storage[self.name]
                    is not None), "parameters must have values!"

            return obj._parameter_storage[self.name]

        except (KeyError):

            obj._parameter_storage[self.name] = self._default

        assert (obj._parameter_storage[self.name]
                is not None), "parameters must have values!"

        return obj._parameter_storage[self.name]

    def __set__(self, obj, value) -> None:
        self._is_set = True

        if not self._free:

            log.error(f"{self.name} is fixed and cannot be set")
            raise RuntimeError()

        if self._vmin is not None:
            if not (value >= self._vmin):
                log.error(
                    f"trying to set {self.name} to a value below {self._vmin} is not allowed"
                )

                raise RuntimeError()

        if self._vmax is not None:
            if not (value <= self._vmax):

                log.error(
                    f"trying to set {self.name} to a value above {self._vmax} is not allowed"
                )
                raise RuntimeError()

        obj._parameter_storage[self.name] = value

        # Define property "free"

    def _set_free(self, value=True):

        self._free = value

    def _get_free(self):

        return self._free

    free = property(
        _get_free,
        _set_free,
        doc=
        "Gets or sets whether the parameter is free or not. Use booleans, like: 'p.free = True' "
        " or 'p.free = False'. ",
    )

    # Define property "fix"

    def _set_fix(self, value=True):

        self._free = not value

    def _get_fix(self):

        return not self._free

    fix = property(
        _get_fix,
        _set_fix,
        doc=
        "Gets or sets whether the parameter is fixed or not. Use booleans, like: 'p.fix = True' "
        " or 'p.fix = False'. ",
    )


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

        # parameters

        cls._parameter_storage = {}

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
