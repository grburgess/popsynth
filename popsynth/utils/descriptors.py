class DistributionParameter(object):
    def __init__(self, default=None, vmin=None, vmax=None):

        self.name = None
        self._vmin = vmin
        self._vmax = vmax
        self._default = default
        self._is_set = False

    def __get__(self, obj, type=None) -> object:
        if not self._is_set:
            obj._parameter_storage[self.name] = self._default

        return obj._parameter_storage[self.name]

    def __set__(self, obj, value) -> None:
        self._is_set = True

        if self._vmin is not None:
            assert value >= self._vmin, f"trying to set {self.x} to a value below {self._vmin} is not allowed"

        if self._vmax is not None:
            assert value <= self._vmax, f"trying to set {self.x} to a value above {self._vmax} is not allowed"

        obj._parameter_storage[self.name] = value


class DistributionMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):

        out = {}
        out["_parameter_storage"] = {}

        return out

    def __new__(mcls, name, bases, attrs):

        for k, v in attrs.items():
            if isinstance(v, DistributionParameter):
                v.name = k

        return super().__new__(mcls, name, bases, attrs)
