import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from popsynth.utils.rejection_sample import rejection_sample


class FunctionGen(object):

    def __init__(self, xmax):

        self.xmax = xmax

        self.find_max()

    def find_max(self):

        xspace = np.linspace(0, self.xmax, 100)

        yspace = self.func(xspace)

        self.ymax = yspace.max()


class LinearFuncGen(FunctionGen):

    def __init__(self, alpha, beta, xmax):

        def func(x):

            return alpha + beta * x

        self.func = func

        super(LinearFuncGen, self).__init__(xmax)


# class NonLinearFuncGen(FunctionGen):
#     def __init__(self, alpha, beta, xmax):

#         def func(x):

#             return alpha + beta * x

#         self.func = func

#         super(LinearFuncGen, self).__init__(xmax)


@given(
    st.floats(min_value=0, max_value=10.0),
    st.floats(min_value=1.0, max_value=5.0),
    st.floats(min_value=10.0, max_value=1000),
    st.integers(min_value=10, max_value=100),
)
@settings(deadline=None)
def test_linear_functions(alpha, beta, xmax, size):

    func_gen = LinearFuncGen(alpha, beta, xmax)

    out = rejection_sample(size, func_gen.ymax, func_gen.xmax, func_gen.func)

    print(out)

    assert len(out) == size
