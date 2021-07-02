import matplotlib as mpl
import matplotlib.pyplot as plt


def array_to_cmap(values, cmap, use_log: bool = False):
    """
    Generates a color map and color list that is normalized
    to the values in an array. Allows for adding a 3rd dimension
    onto a plot.

    :param values: A list a values to map into a cmap
    :param cmap: The mpl colormap to use
    :param use_log: `True` if the mapping should be done in log space
        `False` by default.
    :type use_log: bool
    :returns: A color map and a normalized color list
    """

    if use_log:

        norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))

    else:

        norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))

    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgb_colors = [cmap.to_rgba(v) for v in values]

    return cmap, rgb_colors
