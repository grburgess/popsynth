import matplotlib as mpl
import matplotlib.pyplot as plt

def array_to_cmap(values, cmap, use_log=False, vmin=None, vmax=None):
    """
    Generates a color map and color list that is normalized
    to the values in an array. Allows for adding a 3rd dimension
    onto a plot
    
    :param values: a list a values to map into a cmap
    :param cmap: the mpl colormap to use
    :param use_log: if the mapping should be done in log space
    """

    if (vmin is None) and (vmax is None):

        vmin = min(values)
        vmax = max(values)

    if use_log:

        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    else:

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    rgb_colors = map(cmap.to_rgba, values)

    return cmap, rgb_colors
