"""Module for handling plotting functions

This module contains plotting classes to plot :class:`.Binning` objects.

Examples
--------

::

    plt = plotting.get_plotter(binning)
    plt.plot_values()
    plt.savefig('output.png')

"""

from __future__ import division
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
import numpy as np
from itertools import cycle

from . import binning
from . import migration

def get_plotter(obj, *args, **kwargs):
    """Return a suitable plotting class instance for the object.

    Parameters
    ----------

    obj : object
        The object for which a plotter should be returned.
    *args : optional
    **kwargs : optional
        Additional arguments are passed to the init method of the plotter.

    """

    if isinstance(obj, binning.RectilinearBinning):
        return RectilinearBinningPlotter(obj, *args, **kwargs)
    if isinstance(obj, binning.LinearBinning):
        return LinearBinningPlotter(obj, *args, **kwargs)
    if isinstance(obj, binning.CartesianProductBinning):
        return CartesianProductBinningPlotter(obj, *args, **kwargs)
    if isinstance(obj, binning.Binning):
        return BinningPlotter(obj, *args, **kwargs)
    if isinstance(obj, np.ndarray):
        return ArrayPlotter(obj, *args, **kwargs)
    raise TypeError("No known Plotter class for type %s"%(type(obj),))

class Plotter(object):
    """Plotting base class.

    Arguments
    ---------

    figax : (Figure, Axes), optional
        The figure and axis to plot in.

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.

    """

    def __init__(self, figax=None):
        self.figax = figax
        self.color = cycle('C%d'%(i,) for i in range (0,10))
        self.hatch = cycle(['//', '\\\\', 'oo', '..'])

    def __del__(self):
        """Clean up figures."""
        if self.figax is not None:
            plt.close(self.figax[0])

    def subplots(self, *args, **kwargs):
        """Return the ``(Figure, Axes)`` tuple of the binning.

        Creates one using Matplotlib's ``subplots``, if necessary.

        """

        if self.figax is None:
            self.figax = plt.subplots(*args, **kwargs)
        return self.figax

    def savefig(self, *args, **kwargs):
        """Save the figure."""
        kwargs2 = {'bbox_inches': 'tight'}
        kwargs2.update(kwargs)
        self.figax[0].savefig(*args, **kwargs2)

class ArrayPlotter(Plotter):
    """Plotting class for numpy arrays.

    Parameters
    ----------

    array : ndarray
        The ndarray to be plotted.
    bins_per_row : int, optional
        How many bins are going to be plotted per row.
    **kwargs : optional
        Addittional keyword arguments are passed to :class:`Plotter`.

    See also
    --------

    Plotter

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.
    array : ndarray
        The ndarray to be plotted.
    bins_per_row : int, optional
        How many bins are going to be plotted per row.

    """

    def __init__(self, array, bins_per_row=25, **kwargs):
        self.array = array
        self.bins_per_row = bins_per_row
        Plotter.__init__(self, **kwargs)

    def _get_array(self, array):
        if array is None:
            array = self.array
        else:
            array = np.asarray(array)
            if array.shape != self.array.shape:
                raise TypeError("Array must be of equal shape as the initial one.")
        return array

    def get_bin_edges(self, i_min, i_max):
        """Get the bin edges corresponding to bins i_min to i_max."""
        x = np.arange(i_min,i_max)
        return np.append(x-0.5, x[-1]+0.5) # Bins centred on integers

    def get_axis_label(self):
        """Return the default label for the axis."""
        return "Bin #"

    def plot_array(self, array=None, density=False, margin_function=None, **kwargs):
        """Plot an array.

        Parameters
        ----------

        array : ndarray
            The thing to plot.
        density : bool, optional
            Divide the data by the relative bin width: ``width / total_plot_range``.

        """

        # The `margin_function` parameter is only here so it can be
        # safely used with all plotting methods

        array = self._get_array(array)

        bins_per_row = self.bins_per_row
        if bins_per_row >= 1:
            n_rows = int(np.ceil(array.size / bins_per_row))
        else:
            n_rows = 1
            bins_per_row = array.size

        figax = self.subplots(nrows=n_rows, sharey=True, figsize=(6.4, max(2.4*n_rows, 4.8)), squeeze=False)

        color = next(self.color)
        hatch = next(self.hatch)

        for i, ax in enumerate(figax[1][:,0]):
            i_min = i*bins_per_row
            i_max = min((i+1)*bins_per_row, array.size)
            y = np.asfarray(array.flat[i_min:i_max])
            bins = np.asfarray(self.get_bin_edges(i_min, i_max))
            x = (bins[1:] + bins[:-1]) / 2. # Bin content in centre of bins

            # Divide by relative bin widths
            if density:
                total_width = bins[-1] - bins[0]
                rel_widths = (bins[1:] - bins[:-1]) / total_width
                y /= np.asfarray(rel_widths)

            args = {
                'histtype': 'step',
                'color': color,
                'hatch': hatch,
                }
            args.update(kwargs)
            ax.hist(x, weights=y, bins=bins, **args)

            ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set_xlabel(self.get_axis_label())

    def legend(self, **kwargs):
        """Draw a legend in the first axis."""
        args = {
            'loc': 'best',
            }
        args.update(kwargs)
        self.figax[1][0,0].legend(**args)

class BinningPlotter(ArrayPlotter):
    """Plotting class for the simplest :class:`.Binning` class.

    Parameters
    ----------

    binning : Binning
        The binning to be plotted.
    marginalize_subbinnings : bool, optional
        Plot the contents of subbinnings as a single bin.
    **kwargs : optional
        Addittional keyword arguments are passed to :class:`ArrayPlotter`.

    See also
    --------

    ArrayPlotter
    .Binning

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.
    binning : Binning
        The binning defining what will be plotted.
    marginalize_subbinnings : bool
        Whether or not subbinnings will be marginalized before plotting.

    """

    def __init__(self, binning, marginalize_subbinnings=False , **kwargs):
        self.binning = binning
        self.marginalize_subbinnings = marginalize_subbinnings
        array = self.binning.value_array
        if marginalize_subbinnings:
            array = self.binning.marginalize_subbinnings_on_ndarray(array)
        ArrayPlotter.__init__(self, array, **kwargs)

    def _get_array(self, array):
        if array is None:
            array = self.array
        else:
            array = np.asarray(array)

        # Marginalize subbinnings if necessary
        if self.marginalize_subbinnings and array.shape != self.array.shape:
            array = self.binning.marginalize_subbinnings_on_ndarray(array)

        if array.shape != self.array.shape:
            raise TypeError("Array must be of equal shape as the initial one.")

        return array

    def _get_binning(self, binning):
        if binning is None:
            binning = self.binning
        return binning

    def plot_values(self, binning=None, **kwargs):
        """Plot the values of a Binning."""

        binning = self._get_binning(binning)
        return self.plot_array(binning.value_array, **kwargs)

    def plot_entries(self, binning=None, **kwargs):
        """Plot the entries of a Binning."""

        binning = self._get_binning(binning)
        return self.plot_array(binning.entries_array, **kwargs)

    def plot_sumw2(self, binning=None, **kwargs):
        """Plot the sumw2 of a Binning."""

        binning = self._get_binning(binning)
        return self.plot_array(binning.sumw2_array, **kwargs)

class CartesianProductBinningPlotter(BinningPlotter):
    """Plotting class for :class:`.CartesianProductBinning`

    Parameters
    ----------

    binning : CartesianProductBinning
        The binning to be plottet
    x_axis_binnings : list of int, optional
        The indices of binnings to be plotted on the x-axis.
    y_axis_binnings : list of int, optional
        The indices of binnings to be plotted on the y-axis.
    **kwargs : optional
        Additional keyword arguments are passed to :class:`BinningPlotter`.

    Notes
    -----

    This plotter does always marginalize the subbinnings.

    See also
    --------

    BinningPlotter
    .CartesianProductBinning

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.
    binning : CartesianProductBinning
        The binning defining what will be plotted.
    marginalize_subbinnings : bool
        Whether or not subbinnings will be marginalized before plotting.
    x_axis_binnings : list of int
        The indices of binnings to be plotted on the x-axis.
    y_axis_binnings : list of int
        The indices of binnings to be plotted on the y-axis.

    """

    def __init__(self, binning, x_axis_binnings=None, y_axis_binnings=None, **kwargs):
        if x_axis_binnings is None:
            x_axis_binnings = list(range(int(np.ceil(len(binning.binnings) / 2.))))
        self.x_axis_binnings = x_axis_binnings
        if y_axis_binnings is None:
            y_axis_binnings = list(range(int(np.ceil(len(binning.binnings) / 2.)), len(binning.binnings)))
        self.y_axis_binnings = y_axis_binnings
        kwargs['marginalize_subbinnings'] = True
        kwargs['bins_per_row'] = -1
        BinningPlotter.__init__(self, binning, **kwargs)

    def get_bin_edges(self, i_min, i_max, j_binning):
        """Get the bin edges corresponding to bins i_min to i_max."""
        x = np.arange(i_min,i_max)
        return np.append(x-0.5, x[-1]+0.5) # Bins centred on integers

    def get_axis_label(self, j_binning):
        """Return the default label for the axis."""
        return "Binning %d Bin #"%(j_binning,)

    def plot_array(self, array=None, density=True, margin_function=np.sum, scatter=0, **kwargs):
        """Plot an array.

        Parameters
        ----------

        array : ndarray, optional
            The data to be plotted.
        density : bool, optional
            Divide the data by the relative bin width: ``width / total_plot_range``.
            Dividing by the relative bin width, rather than the bin width directly,
            ensures that the maximum values in all 1D projections are comparable.
        margin_function : function, optional
            The function used to marginalize the data.
        scatter : int, optional
            Use a pseudo scatter plot with `scatter` number of points instead
            of a 2D histogram. Allows to draw multiple sets of 2D data in the
            same plot. The number of points in each cell is proportional to
            the value being plotted. Using the `scatter` option is thus
            implicitly replicating the behaviour of the `density` option for
            the 2D plots. The `density` argument has no effect on the scatter
            plots.

        """

        array = self._get_array(array)

        shape = self.binning.bins_shape
        array = array.reshape(shape)

        n_col = len(self.x_axis_binnings) + 1 # "+1" for the 1D projections
        n_row = len(self.y_axis_binnings) + 1

        # Widths and heights according to number of bins, 10 px (= 0.1") per bin
        widths = [0.1 * self.binning.binnings[i].data_size for i in self.x_axis_binnings]
        heights = [0.1 * self.binning.binnings[i].data_size for i in self.y_axis_binnings]
        heights.reverse() # Axes are counted top to bottom, but we want binnings bottom to top

        # Total figure size
        total_width = np.sum(widths)
        total_height = np.sum(heights)
        scale = 4.0 / min(max(total_width, total_height), 4.0)

        # Room for the 1D histograms
        if total_width == 0.:
            widths.append(6 / scale)
        else:
            widths.append(1.5 / scale)
        if total_height == 0.:
            heights.insert(0, 4 / scale)
        else:
            heights.insert(0, 1.5 / scale)

        # Update total sizes
        total_width = np.sum(widths)
        total_height = np.sum(heights)

        fig_x = total_width * scale
        fig_y = total_height * scale

        # Subplot spacing is specified as multiple of average axis size
        # We want it to be relative to the 1D projections
        wspace = 0.1 * widths[-1] / (total_width / len(widths))
        hspace = 0.1 * heights[0] / (total_height / len(heights))

        figax = self.subplots(nrows=n_row, ncols=n_col, sharex='col', sharey='row',
            figsize=(fig_x, fig_y), gridspec_kw={'width_ratios': widths,
            'height_ratios': heights, 'wspace': wspace, 'hspace': hspace},
            squeeze=False)

        color = next(self.color)
        hatch = next(self.hatch)

        # 2D histograms
        for x, i in enumerate(self.x_axis_binnings):
            for y, j in enumerate(self.y_axis_binnings):
                # Get axis to plot in
                ax = figax[1][-y-1,x] # rows are counted top to bottom

                # Project array
                axis = list(range(array.ndim))
                for k in sorted((i,j), reverse=True):
                    del axis[k]
                data = margin_function(array, axis=tuple(axis))

                # Flip axes if necessary
                if i < j:
                    data = data.T

                # Bin edges
                x_edg = self.get_bin_edges(0, data.shape[1], i)
                y_edg = self.get_bin_edges(0, data.shape[0], j)

                # Plot the data
                if scatter > 0:
                    # Draw a set of random points and plot these

                    # Get bin numbers
                    csum = np.asfarray(data.cumsum())
                    csum /= np.max(csum)
                    indices = np.digitize( np.random.uniform(size=scatter), csum)

                    # Get x and y bin numbers
                    x_indices = indices % data.shape[1]
                    y_indices = indices // data.shape[1]

                    # Throw X and Y for each event
                    x = []
                    y = []
                    for ix, iy in zip(x_indices, y_indices):
                        x_min = x_edg[ix]
                        x_max = x_edg[ix+1]
                        y_min = y_edg[iy]
                        y_max = y_edg[iy+1]
                        x.append(np.random.uniform(x_min, x_max))
                        y.append(np.random.uniform(y_min, y_max))

                    # Plot the points
                    if data.sum() > 0:
                        # Only actually draw something if we have some events
                        ax.scatter(x, y, 1, color=color, marker=',')

                else:
                    # Plot a regular 2D histogram

                    # Bin centres
                    x = np.convolve(x_edg, np.ones(2)/2, mode='valid')
                    y = np.convolve(y_edg, np.ones(2)/2, mode='valid')
                    xx = np.broadcast_to(x, (len(y),len(x))).flatten()
                    yy = np.repeat(y, len(x))

                    # Plot it
                    if data.sum() == 0:
                        # Empty data messes with the normalisation
                        data.fill(0.001)
                    ax.hist2d(xx, yy, weights=data.flat, bins=(x_edg, y_edg), normed=density)

        # 1D vertical histograms
        for x, i in enumerate(self.x_axis_binnings):
            # Get axis to plot in
            ax = figax[1][0,x]

            # Project array
            axis = list(range(array.ndim))
            del axis[i]
            data = np.asfarray(margin_function(array, axis=tuple(axis)))

            # Divide by relative bin widths
            bins = np.asfarray(self.get_bin_edges(0, data.shape[0], i))
            if density:
                total_width = bins[-1] - bins[0]
                rel_widths = (bins[1:] - bins[:-1]) / total_width
                data /= rel_widths

            # Plot the data
            entries = np.convolve(bins, np.ones(2)/2, mode='valid')
            args = {
                'histtype': 'step',
                'color': color,
                'hatch': hatch,
                }
            args.update(kwargs)
            ax.hist(entries, weights=data, bins=bins, orientation='vertical', **args)

            # Only int tick label
            ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

            # Add labels at the appropriate axes
            ax = figax[1][-1,x]
            ax.set_xlabel(self.get_axis_label(i))

        # 1D horizontal histograms
        for y, i in enumerate(self.y_axis_binnings):
            # Get axis to plot in
            ax = figax[1][-y-1,-1] # Rows are counted top to bottom

            # Project array
            axis = list(range(array.ndim))
            del axis[i]
            data = np.asfarray(margin_function(array, axis=tuple(axis)))

            # Divide by relative bin widths
            bins = np.asfarray(self.get_bin_edges(0, data.shape[0], i))
            if density:
                total_width = bins[-1] - bins[0]
                rel_widths = (bins[1:] - bins[:-1]) / total_width
                data /= rel_widths

            # Plot the data
            entries = np.convolve(bins, np.ones(2)/2, mode='valid')
            args = {
                'histtype': 'step',
                'color': color,
                'hatch': hatch,
                }
            args.update(kwargs)
            ax.hist(entries, weights=data, bins=bins, orientation='horizontal', **args)

            # Only int tick label
            ax.get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))

            # Add labels at the appropriate axes
            ax = figax[1][-y-1,0] # Rows are counted top to bottom
            ax.set_ylabel(self.get_axis_label(i))

        # Hide empty axes
        figax[1][0,-1].set_axis_off()

    def legend(self, **kwargs):
        """Draw a legend in the upper right corner of the plot."""
        handles, labels = self.figax[1][0,0].get_legend_handles_labels()
        args = {
            'loc': 'center',
            'borderaxespad': 0.,
            'frameon': False,
        }
        args.update(kwargs)
        self.figax[1][0,-1].legend(handles, labels, **args)

class LinearBinningPlotter(BinningPlotter):
    """Plotting class for :class:`.LinearBinning`

    Parameters
    ----------

    binning : LinearBinning
        The binning to be plottet
    **kwargs : optional
        Additional keyword arguments are passed to :class:`BinningPlotter`.

    Notes
    -----

    This plotter does always marginalize the subbinnings.

    See also
    --------

    BinningPlotter
    .LinearBinning

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.
    binning : LinearBinning
        The binning defining what will be plotted.
    marginalize_subbinnings : bool
        Whether or not subbinnings will be marginalized before plotting.

    """

    def __init__(self, binning, **kwargs):
        kwargs['marginalize_subbinnings'] = True
        args = {
            'bins_per_row': -1,
            }
        args.update(kwargs)
        BinningPlotter.__init__(self, binning, **args)

    def plot_array(self, *args, **kwargs):
        """Plot an array.

        See :meth:`ArrayPlotter.plot_array`.

        """
        # Change default behaviour of `density`
        kwargs['density'] = kwargs.get('density', True)
        return ArrayPlotter.plot_array(self, *args, **kwargs)

    def get_bin_edges(self, i_min, i_max):
        """Get the finite bin edges."""

        bins = self.binning.bin_edges[i_min:i_max+1]

        ret = list(bins)
        if not np.isfinite(ret[0]):
            if len(ret) >= 3 and np.isfinite(ret[2]):
                ret[0] = ret[1] - (ret[2] - ret[1])
            elif np.isfinite(ret[1]):
                ret[0] = ret[1]-1
            else:
                ret[0] = -0.5
        if not np.isfinite(ret[-1]):
            if len(ret) >= 3 and np.isfinite(ret[-3]):
                ret[-1] = ret[-2] + (ret[-2] - ret[-3])
            else:
                ret[-1] = ret[-2]+1

        return np.array(ret)

    def get_axis_label(self):
        """Return variable name."""
        return self.binning.variable

class RectilinearBinningPlotter(CartesianProductBinningPlotter):
    """Plotting class for :class:`.RectilinearBinning`

    Parameters
    ----------

    binning : RectilinearBinning
        The binning to be plottet
    x_axis_binnings : list of int/str, optional
        The indices of binnings to be plotted on the x-axis.
    y_axis_binnings : list of int/str, optional
        The indices of binnings to be plotted on the y-axis.
    **kwargs : optional
        Additional keyword arguments are passed to :class:`CartesianProductBinningPlotter`.

    Notes
    -----

    This plotter does always marginalize the subbinnings.

    See also
    --------

    CartesianProductBinningPlotter
    .RectilinearBinning

    Attributes
    ----------

    figax : (Figure, [[Axes, ...], ...])
        The figure and axes that are used for the plotting.
    color : cycle of str
        Cycler that determines the color of plotting commands.
    hatch : cycle of str
        Cycler that determines the hatching style of plotting commands.
    binning : RectilinearBinning
        The binning defining what will be plotted.
    marginalize_subbinnings : bool
        Whether or not subbinnings will be marginalized before plotting.
    x_axis_binnings : list of int
        The indices of binnings to be plotted on the x-axis.
    y_axis_binnings : list of int
        The indices of binnings to be plotted on the y-axis.

    """

    def __init__(self, binning, x_axis_binnings=None, y_axis_binnings=None, **kwargs):
        if x_axis_binnings is None:
            x_axis_binnings = list(range(int(np.ceil(len(binning.binnings) / 2.))))
        else:
            x_axis_binnings = map(self.binning.get_variable_index, x_axis_binnings)
        if y_axis_binnings is None:
            y_axis_binnings = list(range(int(np.ceil(len(binning.binnings) / 2.)), len(binning.binnings)))
        else:
            y_axis_binnings = map(self.binning.get_variable_index, y_axis_binnings)
        kwargs['x_axis_binnings'] = x_axis_binnings
        kwargs['y_axis_binnings'] = y_axis_binnings
        kwargs['marginalize_subbinnings'] = True
        kwargs['bins_per_row'] = -1
        CartesianProductBinningPlotter.__init__(self, binning, **kwargs)

    def get_bin_edges(self, i_min, i_max, j_binning):
        """Get the finite bin edges."""

        bins = self.binning.binnings[j_binning].bin_edges[i_min:i_max+1]

        ret = list(bins)
        if not np.isfinite(ret[0]):
            if len(ret) >= 3 and np.isfinite(ret[2]):
                ret[0] = ret[1] - (ret[2] - ret[1])
            elif np.isfinite(ret[1]):
                ret[0] = ret[1]-1
            else:
                ret[0] = -0.5
        if not np.isfinite(ret[-1]):
            if len(ret) >= 3 and np.isfinite(ret[-3]):
                ret[-1] = ret[-2] + (ret[-2] - ret[-3])
            else:
                ret[-1] = ret[-2]+1

        return np.array(ret)

    def get_axis_label(self, j_binning):
        """Return variable name."""
        return self.binning.binnings[j_binning].variable
