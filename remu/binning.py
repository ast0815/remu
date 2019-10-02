"""Binning/histogramming classes for scientific computing

YAML interface
==============

All classes defined in `binning` can be stored as and read from YAML files
using the ``binning.yaml`` module::

    with open("filename.yml", 'w') as f:
        binning.yaml.dump(some_binning, f)

    with open("filename.yml", 'r') as f:
        some_binning = binning.yaml.full_load(f)

"""

from __future__ import division
from six.moves import map, zip, range
from copy import copy, deepcopy
import yaml
import re
import numpy as np
from numpy.lib.recfunctions import rename_fields
import csv
from tempfile import TemporaryFile

class PhaseSpace(yaml.YAMLObject):
    """A PhaseSpace defines the possible combinations of variables that characterize an event.

    Parameters
    ----------

    variables : iterable of strings
        The set of variables that define the phase space.

    Attributes
    ----------

    variables : set of str
        The set of variables that define the phase space.

    Notes
    -----

    A PhaseSpace can be seen as the carthesian product of its `variables`::

        >>> ps = PhaseSpace(variables=['a', 'b', 'c'])
        >>> print ps
        ('a' X 'c' X 'b')

    You can check whether a variable is part of a phase space::

        >>> 'a' in ps
        True

    Phase spaces can be compared to one another.

    Check whether two phase spaces are identical::

        >>> PhaseSpace(['a','b']) == PhaseSpace(['b', 'a'])
        True
        >>> PhaseSpace(['a', 'b']) == PhaseSpace(['a', 'c'])
        False
        >>> PhaseSpace(['a', 'b']) != PhaseSpace(['a', 'c'])
        True

    Check whether one phase space is a sub-space of the other::

        >>> PhaseSpace(['a', 'b','c')] > PhaseSpace(['a', 'b'])
        True
        >>> PhaseSpace(['a', 'c']) < PhaseSpace(['a', 'b','c'])
        True

    """

    def __init__(self, variables):
        self.variables = set(variables)

    def __contains__(self, var):
        return var in self.variables

    def __len__(self):
        return len(self.variables)

    def __eq__(self, phasespace):
        return self.variables == phasespace.variables

    def __ne__(self, phasespace):
        return self.variables != phasespace.variables

    def __le__(self, phasespace):
        return self.variables <= phasespace.variables

    def __ge__(self, phasespace):
        return self.variables >= phasespace.variables

    def __lt__(self, phasespace):
        return (self.variables <= phasespace.variables) and not (self.variables == phasespace.variables)

    def __gt__(self, phasespace):
        return (self.variables >= phasespace.variables) and not (self.variables == phasespace.variables)

    def __mul__(self, phasespace):
        return PhaseSpace(variables = (self.variables | phasespace.variables))

    def __div__(self, phasespace):
        return PhaseSpace(variables = (self.variables - phasespace.variables))

    def __truediv__(self, phasespace):
        # Python 3 div operator
        return self.__div__(phasespace)

    def __str__(self):
        return "('" + "' X '".join(self.variables) + "')"

    def __repr__(self):
        return '%s(variables=%r)'%(type(self).__name__, self.variables)

    def clone(self):
        """Return a copy of the object."""
        return deepcopy(self)

    @classmethod
    def to_yaml(cls, dumper, obj):
        return dumper.represent_sequence('!PhaseSpace', list(obj.variables))

    @classmethod
    def from_yaml(cls, loader, node):
        seq = loader.construct_sequence(node)
        return cls(variables=seq)

    yaml_loader = yaml.FullLoader
    yaml_tag = u'!PhaseSpace'

class Bin(yaml.YAMLObject):
    """A Bin is a container for a value that is defined on a subset of an n-dimensional phase space.

    Parameters
    ----------

    phasespace : PhaseSpace
        The :class:`PhaseSpace` the `Bin` resides in.
    value : float, optional
        The initialization value of the bin. Default: 0.0
    entries : int, optional
        The initialization value of the number of entries. Default: 0
    sumw2 : float, optional
        The initialization value of the sum of squared weights. Default: ``value**2``
    value_array : slice of ndarray, optional
        A slice of a numpy array, where the value of the bin will be stored.
        Default: ``None``
    entries_array : slice of ndarray, optional
        A slice of a numpy array, where the number entries will be stored.
        Default: ``None``
    sumw2_array : slice of ndarray, optional
        A slice of a numpy array, where the squared weights will be stored.
        Default: ``None``
    dummy : bool, optional
        Do not create a any arrays to store the data.
        Default: ``False``

    Attributes
    ----------

    value : float
        The value of the bin.
    entries : int
        The number of entries in the bin.
    sumw2 : float
        The sum of squared weights in the bin.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the bin is defined on

    """

    def __init__(self, **kwargs):
        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise TypeError("Undefined phase space!")

        if not kwargs.pop('dummy', False):
            self.value_array = kwargs.pop('value_array', None)
            if self.value_array is None:
                self.value_array = np.array([kwargs.pop('value', 0.)], dtype=float)
            self.entries_array = kwargs.pop('entries_array', None)
            if self.entries_array is None:
                self.entries_array = np.array([kwargs.pop('entries', 0)], dtype=int)
            self.sumw2_array = kwargs.pop('sumw2_array', None)
            if self.sumw2_array is None:
                self.sumw2_array = np.array([kwargs.pop('sumw2', self.value**2)], dtype=float)
        else:
            for key in ['value_array', 'entries_array', 'sumw2_array']:
                if key in kwargs:
                    del kwargs[key]

        if len(kwargs) > 0:
            raise TypeError("Unknown kwargs: %s"%(kwargs,))

    @property
    def value(self):
        """(float) The value of the bin.

        The sum of weights.
        """
        return self.value_array[0]

    @value.setter
    def value(self, v):
        self.value_array[0] = v

    @property
    def entries(self):
        """(int) The number of entries in the bin."""
        return self.entries_array[0]

    @entries.setter
    def entries(self, v):
        self.entries_array[0] = v

    @property
    def sumw2(self):
        """(float) The sum of squared weights in the bin."""
        return self.sumw2_array[0]

    @sumw2.setter
    def sumw2(self, v):
        self.sumw2_array[0] = v

    def event_in_bin(self, event):
        """Check whether the variable combination falls within the bin.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable
            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        bool
            Whether or not the variable combination lies within the bin.

        """

        raise NotImplementedError("This method must be defined in an inheriting class.")

    def fill(self, weight=1.):
        """Add the weight(s) to the bin.

        Also increases the number of entries and sum of squared weights accordingly.

        Parameters
        ----------

        weight : float or iterable of floats, optional
            Weight(s) to be added to the value of the bin.

        """

        try:
            # Does the weight have a length?
            n = len(weight)
        except TypeError:
            # No
            w = weight
            w2 = w**2
            n = 1
        else:
            # Yes
            weight = np.asarray(weight)
            w = np.sum(weight)
            w2 = np.sum(weight**2)

        self.value += w
        self.entries += n
        self.sumw2 += w2

    def is_dummy(self):
        """Return `True` if there is no data array linked to this bin."""
        try:
            self.value_array
        except AttributeError:
            return True
        else:
            return False

    def __contains__(self, event):
        """Return True if the event falls within the bin."""
        return self.event_in_bin(event)

    def __eq__(self, other):
        """Bins are equal if they are of the same type, defined on the same phase space."""
        return (type(self) == type(other)
                and self.phasespace == other.phasespace)

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        ret = deepcopy(self)
        ret.value = self.value + other.value
        ret.entries = self.entries + other.entries
        ret.sumw2 = self.sumw2 + other.sumw2
        return ret

    def __sub__(self, other):
        ret = deepcopy(self)
        ret.value = self.value - other.value
        return ret

    def __mul__(self, other):
        ret = deepcopy(self)
        ret.value = self.value * other.value
        return ret

    def __div__(self, other):
        ret = deepcopy(self)
        ret.value = self.value / other.value
        return ret

    def __truediv__(self, other):
        # Python 3 div operator
        return self.__div__(other)

    def __repr__(self):
        return '%s(%s)'%(type(self).__name__, ", ".join(["%s=%r"%(k,v) for k,v in self._get_clone_kwargs().items()]))

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""
        args = {
            'phasespace': deepcopy(self.phasespace),
            }
        if self.is_dummy() or kwargs.get('dummy', False):
            args['dummy'] = True
        else:
            args.update({
                'value_array': deepcopy(self.value_array),
                'entries_array': deepcopy(self.entries_array),
                'sumw2_array': deepcopy(self.sumw2_array)
                })
        args.update(kwargs)
        return args

    def clone(self, **kwargs):
        """Create a functioning copy of the Bin.

        Can specify additional kwargs for the initialisation of the new Binning.

        """
        args = self._get_clone_kwargs(**kwargs)
        return type(self)(**args)

    @classmethod
    def to_yaml(cls, dumper, obj):
        dic = obj._get_clone_kwargs(dummy=True)
        if not obj.is_dummy():
            del dic['dummy']
        return dumper.represent_mapping(cls.yaml_tag, dic)

    @classmethod
    def from_yaml(cls, loader, node):
        dic = loader.construct_mapping(node, deep=True)
        return cls(**dic)

    yaml_loader = yaml.FullLoader
    yaml_tag = u'!Bin'

class RectangularBin(Bin):
    """A Bin defined by min and max values in all variables.

    Parameters
    ----------

    variables : iterable of str
        The variables with defined edges.

    edges : iterable of (int, int)
        lower and upper edges for all variables::

            [[x_lower, x_upper], [y_lower, y_upper], ...]

    include_lower : bool, optional
        Does the bin include the lower edges?

    include_upper : bool, optional
        Does the bin include the upper edges?

    **kwargs : optional
        Additional keyword arguments are passed to :class:`Bin`.

    Attributes
    ----------

    value : float
        The value of the bin.
    entries : int
        The number of entries in the bin.
    sumw2 : float
        The sum of squared weights in the bin.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the bin is defined on
    variables : tuple of str
        The variable names.
    edges : tuple of (int, int)
        The bin edges for each variable.
    include_lower : bool
        Does the bin include the lower edges?
    include_upper : bool
        Does the bin include the upper edges?

    """

    def __init__(self, variables, edges, include_lower=True, include_upper=False, **kwargs):
        self.variables = tuple(variables)
        self.edges = tuple(tuple(x) for x in edges)
        self.include_lower = bool(include_lower)
        self.include_upper = bool(include_upper)

        # Create PhaseSpace from edges if necessary
        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            kwargs['phasespace'] = PhaseSpace(self.variables)

        # Handle default bin initialization
        Bin.__init__(self, **kwargs)

        # Check that all edges are valid tuples
        for i, var in enumerate(self.variables):
            if var not in self.phasespace:
                raise ValueError("Variable not part of PhaseSpace: %s"%(var,))
            mi, ma = self.edges[i]

            if ma < mi:
                raise ValueError("Upper edge is smaller than lower edge for variable %s."%(var,))

    def event_in_bin(self, event):
        """Check whether an event is within all bin edges.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable
            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        bool
            Whether or not the variable combination lies within the bin.

        """

        inside = True

        for i, var in enumerate(self.variables):
            mi, ma = self.edges[i]
            val = event[var]
            if self.include_lower:
                if val < mi:
                    inside = False
                    break
            else:
                if val <= mi:
                    inside = False
                    break
            if self.include_upper:
                if val > ma:
                    inside = False
                    break
            else:
                if val >= ma:
                    inside = False
                    break

        return inside

    def get_center(self):
        """Return the bin center coordinates.

        Returns
        -------

        ndarray
            The center coordinates for each variable.

        """
        arr = np.asfarray(self.edges)
        return arr.sum(axis=1)/2.

    def __eq__(self, other):
        """RectangularBins are equal if they have the same edges."""
        return (Bin.__eq__(self, other)
            and sorted(zip(self.variables, self.edges)) == sorted(zip(other.variables, other.edges))
            and self.include_lower == other.include_lower
            and self.include_upper == other.include_upper)

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        args = {
            'include_upper': self.include_upper,
            'include_lower': self.include_lower,
            'variables': list(self.variables),
            'edges': np.asarray(self.edges).tolist(),
            }
        args.update(Bin._get_clone_kwargs(self, **kwargs))
        return args

    yaml_tag = u'!RectangularBin'

class CartesianProductBin(Bin):
    """A Bin that is part of a CartesianProductBinning.

    An event is part of a bin, if it has the right data indices in the
    constituent binnings.

    Parameters
    ----------

    binnings : iterable of Binning
    data_indices : iterable of int
        Specifies the constituent binnings and the respective data indices.
    **kwargs : optional
        Additional keyword arguments are passed to :class:`Bin`.

    Attributes
    ----------

    value : float
        The value of the bin.
    entries : int
        The number of entries in the bin.
    sumw2 : float
        The sum of squared weights in the bin.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the bin is defined on
    binnings : tuple of Binning
    data_indices : tuple of int
        Specifies the constituent binnings and the respective data indices.

    """

    def __init__(self, binnings, data_indices, **kwargs):
        self.binnings = tuple(binnings)
        self.data_indices = tuple(data_indices)

        # Create PhaseSpace from binnings if necessary
        if 'phasespace' not in kwargs:
            kwargs['phasespace'] = PhaseSpace([])
            for binning in self.binnings:
                kwargs['phasespace'] *= binning.phasespace

        Bin.__init__(self, **kwargs)

    def event_in_bin(self, event):
        """Check whether an event is within the bin.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable
            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        bool
            Whether or not the variable combination lies within the bin.

        """

        # Check that the event is at the right data position in all binnings
        for binning, i in zip(self.binnings, self.data_indices):
            if binning.get_event_data_index(event) != i:
                return False
        else:
            return True

    def __eq__(self, other):
        """CartesianProductBins are equal, if the binnings and indices are equal."""
        try:
            if len(self.binnings) != len(other.binnings):
                return False
            # Try both combinations of self and other
            for A, B in [(self, other), (other, self)]:
                for self_binning, i in zip(A.binnings, A.data_indices):
                    # For each binning and index in self...
                    for other_binning, j in zip(B.binnings, B.data_indices):
                        # ... check that there is a matchin binning and index in other
                        if self_binning == other_binning and i == j:
                            break
                    else:
                        # Otherwise return `False`
                        return False
            # Found a match for all elements
            return Bin.__eq__(self, other)
        except AttributeError:
            return False

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        args = {
            'binnings': [binning.clone(dummy=True) for binning in self.binnings],
            'data_indices': list(self.data_indices),
            }
        args.update(Bin._get_clone_kwargs(self, **kwargs))
        return args

    yaml_tag = u'!CartesianProductBin'

class Binning(yaml.YAMLObject):
    """A Binning is a set of disjunct Bins.

    Parameters
    ----------

    bins : list of Bin
        The list of disjoint bins.
    subbinnings : dict of {bin_index: Binning}, optional
        Subbinnings to replace certain bins.
    value_array : slice of ndarray, optional
        A slice of a numpy array, where the values of the bins will be stored.
    entries_array : slice of ndarray, optional
        A slice of a numpy array, where the number of entries will be stored.
    sumw2_array : slice of ndarray, optional
        A slice of a numpy array, where the squared weights will be stored.
    phasespace : PhaseSpace, optional
        The :class:`PhaseSpace` the binning resides in.
    dummy : bool, optional
        Do not create any arrays to store the data.

    Attributes
    ----------

    bins : tuple of Bin
        The list of disjoint bins on the PhaseSpace.
    nbins : int
        The number of bins in the binning.
    data_size : int
        The number of elements in the data arrays.
        Might differ from ``nbins`` due to subbinnings.
    subbinnings : dict of {bin_index: Binning}, optional
        Subbinnings to replace certain bins.
    value_array : slice of ndarray
        A slice of a numpy array, where the values of the bins are stored.
    entries_array : slice of ndarray
        A slice of a numpy array, where the number of entries are stored.
    sumw2_array : slice of ndarray
        A slice of a numpy array, where the squared weights are stored.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the binning resides in.

    Notes
    -----

    Subbinnings are used to get a finer binning within a given bin. The bin to
    be replaced by the finer binning is specified using the *native* bin
    index, i.e. the number it would have before the sub binnings are assigned.
    Subbinnings are inserted into the numpy arrays at the position of the
    original bins. This changes the *effective* bin number of all later bins.

    The data itself is stored in Numpy arrays (or views of such) that are
    managed by the :class:`Binning`. The arrays are linked to the contained
    :class:`Bin` objects and subbinnings by setting their respective storage
    arrays to sliced views of the data arrays. The original arrays in the bins
    and subbinnings will always be replaced.

    """

    def __init__(self, bins, subbinnings={}, value_array=None, entries_array=None,
                 sumw2_array=None, phasespace=None, dummy=False):

        if isinstance(bins, _BinProxy):
            self.bins = bins
        else:
            self.bins = tuple(bins)
        self.subbinnings = dict(subbinnings)
        self.phasespace = phasespace
        if self.phasespace is None:
            self.phasespace = self._get_phasespace()

        self.nbins = len(self.bins)

        self.data_size = self.nbins
        for binning in self.subbinnings.values():
            self.data_size += binning.data_size - 1 # Minus one, since one bin gets replaced

        if not dummy:
            self.value_array = value_array
            if self.value_array is None:
                self.value_array = np.zeros(self.data_size, dtype=float)
            if self.value_array.shape != (self.data_size,):
                raise TypeError("Value array shape is not same as (data_size,)!")
            self.entries_array = entries_array
            if self.entries_array is None:
                self.entries_array = np.zeros(self.data_size, dtype=int)
            if self.entries_array.shape != (self.data_size,):
                raise TypeError("Entries array shape is not same as (data_size,)!")
            self.sumw2_array = sumw2_array
            if self.sumw2_array is None:
                self.sumw2_array = np.zeros(self.data_size, dtype=float)
            if self.sumw2_array.shape != (self.data_size,):
                raise TypeError("Sumw2 array shape is not same as (data_size,)!")
            self.link_arrays()
        else:
            self.value_array = None
            self.entries_array = None
            self.sumw2_array = None

    def _get_phasespace(self):
        """Get PhaseSpace from Bins and subbinnings."""
        ps = PhaseSpace([])
        for bin in self.bins:
            ps *= bin.phasespace
        for binning in self.subbinnings.values():
            ps *= binning.phasespace
        return ps

    def link_arrays(self):
        """Link the data storage arrays into the bins and sub_binnings."""
        self._link_bins()
        self._link_subbinnings()

    def _link_bins(self):
        for i, bin in enumerate(self.bins):
            j = self.get_bin_data_index(i)
            bin.value_array = self.value_array[j:j+1]
            bin.entries_array = self.entries_array[j:j+1]
            bin.sumw2_array = self.sumw2_array[j:j+1]

    def _link_subbinnings(self):
        for i, binning in self.subbinnings.items():
            j = self.get_bin_data_index(i)
            n = binning.data_size
            binning.value_array = self.value_array[j:j+n]
            binning.entries_array = self.entries_array[j:j+n]
            binning.sumw2_array = self.sumw2_array[j:j+n]
            # Also make the subbinnings link the new arrays
            binning.link_arrays()

    def get_event_data_index(self, event):
        """Get the data array index of the given event.

        Returns `None` if the event does not belong to any bin.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable
            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        int or None
            The bin number

        See also
        --------

        get_event_bin_index

        """

        bin_i = self.get_event_bin_index(event)
        data_i = self.get_bin_data_index(bin_i)
        if bin_i in self.subbinnings:
            data_i += self.subbinnings[bin_i].get_event_data_index(event)
        return data_i

    def get_event_bin_index(self, event):
        """Get the bin number of the given event.

        Returns `None` if the event does not belong to any bin.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable
            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        int or None
            The bin number

        Notes
        -----

        The bin number can be used to access the corresponding :class:`Bin`,
        or the subbinning in that bin (if it exists)::

            i = binning.get_event_bin_index(event)
            binning.bins[i]
            binning.subbinnings[i]

        This is *not* the same as the corresponding index in the data array if
        there are any subbinnings present.

        This is a dumb method that just loops over all bins until it finds a
        fitting one. It should be replaced with something smarter for more
        specifig binning classes.

        See also
        --------

        get_event_data_index
        get_event_bin

        """

        for i in range(len(self.bins)):
            if event in self.bins[i]:
                return i

        return None

    def get_bin_data_index(self, bin_i):
        """Calculate the data array index from the bin number."""

        if bin_i is None:
            return None

        data_i = bin_i
        for i, binning in self.subbinnings.items():
            if i < bin_i:
                data_i = data_i + (binning.data_size - 1) # Minus one, because the original bin is replaced
        return data_i

    def get_data_bin_index(self, data_i):
        """Calculate the bin number from the data array index.

        All data indices inside a subbinning will return the bin index of that
        subbinning.

        """

        if data_i is None:
            return None

        bin_i = data_i
        for i in sorted(self.subbinnings.keys()):
            if i > bin_i:
                return bin_i
            if i + self.subbinnings[i].data_size > bin_i:
                return i
            bin_i -= (self.subbinnings[i].data_size - 1)

        return bin_i

    def get_event_bin(self, event):
        """Get the bin of the event.

        Returns `None` if the event does not fit in any bin.

        Parameters
        ----------

        event : dict like
            A dictionary (or similar object) with one value of each variable

            in the binning, e.g.::

                {'x': 1.4, 'y': -7.47}

        Returns
        -------

        Bin or None
            The :class:`Bin` object the event fits into.

        """

        i = self.get_event_bin_index(event)
        if i is not None:
            return self.bins[i]
        else:
            return None

    def get_adjacent_bin_indices(self):
        """Return a list of adjacent bin indices.

        Returns
        -------

        adjacent_indices : list of ndarray
            The adjacent indices of each bin

        """

        # The general case is that we just don't know which bin is adjacent to
        # which. Return a list of empty lists.

        return [np.array([], dtype=int)] * self.nbins

    def get_adjacent_data_indices(self):
        """Return a list of adjacent data indices.

        Returns
        -------

        adjacent_indices : list of ndarray
            The adjacent indices of each data index

        Notes
        -----

        Data indices inside a subbinning will only ever be adjacent to other
        indices inside the same subbinning. There is no information available
        about which bins in a subbinning are adjacent to which bins in the
        parent binning.

        """

        # Start with adjacent bins
        i_bin = self.get_adjacent_bin_indices()

        # Replace bin indices with data indices
        # and remove references to subbinnings
        i_data = []
        for i, adj in enumerate(i_bin):
            if i not in self.subbinnings:
                # Regular bin
                # Add neighbouring bins translated to data indices
                i_data.append([])
                for j in adj:
                    if j not in self.subbinnings:
                        i_data[-1].append(self.get_bin_data_index(j))
                i_data[-1] = np.array(i_data[-1], dtype=int)
            else:
                # Subbinning
                # Add its adjacent data indices offset to correct position
                offset = self.get_bin_data_index(i)
                for adj in self.subbinnings[i].get_adjacent_data_indices():
                    i_data.append(adj + offset)

        return i_data

    def fill(self, event, weight=1, raise_error=False, rename={}):
        """Fill the events into their respective bins.

        Parameters
        ----------

        event : [iterable of] dict like or Numpy structured array or Pandas DataFrame
            The event(s) to be filled into the binning.
        weight : float or iterable of floats, optional
            The weight of the event(s).
            Can be either a scalar which is then used for all events
            or an iterable of weights for the single events.
            Default: 1.
        raise_error : bool, optional
            Raise a ValueError if an event is not in the binning.
            Otherwise ignore the event.
            Default: False
        rename : dict, optional
            Dict for translating event variable names to binning variable names.
            Default: `{}`, i.e. no translation

        """

        try:
            if len(event) == 0:
                # Empty iterable? Stop right here
                return
        except TypeError:
            # Not an iterable
            event = [event]

        if len(rename) > 0:
            try:
                # Numpy array?
                event = rename_fields(event, rename)
            except AttributeError:
                try:
                    # Pandas DataFrame?
                    event = event.rename(index=str, columns=rename)
                except AttributeError:
                    # Dict?
                    for e in event:
                        for name in rename:
                            e[rename[name]] = e[name]

        ibins = None

        if ibins is None:
            try:
                # Try to get bin numbers from a pandas DataFrame
                ibins = list(map(lambda irow: self.get_event_data_index(irow[1]), event.iterrows()))
            except AttributeError:
                # Seems like this is not a DataFrame
                pass

        if ibins is None:
            try:
                # Try to get bin numbers from structured numpy array
                ibins = list(map(self.get_event_data_index, np.nditer(event)))
            except TypeError:
                # Seems like this is not a structured numpy array
                pass

        if ibins is None:
            try:
                # Try to get bin numbers from any iterable of events
                ibins = list(map(self.get_event_data_index, event))
            except TypeError:
                # We probably only have a single event
                ibins = [self.get_event_data_index(event)]

        if raise_error and None in ibins:
            raise ValueError("Event not part of binning!")

        # Compare len of weight list and event list
        try:
            if len(ibins) != len(weight):
                raise ValueError("Different length of event and weight lists!")
        except TypeError:
            weight = [weight] * len(ibins)

        for i, w in zip(ibins, weight):
            if i is not None:
                self.fill_data_index(i, w)

    def fill_data_index(self, i, weight=1.):
        """Add the weight(s) to the given data position.

        Also increases the number of entries and sum of squared weights accordingly.

        Parameters
        ----------

        i : int
            The index of the data arrays to be filled.
        weight : float or iterable of floats, optional
            Weight(s) to be added to the value of the bin.

        """

        try:
            # Does the weight have a length?
            n = len(weight)
        except TypeError:
            # No
            w = weight
            w2 = w**2
            n = 1
        else:
            # Yes
            weight = np.asarray(weight)
            w = np.sum(weight)
            w2 = np.sum(weight**2)

        self.value_array[i] += w
        self.entries_array[i] += n
        self.sumw2_array[i] += w2

    @staticmethod
    def _genfromtxt(filename, delimiter=',', names=True, chunksize=10000):
        """Replacement for numpy's genfromtxt, that should need less memory."""

        with open(filename, 'r') as f:
            if names:
                namelist = f.readline().split(delimiter)
                dtype = [ (name.strip(), float) for name in namelist ]
            else:
                namelist = None
                dtype = float

            arr = np.array([], dtype=dtype)
            rows = []
            for line in f:
                if len(rows) >= chunksize:
                    arr = np.concatenate((arr, np.array(rows, dtype=dtype)), axis=0)
                    rows = []
                rows.append( tuple(map(float, line.split(delimiter))) )
            arr = np.concatenate((arr, np.array(rows, dtype=dtype)), axis=0)

        return arr

    _csv_buffer = {}
    @classmethod
    def _load_csv_file_buffered(cls, filename, chunksize):
        """Load a CSV file and save the resulting array in a temporary file.

        If the same file is loaded a second time, the buffer is loaded instead
        of re-parsing the CSV file.
        """

        if filename in cls._csv_buffer:
            # File has been loaded before
            f = cls._csv_buffer[filename]
            f.seek(0)
            arr = np.load(f)
        else:
            # New file
            f = TemporaryFile()
            arr = cls._genfromtxt(filename, delimiter=',', names=True, chunksize=chunksize)
            np.save(f, arr)
            cls._csv_buffer[filename] = f

        return arr

    @classmethod
    def fill_multiple_from_csv_file(cls, binnings, filename, weightfield=None, weight=1.0, rename={}, cut_function=lambda x: x, buffer_csv_files=False, chunksize=10000, **kwargs):
        """Fill multiple Binnings from the same csv file(s).

        This method saves time, because the numpy array only has to be
        generated once. Other than the list of binnings to be filled, the
        (keyword) arguments are identical to the ones used by the instance
        method :meth:`fill_from_csv_file`.

        """

        # Handle lists recursively
        if isinstance(filename, list):
            try:
                for item, w in zip(filename, weight):
                    cls.fill_multiple_from_csv_file(binnings, item, weightfield=weightfield, weight=w, rename=rename, cut_function=cut_function, buffer_csv_files=buffer_csv_files, **kwargs)
            except TypeError:
                for item in filename:
                    cls.fill_multiple_from_csv_file(binnings, item, weightfield=weightfield, weight=weight, rename=rename, cut_function=cut_function, buffer_csv_files=buffer_csv_files, **kwargs)
            return

        if buffer_csv_files:
            data = cls._load_csv_file_buffered(filename, chunksize=chunksize)
        else:
            data = cls._genfromtxt(filename, delimiter=',', names=True, chunksize=chunksize)
        data = rename_fields(data, rename)
        data = cut_function(data)

        if weightfield is not None:
            weight = data[weightfield] * weight

        for binning in binnings:
            binning.fill(data, weight=weight, **kwargs)

    def fill_from_csv_file(self, *args, **kwargs):
        """Fill the binning with events from a CSV file.

        Parameters
        ----------

        filename : string or list of strings
            The csv file with the data. Can be a list of filenames.
        weightfield : string, optional
            The column with the event weights.
        weight : float or iterable of floats, optional
            A single weight that will be applied to all events in the file.
            Can be an iterable with one weight for each file if `filename` is a list.
        rename : dict, optional
            A dict with columns that should be renamed before filling::

                {'csv_name': 'binning_name'}

        cut_function : function, optional
            A function that modifies the loaded data before filling into the binning,
            e.g.::

                cut_function(data) = data[ data['binning_name'] > some_threshold ]

            This is done *after* the optional renaming.
        buffer_csv_files : bool, optional
            Save the results of loading CSV files in temporary files
            that can be recovered if the same CSV file is loaded again. This
            speeds up filling multiple Binnings with the same CSV-files considerably!
            Default: False
        chunksize : int, optional
            Load csv file in chunks of <chunksize> rows. This reduces the memory
            footprint of the loading operation, but can slow it down.
            Default: 10000

        Notes
        -----

        The file must be formated like this::

            first_varname,second_varname,...
            <first_value>,<second_value>,...
            <first_value>,<second_value>,...
            <first_value>,<second_value>,...
            ...

        For example::

            x,y,z
            1.0,2.1,3.2
            4.1,2.0,2.9
            3,2,1

        All values are interpreted as floats. If `weightfield` is given, that
        field will be used as weigts for the event. Other keyword arguments
        are passed on to the Binning's :meth:`fill` method. If filename is a list,
        all elemets are handled recursively.

        """

        # Actual filling is handled by static method
        Binning.fill_multiple_from_csv_file([self], *args, **kwargs)

    def reset(self, value=0., entries=0, sumw2=0.):
        """Reset all bin values to 0.

        Parameters
        ----------

        value : float, optional
            Set the bin values to this value.
        entries : int, optional
            Set the number of entries in each bin to this value.
        sumw2 : float, optional
            Set the sum of squared weights in each bin to this value.

        """

        self.value_array.fill(value)
        self.entries_array.fill(entries)
        self.sumw2_array.fill(sumw2)

    def get_values_as_ndarray(self, shape=None, indices=None):
        """Return the bin values as ndarray.

        Parameters
        ----------

        shape: tuple of ints
            Shape of the resulting array.
            Default: ``(len(bins),)``
        indices: list of ints
            Only return the given bins.
            Default: Return all bins.

        Returns
        -------

        ndarray
            An ndarray with the values of the bins.

        """

        if indices is None:
            indices = slice(None, None, None)

        ret = np.array(self.value_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_values_from_ndarray(self, arr):
        """Set the bin values to the values of the ndarray."""

        self.value_array.flat[:] = np.asarray(arr).flat

    def get_entries_as_ndarray(self, shape=None, indices=None):
        """Return the number of entries in the bins as ndarray.

        Parameters
        ----------

        shape: tuple of ints
            Shape of the resulting array.
            Default: ``(len(bins),)``
        indices: list of ints
            Only return the given bins.
            Default: Return all bins.

        Returns
        -------

        ndarray
            An ndarray with the numbers of entries of the bins.

        """
        if indices is None:
            indices = slice(None, None, None)

        ret = np.array(self.entries_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_entries_from_ndarray(self, arr):
        """Set the number of bin entries to the values of the ndarray."""

        self.entries_array.flat[:] = np.asarray(arr).flat

    def get_sumw2_as_ndarray(self, shape=None, indices=None):
        """Return the sum of squared weights in the bins as ndarray.

        Parameters
        ----------

        shape: tuple of ints
            Shape of the resulting array.
            Default: ``(len(bins),)``
        indices: list of ints
            Only return the given bins.
            Default: Return all bins.

        Returns
        -------

        ndarray
            An ndarray with the sum of squared weights of the bins.

        """
        if indices is None:
            indices = slice(None, None, None)

        ret = np.copy(self.sumw2_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_sumw2_from_ndarray(self, arr):
        """Set the sums of squared weights to the values of the ndarray."""

        self.sumw2_array.flat[:] = np.asarray(arr).flat

    def event_in_binning(self, event):
        """Check whether an event fits into any of the bins."""

        i = self.get_event_data_index(event)
        if i is None:
            return False
        else:
            return True

    def is_dummy(self):
        """Return `True` if there is no data array linked to this binning."""
        if self.value_array is None:
            return True
        else:
            return False

    def __contains__(self, event):
        return self.event_in_binning(event)

    def __eq__(self, other):
        """Binnings are equal if all bins and the phase space are equal."""
        return (self.bins == other.bins
            and self.phasespace == other.phasespace
            and self.subbinnings == other.subbinnings)

    def __ne__(self, other):
        return not self == other

    def marginalize_subbinnings_on_ndarray(self, array, bin_indices=None):
        """Marginalize out the bins corresponding to the subbinnings.

        Parameters
        ----------

        array : ndarray
            The data to work on.
        bin_indices : list of int, optional
            The bin indices of the subbinnings to be marginalized.
            If no indices are specified, all subbinnings are marginalized.

        Returns
        -------

        new_array : ndarray

        """

        if bin_indices is None:
            bin_indices = self.subbinnings.keys()

        # Create working copy of input array
        new_array = np.array(array)

        # Determine indices to be removed and set new values
        remove_i = []
        for i in bin_indices:
            if i in self.subbinnings:
                binning = self.subbinnings[i]
            else:
                raise ValueError("No subbinning at bin index %d!"%(i,))
            i_data = self.get_bin_data_index(i)
            n_data = binning.data_size
            remove_i.extend(range(i_data+1, i_data+n_data)) # Skip first one, since we substitute a single bin

            # Set marginalized value
            new_array[i] = np.sum(new_array[i_data:i_data+n_data])

        # Remove marginalized elements
        remove_i = np.array(sorted(remove_i))
        new_array = np.delete(new_array, remove_i, axis=0)

        return new_array

    def marginalize_subbinnings(self, bin_indices=None):
        """Return a clone of the Binning with subbinnings removed.

        Parameters
        ----------

        bin_indices : list of int, optional
            The bin indices of the subbinnings to be marginalized.
            If no indices are specified, all subbinnings are marginalized.

        Returns
        -------

        new_binning : Binning

        """

        if bin_indices is None:
            bin_indices = self.subbinnings.keys()

        # Clone the subbinnings that will remain in the binning
        subbinnings = {}
        for i in self.subbinnings:
            if i not in bin_indices:
                binning = self.subbinnings[i].clone(dummy=True)
                subbinnings[i] = binning

        kwargs = {'subbinnings': subbinnings}

        if self.is_dummy():
            pass
        else:
            kwargs.update({
                'value_array': self.marginalize_subbinnings_on_ndarray(self.value_array, bin_indices),
                'entries_array': self.marginalize_subbinnings_on_ndarray(self.entries_array, bin_indices),
                'sumw2_array': self.marginalize_subbinnings_on_ndarray(self.sumw2_array, bin_indices),
                })

        return self.clone(**kwargs)

    def insert_subbinning_on_ndarray(self, array, bin_index, insert_array):
        """Insert values of a new subbinning into the array.

        Parameters
        ----------

        array : ndarray
            The data to work on.
        bin_index : int
            The bin to be replaced with the subbinning.
        insert_array : ndarrau
            The array to be inserted.

        Returns
        -------

        new_array : ndarray
            The modified array.

        """

        i_data = self.get_bin_data_index(bin_index)
        new_array = np.insert(array, i_data+1, insert_array[1:], axis=0) # Do not insert the first element
        new_array[i_data] = insert_array[0] # Instead set overwrite the values of the bin
        return new_array

    def insert_subbinning(self, bin_index, binning):
        """Insert a new subbinning into the binning.

        Parameters
        ----------

        bin_index : int
            The bin to be replaced with the subbinning.
        binning : Binning
            The new subbinning

        Returns
        -------

        new_binning : Binning
            A copy of this binning with the new subbinning.

        Warnings
        --------

        This will replace the content of the bin with the content of the new
        subbinning!

        """

        if bin_index in self.subbinnings:
            raise ValueError("Bin %d already has a subbinning!"%(bin_index,))

        subbinnings = {}
        for i, b in self.subbinnings.items():
            subbinnings[i] = b.clone(dummy=True)

        subbinnings[bin_index] = binning

        kwargs = {
            'subbinnings': subbinnings,
            'value_array': self.insert_subbinning_on_ndarray(self.value_array, bin_index, binning.value_array),
            'entries_array': self.insert_subbinning_on_ndarray(self.entries_array, bin_index, binning.entries_array),
            'sumw2_array': self.insert_subbinning_on_ndarray(self.sumw2_array, bin_index, binning.sumw2_array),
            }

        return self.clone(**kwargs)

    def __add__(self, other):
        ret = self.clone()
        ret.set_values_from_ndarray(self.get_values_as_ndarray() + other.get_values_as_ndarray())
        ret.set_entries_from_ndarray(self.get_entries_as_ndarray() + other.get_entries_as_ndarray())
        ret.set_sumw2_from_ndarray(self.get_sumw2_as_ndarray() + other.get_sumw2_as_ndarray())
        return ret

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""
        args = {
            'subbinnings': dict((i, binning.clone(dummy=True)) for i, binning in self.subbinnings.items()),
            'phasespace': deepcopy(self.phasespace),
            }
        if 'bins' in kwargs:
            # Overwrite bins and do not re-create them one by one
            args['bins'] = kwargs['bins']
        else:
            # Re-create the bins one by one
            args['bins'] = [ bin.clone(dummy=True) for bin in self.bins ]
        if self.is_dummy() or kwargs.get('dummy', False):
            args['dummy'] = True
        else:
            args.update({
                'value_array': deepcopy(self.value_array),
                'entries_array': deepcopy(self.entries_array),
                'sumw2_array': deepcopy(self.sumw2_array)
                })
        args.update(kwargs)
        return args

    def clone(self, **kwargs):
        """Create a functioning copy of the Binning.

        Can specify additional kwargs for the initialisation of the new Binning.

        """
        args = self._get_clone_kwargs(**kwargs)
        return type(self)(**args)

    def __repr__(self):
        return '%s(%s)'%(type(self).__name__, ", ".join(["%s=%r"%(k,v) for k,v in self._get_clone_kwargs().items()]))

    @classmethod
    def to_yaml(cls, dumper, obj):
        dic = obj._get_clone_kwargs(dummy=True)
        if not obj.is_dummy():
            del dic['dummy']
        return dumper.represent_mapping(cls.yaml_tag, dic)

    @classmethod
    def from_yaml(cls, loader, node):
        dic = loader.construct_mapping(node, deep=True)
        return cls(**dic)

    yaml_loader = yaml.FullLoader
    yaml_tag = u'!Binning'

class RectangularBinning(Binning):
    """Binning that contains only :class:`RectangularBin`

    Parameters
    ----------

    variables : list of str
        The variables the binning is defined on.

    bin_edges : list of ((float, float), (float, float), ...)
        The list of bin edges defining the bins. The tuples contain the lower
        and upper edges of all `variables`, e.g.::

            [
            ((x_low, x_high), (y_low, y_high)),
            ((x_low, x_high), (y_low, y_high)),
            ...
            ]

    **kwargs : optional
        Additional keyword arguments will be passed to :class:`Binning`.

    Attributes
    ----------

    variables : tuple of str
        The variables corresponding to the bin edges.
    include_upper : bool
        Include the upper rather than the lower bin edges.
    bins : tuple of Bin
        The tuple of RectangularBins.
    nbins : int
        The number of bins in the binning.
    data_size : int
        The number of elements in the data arrays.
        Might differ from ``nbins`` due to subbinnings.
    subbinnings : dict of {bin_index: Binning}
        Subbinnings to replace certain bins.
    value_array : slice of ndarray
        A slice of a numpy array, where the values of the bins are stored.
    entries_array : slice of ndarray
        A slice of a numpy array, where the number of entries are stored.
    sumw2_array : slice of ndarray
        A slice of a numpy array, where the squared weights are stored.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the binning resides in.

    """

    def __init__(self, variables, bin_edges, include_upper=False, **kwargs):
        self.variables = tuple(variables)
        self.include_upper = bool(include_upper)
        bins = []
        for i, edges in enumerate(bin_edges):
            bins.append(RectangularBin(variables=variables,
                edges=bin_edges[i],
                include_upper=self.include_upper,
                include_lower=not self.include_upper,
                dummy=True))

        Binning.__init__(self, bins=bins, **kwargs)

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        variables = list(self.variables)
        bin_edges = [] # Turn all tuples into lists
        for bn in self.bins:
            bin_edges.append([list(x) for x in bn.edges])
        args = {
            'variables': list(variables),
            'bin_edges': bin_edges,
            'include_upper': self.include_upper,
            }
        args.update(Binning._get_clone_kwargs(self, bins=None, **kwargs))
        del args['bins']
        return args

    yaml_tag = u'!RectangularBinning'

class _BinProxy(object):
    """Base class for all bin proxies."""

    def __init__(self, binning):
        self.binning = binning

    def __len__(self):
        return self.binning.nbins

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        return self.binning == other.binning

    def __ne__(self, other):
        return not self == other

class _CartesianProductBinProxy(_BinProxy):
    """Indexable class that creates bins on the fly."""

    def __getitem__(self, index):
        """Dynamically build an CartesianProductBin when requested."""
        tup = self.binning.get_bin_index_tuple(index)
        index = self.binning.get_bin_data_index(index)
        val_slice = self.binning.value_array[index:index+1]
        ent_slice = self.binning.entries_array[index:index+1]
        sumw2_slice = self.binning.sumw2_array[index:index+1]
        binnings = []
        data_indices = []
        for i,j in enumerate(tup):
            binnings.append(self.binning.binnings[i])
            data_indices.append(j)
        bin = CartesianProductBin(binnings, data_indices, value_array=val_slice, entries_array=ent_slice, sumw2_array=sumw2_slice)
        return bin

class CartesianProductBinning(Binning):
    """A Binning that is the cartesian product of two or more Binnings

    Parameters
    ----------

    binnings : list of Binning
        The Binning objects to be multiplied.

    Attributes
    ----------

    binnings : tuple of Binning
        The :class:`Binning` objects that make up the Cartesian product.
    bins : proxy for Bins
        Proxy that will generate :class:`CartesianProductBin` instances,
        when accessed.
    nbins : int
        The number of bins in the binning.
    bins_shape : tuple of int
        The sizes of the constituent binnings.
    data_size : int
        The number of elements in the data arrays.
        Might differ from ``nbins`` due to subbinnings.
    subbinnings : dict of {bin_index: Binning}
        Subbinnings to replace certain bins.
    value_array : slice of ndarray
        A slice of a numpy array, where the values of the bins are stored.
    entries_array : slice of ndarray
        A slice of a numpy array, where the number of entries are stored.
    sumw2_array : slice of ndarray
        A slice of a numpy array, where the squared weights are stored.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the binning resides in.

    Notes
    -----

    This creates a Binning with as many bins as the product of the number of
    bins in the iput binnings.

    """

    def __init__(self, binnings, **kwargs):
        self.binnings = tuple(binnings)

        self.bins_shape = tuple(binning.data_size for binning in self.binnings)
        self._stepsize = [1]
        # Calculate the step size (or stride) for each binning index.
        # We use a row-major ordering (C-style).
        # The index of the later binnings varies faster than the ones before:
        #
        #   (0,0) <-> 0
        #   (0,1) <-> 1
        #   (0,2) <-> 2
        #   (1,0) <-> 3
        #   ...
        #
        # _stepsize is 1 longer than binnings and bins_shape!
        for n in reversed(self.bins_shape):
            self._stepsize.insert(0, self._stepsize[0] * n)
        self._stepsize = tuple(self._stepsize)

        self.nbins = self._stepsize[0]

        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            # Create phasespace from binnings
            phasespace = PhaseSpace([])
            for binning in self.binnings:
                phasespace *= binning.phasespace
            kwargs['phasespace'] = phasespace

        bins = kwargs.pop('bins', None)
        if bins is not None:
            raise TypeError("Cannot define bins of CartesianProductBinning! Define binnings instead.")
        else:
            # Create bin proxy
            bins = _CartesianProductBinProxy(self)

        kwargs['bins'] = bins
        Binning.__init__(self, **kwargs)

    def _link_bins(self):
        # We do not need to link each bin separately,
        # the bin proxy takes care of this
        pass

    def get_tuple_bin_index(self, tup):
        """Translate a tuple of binning specific bin indices to the linear bin index of the event.

        Turns this::

            (i_x, i_y, i_z)

        into this::

            i_bin

        The order of the indices in the tuple must conform to the order of
        `binnings`. The bins are ordered row-major (C-style), i.e. increasing
        the bin number of the last binning by one increases the overall bin
        number also by one. The increments of the other variables depend on the
        number of bins in each variable.

        """

        if None in tup:
            return None

        i_bin = 0
        for i,s in zip(tup, self._stepsize[1:]):
            i_bin += s*i

        return i_bin

    def get_bin_index_tuple(self, i_bin):
        """Translate the linear bin index of the event to a tuple of single binning bin indices.

        Turns this::

            i_bin

        into this::

            (i_x, i_y, i_z)

        The order of the indices in the tuple conforms to the order of
        `binnings`. The bins are ordered row-major (C-style), i.e. increasing
        the bin number of the last variable by one increases the overall bin
        number also by one. The increments of the other variables depend on the
        number of bins in each variable.

        """

        if i_bin is None or i_bin < 0 or i_bin >= self.nbins:
            return tuple([None]*len(self.binnings))

        tup = tuple((i_bin % t) // s for t,s in zip(self._stepsize[:-1], self._stepsize[1:]))
        return tup

    def get_event_tuple(self, event):
        """Get the variable index tuple for a given event."""

        tup = []
        for binning in self.binnings:
            i = binning.get_event_data_index(event)
            tup.append(i)

        return tuple(tup)

    def get_event_bin_index(self, event):
        """Get the bin index for a given event."""

        tup = self.get_event_tuple(event)
        return self.get_tuple_bin_index(tup)

    def get_adjacent_bin_indices(self):
        """Return a list of adjacent bin indices.

        Returns
        -------

        adjacent_indices : list of ndarray
            The adjacent indices of each bin

        """

        # Adjacent bins are based on the adjacent data indices of the
        # constituting binnings

        adj_tuple = tuple( b.get_adjacent_data_indices() for b in self.binnings )

        adj = []
        # For all bins
        for i_bin in range(self.nbins):
            i_adj = []
            # Get the tuple of binning data indices
            tup = self.get_bin_index_tuple(i_bin)
            for i_binning in range(len(tup)):
                variations = adj_tuple[i_binning][tup[i_binning]]
                var_tup = list(tup)
                for k in variations:
                    var_tup[i_binning] = k
                    i_adj.append(self.get_tuple_bin_index(var_tup))
            adj.append(np.array(sorted(i_adj), dtype=int))

        return adj

    def marginalize(self, binning_i, reduction_function=np.sum):
        """Marginalize out the given binnings and return a new CartesianProductBinning.

        Parameters
        ----------

        binning_i : iterable of int
            Iterable of index of binning to be marginalized.
        reduction_function : function
            Use this function to marginalize out the entries over the specified variables.
            Must support the `axis` keyword argument.
            Default: numpy.sum

        """

        try:
            len(binning_i)
        except TypeError:
            binning_i = [binning_i]

        # Create new binning
        new_binnings = [binning.clone(dummy=True) for binning in self.binnings]
        for i in sorted(binning_i, reverse=True):
            del new_binnings[i]

        new_binning = CartesianProductBinning(new_binnings)

        # Copy and project values, from binning without subbinnings
        axes = tuple(sorted(binning_i))
        temp_binning = self.marginalize_subbinnings()
        new_values = reduction_function(temp_binning.get_values_as_ndarray(shape=temp_binning.bins_shape), axis=axes)
        new_entries = reduction_function(temp_binning.get_entries_as_ndarray(shape=temp_binning.bins_shape), axis=axes)
        new_sumw2 = reduction_function(temp_binning.get_sumw2_as_ndarray(shape=temp_binning.bins_shape), axis=axes)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def _unpack(self):
        """Return the unpacked last remaining binning."""

        if len(self.binnings) != 1:
            raise RuntimeError("Unpacking only works if there is exactly one binning.")
        if len(self.subbinnings) != 0:
            raise RuntimeError("Unpacking only works if there is exactly zero subbinnings.")

        kwargs = {
            'value_array': self.value_array,
            'entries_array': self.entries_array,
            'sumw2_array': self.sumw2_array,
            'dummy': False,
            }

        return self.binnings[0].clone(**kwargs)

    def project(self, binning_i, **kwargs):
        """Project the binning onto the given binnings and return a new CartesianProductBinning.

        The order of the original binnings is preserved. If a single ``int`` is
        provided, the returned Binning is of the same type as the respective
        binning.

        Parameters
        ----------

        binning_i : iterable of int, or int
            Iterable of index of binning to be marginalized.
        kwargs : optional
            Additional keyword arguments are passed on to :meth:`marginalize`.

        Returns
        -------

        CartesianProductBinning or type(self.binnings[binning_i])

        """

        try:
            i = list(binning_i)
        except TypeError:
            i = [binning_i]

        # Which variables to remove
        rm_i = list(range(len(self.binnings)))
        list(map(rm_i.remove, i))

        ret = self.marginalize(rm_i, **kwargs)

        if isinstance(binning_i, int):
            return ret._unpack()
        else:
            return ret

    def __eq__(self, other):
        """CartesianProductBinnings are equal if the included Binnings match."""
        return (type(self) == type(other)
            and self.binnings == other.binnings
            and self.subbinnings == other.subbinnings)

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        args = {
            'binnings': list(binning.clone(dummy=True) for binning in self.binnings),
            }
        args.update(Binning._get_clone_kwargs(self, bins=None, **kwargs))
        del args['bins']
        return args

    yaml_tag = u'!CartesianProductBinning'

class _LinearBinProxy(_BinProxy):
    """Indexable class that creates bins on the fly."""

    def __getitem__(self, index):
        """Dynamically build a RectangularBin when requested."""
        variable = self.binning.variable
        lower = self.binning.bin_edges[index]
        upper = self.binning.bin_edges[index+1]
        data_index = self.binning.get_bin_data_index(index)
        args = {
            'variables': [variable],
            'edges': [(lower, upper)],
            'include_lower': not self.binning.include_upper,
            'include_upper': self.binning.include_upper,
            }
        if not self.binning.is_dummy():
            args.update({
                'value_array': self.binning.value_array[data_index:data_index+1],
                'entries_array': self.binning.entries_array[data_index:data_index+1],
                'sumw2_array': self.binning.sumw2_array[data_index:data_index+1],
                })
        rbin = RectangularBin(**args)
        return rbin

class LinearBinning(Binning):
    """A simple binning, defined by bin edges on a single variable.

    Parameters
    ----------

    variable : str
        The name of te defining variable.
    bin_edges : list of float
        The bin edges defining the bins.
    include_upper : bool, optional
        Include the upper edge of bins instead of the default lower edge.
    **kwargs : optional
        Additional keyword arguments are handed to :class:`Binning`.

    Attributes
    ----------

    variable : str
        The variable on which the bin edges are defined.
    bin_edges : ndarray
        The bin edges.
    include_upper : bool
        Are the upper edges included in each bin?
    bins : proxy for Bins
        Proxy that will generate :class:`RectangularBin` instances,
        when accessed.
    nbins : int
        The number of bins in the binning.
    data_size : int
        The number of elements in the data arrays.
        Might differ from ``nbins`` due to subbinnings.
    subbinnings : dict of {bin_index: Binning}, optional
        Subbinnings to replace certain bins.
    value_array : slice of ndarray
        A slice of a numpy array, where the values of the bins are stored.
    entries_array : slice of ndarray
        A slice of a numpy array, where the number of entries are stored.
    sumw2_array : slice of ndarray
        A slice of a numpy array, where the squared weights are stored.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the binning resides in.

    """

    def __init__(self, variable, bin_edges, include_upper=False, **kwargs):
        self.variable = variable
        self.bin_edges = np.asfarray(bin_edges)
        self.include_upper = bool(include_upper)
        self.nbins = self.bin_edges.size - 1

        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            # Create phasespace from variable
            phasespace = PhaseSpace([variable])
            kwargs['phasespace'] = phasespace

        bins = kwargs.pop('bins', None)
        if bins is not None:
            raise TypeError("Cannot define bins of LinearBinning! Define bin edges instead.")
        else:
            # Create bin proxy
            bins = _LinearBinProxy(self)

        kwargs['bins'] = bins
        Binning.__init__(self, **kwargs)

    def _link_bins(self):
        # We do not need to link each bin separately,
        # the bin proxy takes care of this
        pass

    def get_event_bin_index(self, event):
        """Get the bin index for a given event."""

        i = int(np.digitize(event[self.variable], self.bin_edges, right=self.include_upper))

        # Deal with Numpy's way of handling over- and underflows
        if i > 0 and i < len(self.bin_edges):
            i -= 1
        else:
            i = None

        return i

    def get_adjacent_bin_indices(self):
        """Return a list of adjacent bin indices.

        Returns
        -------

        adjacent_indices : list of ndarray
            The adjacent indices of each bin

        """

        # Adjacent bins are the ones before and after
        i_bin = np.arange(self.nbins)
        i_bin_m = i_bin - 1
        i_bin_p = i_bin + 1

        adj = list(zip(i_bin_m, i_bin_p))
        adj = list(map(np.array, adj))
        # Remove out of range elements
        adj[0] = np.array([adj[0][1]])
        adj[-1] = np.array([adj[-1][0]])

        return adj

    def slice(self, start, stop, step=1):
        """Return a new LinearBinning containing the given variable slice

        Parameters
        ----------

        start : int
        end : int
        step : int, optional
            The start and stop positions as used with Python slice objects.

        Returns
        -------

        sliced_binning : LinearBinning
            A :class:`LinearBinning` consisting of the specified slice.

        Notes
        -----

        This will remove any ``subbinnings`` the linear binning might have.

        """

        bin_slice = slice(start, stop, step)

        # Create new binning
        lower = self.bin_edges[:-1][bin_slice]
        upper = self.bin_edges[1:][bin_slice]
        new_bin_edges = list(lower) + [upper[-1]]

        new_binning = LinearBinning(variable=self.variable, bin_edges=new_bin_edges, include_upper=self.include_upper)

        # Copy and slice values
        temp_binning = self.marginalize_subbinnings()
        new_values = temp_binning.get_values_as_ndarray()[bin_slice]
        new_entries = temp_binning.get_entries_as_ndarray()[bin_slice]
        new_sumw2 = temp_binning.get_sumw2_as_ndarray()[bin_slice]

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def remove_bin_edges(self, bin_edge_indices):
        """Return a new LinearBinning with the given bin edges removed.

        The values of the bins adjacent to the removed bin edges will be
        summed up in the resulting larger bin. Please note that bin values
        are lost if the first or last binedge of a variable are removed.

        Parameters
        ----------

        bin_edge_indices : lists of integers
            A list specifying the bin edge indices that should be removed.

        Notes
        -----

        This will remove any ``subbinnings`` the linear binning might have.

        """

        # Create new binning
        new_bin_edges = list(self.bin_edges)
        for i in sorted(bin_edge_indices, reverse=True):
            del new_bin_edges[i]

        new_binning = LinearBinning(variable=self.variable, bin_edges=new_bin_edges, include_upper=self.include_upper)

        # Copy and slice values
        temp_binning = self.marginalize_subbinnings()
        new_values = temp_binning.get_values_as_ndarray()
        new_entries = temp_binning.get_entries_as_ndarray()
        new_sumw2 = temp_binning.get_sumw2_as_ndarray()

        for i in sorted(bin_edge_indices, reverse=True):
            if i > 0 and i < new_values.size:
                new_values[i-1] += new_values[i]
                new_entries[i-1] += new_entries[i]
                new_sumw2[i-1] += new_sumw2[i]
            if i < new_values.size:
                new_values = np.delete(new_values, i)
                new_entries = np.delete(new_entries, i)
                new_sumw2 = np.delete(new_sumw2, i)
            else:
                new_values = np.delete(new_values, -1)
                new_entries = np.delete(new_entries, -1)
                new_sumw2 = np.delete(new_sumw2, -1)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        args = {
            'variable': self.variable,
            'bin_edges': self.bin_edges.tolist(),
            'include_upper': self.include_upper,
            }
        args.update(Binning._get_clone_kwargs(self, bins=None, **kwargs))
        del args['bins']
        return args

    def __eq__(self, other):
        """Linear binnings are equal if the variable and edges match."""
        return (type(self) == type(other)
            and self.variable == other.variable
            and np.all(self.bin_edges == other.bin_edges)
            and self.include_upper == other.include_upper
            and self.subbinnings == other.subbinnings)

    yaml_tag = u'!LinearBinning'

class _RectilinearBinProxy(_BinProxy):
    """Indexable class that creates bins on the fly."""

    def __getitem__(self, index):
        """Dynamically build a RectangularBin when requested."""
        tup = self.binning.get_bin_index_tuple(index)
        edges = tuple( (edg[j], edg[j+1]) for edg, j in zip(self.binning.bin_edges, tup))
        data_index = self.binning.get_bin_data_index(index)
        args = {
            'variables': self.binning.variables,
            'edges': edges,
            'include_lower': not self.binning.include_upper,
            'include_upper': self.binning.include_upper,
            }
        if not self.binning.is_dummy():
            args.update({
                'value_array': self.binning.value_array[data_index:data_index+1],
                'entries_array': self.binning.entries_array[data_index:data_index+1],
                'sumw2_array': self.binning.sumw2_array[data_index:data_index+1],
                })
        rbin = RectangularBin(**args)
        return rbin

class RectilinearBinning(CartesianProductBinning):
    """Special case of :class:`CartesianProductBinning` only consisting of :class:`LinearBinning`

    Parameters
    ----------
    variables : iterable of str
    bin_edges :  iterable of iterable of float
        The variable names and bin edges for the LinearBinnings.
    include_upper : bool, optional
        Make bins include upper edges instead of lower edges.
    **kwargs : optional
        Additional keyword arguments will be passed to :class:`CartesianProductBinning`.

    Attributes
    ----------

    variables : tuple of str
        The variables on which the bin edges are defined.
    bin_edges : tuple of ndarray
        The bin edges defining the :class:`LinearBinning` objects.
    include_upper : bool
        Are the upper edges included in each bin?
    binnings : list of LinearBinning
        The :class:`LinearBinning` objects that make up the Cartesian product.
    bins : list of Bin
        The :class:`RectangularBin` instances.
    nbins : int
        The number of bins in the binning.
    bins_shape : tuple of int
        The sizes of the constituent binnings.
    data_size : int
        The number of elements in the data arrays.
        Might differ from ``nbins`` due to subbinnings.
    subbinnings : dict of {bin_index: Binning}, optional
        Subbinnings to replace certain bins.
    value_array : slice of ndarray
        A slice of a numpy array, where the values of the bins are stored.
    entries_array : slice of ndarray
        A slice of a numpy array, where the number of entries are stored.
    sumw2_array : slice of ndarray
        A slice of a numpy array, where the squared weights are stored.
    phasespace : PhaseSpace
        The :class:`PhaseSpace` the binning resides in.

    """

    def __init__(self, variables, bin_edges, include_upper=False, **kwargs):
        self.variables = tuple(variables)
        self.bin_edges = tuple(np.array(edg) for edg in bin_edges)
        self.include_upper = bool(include_upper)

        binnings = []
        for var, edges in zip(self.variables, self.bin_edges):
            binnings.append(LinearBinning(var, edges, include_upper=include_upper, dummy=True))

        kwargs['binnings'] = binnings

        bins = kwargs.pop('bins', None)
        if bins is not None:
            raise TypeError("Cannot define bins of RectilinearBinning! Define bin edges instead.")
        else:
            # Create bin proxy
            bins = _RectilinearBinProxy(self)

        CartesianProductBinning.__init__(self, **kwargs)

        # Replace cartesian proxy with one returning rectangular bins
        self.bins = bins

    def get_variable_index(self, variable):
        """Return the index of the binning corresponding to this variable."""

        if isinstance(variable, int):
            return variable
        else:
            return self.variables.index(variable)

    def marginalize(self, binning_i, reduction_function=np.sum):
        """Marginalize out the given binnings and return a new RectilinearBinning.

        Parameters
        ----------

        binning_i : iterable of int/str
            Iterable of index/variable of binning to be marginalized.
        reduction_function : function
            Use this function to marginalize out the entries over the specified variables.
            Must support the `axis` keyword argument.

        """

        try:
            len(binning_i)
        except TypeError:
            binning_i = [binning_i]

        binning_i = [self.get_variable_index(i) for i in binning_i]
        variables = [self.variables[i] for i in binning_i]

        # Create new binning
        new_variables = list(self.variables)
        new_bin_edges = list(deepcopy(self.bin_edges))
        for i in sorted(binning_i, reverse=True):
            del new_bin_edges[i]
            del new_variables[i]
        new_binning = RectilinearBinning(variables=new_variables, bin_edges=new_bin_edges, include_upper=self.include_upper)

        # Copy and project values, from binning without subbinnings

        axes = tuple(sorted(binning_i))
        temp_binning = self.marginalize_subbinnings()
        new_values = reduction_function(temp_binning.get_values_as_ndarray(shape=temp_binning.bins_shape), axis=axes)
        new_entries = reduction_function(temp_binning.get_entries_as_ndarray(shape=temp_binning.bins_shape), axis=axes)
        new_sumw2 = reduction_function(temp_binning.get_sumw2_as_ndarray(shape=temp_binning.bins_shape), axis=axes)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def project(self, binning_i, **kwargs):
        """Project the binning onto the given binnings and return a new RectilinearBinning.

        The order of the original binnings is preserved. If a single ``int`` is
        provided, the returned Binning is of the same type as the respective
        binning.

        Parameters
        ----------

        binning_i : iterable of int/str, or int/str
            Iterable of index of binning to be marginalized.
        **kwargs : optional
            Additional keyword arguments are passed on to :meth:`marginalize`.

        Returns
        -------

        RectilinearBinning or type(self.binnings[binning_i])

        """
        try:
            i = list(binning_i)
        except TypeError:
            i = [binning_i]

        i = [self.get_variable_index(var) for var in binning_i]

        # Which variables to remove
        rm_i = list(range(len(self.binnings)))
        list(map(rm_i.remove, i))

        ret = self.marginalize(rm_i, **kwargs)

        if isinstance(binning_i, int) or  isinstance(binning_i, str):
            return ret._unpack()
        else:
            return ret

    def slice(self, slices):
        """Return a new RectilinearBinning containing the given variable slice

        Parameters
        ----------

        slices : dict of (variable, (start, stop[, step]))
            The start and stop positions for the slices of all variables that
            should be sliced.

        Returns
        -------

        sliced_binning : RectilinearBinning
            A :class:`RectilinearBinning` consisting of the specified slices.

        Notes
        -----

        This will remove any ``subbinnings`` the binning might have.

        """

        # Create new binning edges and slice tuple
        new_bin_edges = list(deepcopy(self.bin_edges))
        all_slices = []
        for i, (var, edges) in enumerate(zip(self.variables, self.bin_edges)):
            if var in slices:
                bin_slice = slice(*slices[var])
                lower = edges[:-1][bin_slice]
                upper = edges[1:][bin_slice]
                new_bin_edges[i] = list(lower) + [upper[-1]]
                all_slices.append(bin_slice)
            else:
                # This variable does not have to be sliced
                all_slices.append(slice(None))
        all_slices = tuple(all_slices)

        # Create new binning
        new_binning = RectilinearBinning(variables=self.variables, bin_edges=new_bin_edges, include_upper=self.include_upper)

        # Copy and slice values
        temp_binning = self.marginalize_subbinnings()
        new_values = temp_binning.get_values_as_ndarray(shape=temp_binning.bins_shape)[all_slices]
        new_entries = temp_binning.get_entries_as_ndarray(shape=temp_binning.bins_shape)[all_slices]
        new_sumw2 = temp_binning.get_sumw2_as_ndarray(shape=temp_binning.bins_shape)[all_slices]

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def remove_bin_edges(self, bin_edge_indices):
        """Return a new RectilinearBinning with the given bin edges removed.

        The values of the bins adjacent to the removed bin edges will be
        summed up in the resulting larger bin. Please note that bin values
        are lost if the first or last binedge of a variable are removed.

        Parameters
        ----------

        bin_edge_indices : dict of (variable: list of int)
            Lists specifying the bin edge indices that should be removed.

        Notes
        -----

        This will remove any ``subbinnings`` the rectilinear binning might have.

        """

        # Create new binning
        new_bin_edges = []
        for var, edg in zip(self.variables, self.bin_edges):
            new_edg = list(edg)
            if var in bin_edge_indices:
                for i in sorted(bin_edge_indices[var], reverse=True):
                    del new_edg[i]
            new_bin_edges.append(new_edg)

        new_binning = RectilinearBinning(variables=self.variables, bin_edges=new_bin_edges, include_upper=self.include_upper)

        # Copy and slice values
        temp_binning = self.marginalize_subbinnings()
        new_values = temp_binning.get_values_as_ndarray(shape=temp_binning.bins_shape)
        new_entries = temp_binning.get_entries_as_ndarray(shape=temp_binning.bins_shape)
        new_sumw2 = temp_binning.get_sumw2_as_ndarray(shape=temp_binning.bins_shape)

        for j, var in enumerate(self.variables):
            if var in bin_edge_indices:
                for i in sorted(bin_edge_indices[var], reverse=True):
                    if i > 0 and i < new_values.shape[j]:
                        lower_tuple = (slice(None),)*j + (i-1,) + (Ellipsis,)
                        upper_tuple = (slice(None),)*j + (i,) + (Ellipsis,)
                        new_values[lower_tuple] += new_values[upper_tuple]
                        new_entries[lower_tuple] += new_entries[upper_tuple]
                        new_sumw2[lower_tuple] += new_sumw2[upper_tuple]
                    if i < new_values.shape[j]:
                        new_values = np.delete(new_values, i, axis=j)
                        new_entries = np.delete(new_entries, i, axis=j)
                        new_sumw2 = np.delete(new_sumw2, i, axis=j)
                    else:
                        new_values = np.delete(new_values, -1, axis=j)
                        new_entries = np.delete(new_entries, -1, axis=j)
                        new_sumw2 = np.delete(new_sumw2, -1, axis=j)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def __eq__(self, other):
        """RectilinearBinnings are equal if the bin edges and variables match."""
        return (type(self) == type(other)
            and self.variables == other.variables
            and all(np.array_equal(self.bin_edges[i], other.bin_edges[i]) for i in range(len(self.variables)))
            and self.include_upper == other.include_upper
            and self.subbinnings == other.subbinnings)

    def _get_clone_kwargs(self, **kwargs):
        """Get the necessary arguments to clone this object."""

        args = {
            'variables': list(self.variables),
            'bin_edges': [ edg.tolist() for edg in self.bin_edges ],
            'include_upper': self.include_upper,
            }
        args.update(Binning._get_clone_kwargs(self, bins=None, **kwargs))
        del args['bins']
        return args

    yaml_tag = u'!RectilinearBinning'
