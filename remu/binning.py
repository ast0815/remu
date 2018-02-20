from __future__ import division
from six.moves import map, zip
from copy import copy, deepcopy
import ruamel.yaml as yaml
import re
import numpy as np
from numpy.lib.recfunctions import rename_fields
import csv
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tempfile import TemporaryFile

class PhaseSpace(object):
    """A PhaseSpace defines the possible combinations of variables that characterize an event.

    It can be seen as the carthesian product of those variables.

        >>> ps = PhaseSpace(variables=['a', 'b', 'c'])
        >>> print ps
        ('a' X 'c' X 'b')

    You can check whether a variable is part of a phase space:

        >>> 'a' in ps
        True

    Phase spaces can be compared to one another.

    Check whether two phase spaces are identical:

        ('a' X 'b') == ('a' X 'b')
        ('a' X 'b') != ('a' X 'c')

    Check whether one phase space is a sub-space of the other:

        ('a' X 'b' X 'c') > ('a' X 'b')
        ('a' X 'c') < ('a' X 'b' X 'c')

    """

    def __init__(self, variables):
        """Create a PhaseSpace object.

        Arguments
        ---------

        variables: The set of variables that define the phase space.
        """

        self.variables = set(variables)

    def __contains__(self, var):
        return var in self.variables

    def __eq__(self, phasespace):
        try:
            return self.variables == phasespace.variables
        except AttributeError:
            return False

    def __ne__(self, phasespace):
        try:
            return not self.variables == phasespace.variables
        except AttributeError:
            return False

    def __le__(self, phasespace):
        try:
            return self.variables <= phasespace.variables
        except AttributeError:
            return False

    def __ge__(self, phasespace):
        try:
            return self.variables >= phasespace.variables
        except AttributeError:
            return False

    def __lt__(self, phasespace):
        try:
            return (self.variables <= phasespace.variables) and not (self.variables == phasespace.variables)
        except AttributeError:
            return False

    def __gt__(self, phasespace):
        try:
            return (self.variables >= phasespace.variables) and not (self.variables == phasespace.variables)
        except AttributeError:
            return False

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
        return '%s(variables=%s)'%(self.__class__.__name__, repr(self.variables))

    @staticmethod
    def _yaml_representer(dumper, obj):
        """Represent PhaseSpaces in a YAML file."""
        return dumper.represent_sequence('!PhaseSpace', list(obj.variables))

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct PhaseSpaces from YAML files."""
        seq = loader.construct_sequence(node)
        return PhaseSpace(variables=seq)

yaml.add_representer(PhaseSpace, PhaseSpace._yaml_representer)
yaml.add_constructor(u'!PhaseSpace', PhaseSpace._yaml_constructor)

class Bin(object):
    """A Bin is container for a value that is defined on a subset of an n-dimensional phase space."""

    def __init__(self, **kwargs):
        """Create basic bin.

        kwargs
        ------

        phasespace : The phase space the Bin resides in.
        value : The initialization value of the bin. Default: 0.0
        entries : The initialization value of the number of entries. Default: 0
        sumw2 : The initialization value of the sum of squared weights. Default: value**2
        value_array : A slice of a numpy array, where the value of the bin will be stored.
                      Default: None
        entries_array : A slice of a numpy array, where the number entries will be stored.
                        Default: None
        sumw2_array : A slice of a numpy array, where the squared weights will be stored.
                      Default: None
        """

        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise TypeError("Undefined phase space!")

        self._value_array = kwargs.pop('value_array', None)
        if self._value_array is None:
            self._value_array = np.array([kwargs.pop('value', 0.)], dtype=float)

        self._entries_array = kwargs.pop('entries_array', None)
        if self._entries_array is None:
            self._entries_array = np.array([kwargs.pop('entries', 0)], dtype=int)

        self._sumw2_array = kwargs.pop('sumw2_array', None)
        if self._sumw2_array is None:
            self._sumw2_array = np.array([kwargs.pop('sumw2', self.value**2)], dtype=float)

        if len(kwargs) > 0:
            raise TypeError("Unknown kwargs: %s"%(kwargs,))

    @property
    def value(self):
        return self._value_array[0]

    @value.setter
    def value(self, v):
        self._value_array[0] = v

    @property
    def entries(self):
        return self._entries_array[0]

    @entries.setter
    def entries(self, v):
        self._entries_array[0] = v

    @property
    def sumw2(self):
        return self._sumw2_array[0]

    @sumw2.setter
    def sumw2(self, v):
        self._sumw2_array[0] = v

    def event_in_bin(self, event):
        """Return True if the variable combination falls within the bin."""

        raise NotImplementedError("This method must be defined in an inheriting class.")

    def fill(self, weight=1.):
        """Add the weight(s) to the bin."""

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

    def __contains__(self, event):
        """Return True if the event falls within the bin."""
        return self.event_in_bin(event)

    def __eq__(self, other):
        """Bins are equal if they are defined on the same phase space."""
        try:
            return self.phasespace == other.phasespace
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        ret = deepcopy(self)
        ret.value = self.value + other.value
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

    def __str__(self):
        return "Bin on %s: %s"%(self.phasespace, self.value)

    def __repr__(self):
        return '%s(phasespace=%s, value=%s)'%(self.__class__.__name__, repr(self.phasespace), repr(self.value))

    @staticmethod
    def _yaml_representer(dumper, obj):
        """Represent Bin in a YAML file."""
        dic = {
                'phasespace': obj.phasespace,
                'value': float(obj.value),
                'entries': int(obj.entries),
              }
        return dumper.represent_mapping('!Bin', dic)

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct Bin from YAML files."""
        dic = loader.construct_mapping(node)
        return Bin(**dic)

yaml.add_representer(Bin, Bin._yaml_representer)
yaml.add_constructor(u'!Bin', Bin._yaml_constructor)

class RectangularBin(Bin):
    """A bin defined by min and max values in all variables."""

    def __init__(self, **kwargs):
        """Initialize a rectangular bin with bin edges.

        kwargs
        ------

        edges: A dict of {'varname': (lower_edge, upper_edge)}
        include_lower: Does the bin include the lower edges? Default: True
        include_upper: Does the bin include the upper edges? Default: False
        """

        self.include_lower = kwargs.pop('include_lower', True)
        self.include_upper = kwargs.pop('include_upper', False)
        self.edges = kwargs.pop('edges', None)
        if self.edges is None:
            raise TypeError("Edges are not defined")

        # Create PhaseSpace from edges if necessary
        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            kwargs['phasespace'] = PhaseSpace(self.edges.keys())

        # Handle default bin initialization
        Bin.__init__(self, **kwargs)

        # Check that all edges are valid tuples
        for var in self.edges:
            if var not in self.phasespace:
                raise ValueError("Variable not part of PhaseSpace: %s"%(var,))
            mi, ma = self.edges[var]

            if ma < mi:
                raise ValueError("Upper edge is smaller than lower edge for variable %s."%(var,))

            self.edges[var] = (mi, ma)

    def event_in_bin(self, event):
        """Check whether an event is within all bin edges."""

        inside = True

        for var in self.edges:
            mi, ma = self.edges[var]
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
        """Return the bin center coordinates."""
        center = {}
        for key, (mi, ma) in self.edges.items():
            center[key] = (float(mi) + float(ma)) / 2.
        return center

    def __eq__(self, other):
        """RectangularBins are equal if they have the same edges."""
        try:
            return (Bin.__eq__(self, other)
                and self.edges == other.edges
                and self.include_lower == other.include_lower
                and self.include_upper == other.include_upper)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        edgerep = repr(self.edges)
        return "RectBin %s; inclow=%s; incup=%s: %s"%(edgerep, repr(self.include_lower), repr(self.include_upper), repr(self.value))

    @staticmethod
    def _yaml_representer(dumper, obj):
        """Represent RectangularBin in a YAML file."""
        dic = {
                'phasespace': obj.phasespace,
                'value': float(obj.value),
                'entries': int(obj.entries),
                'include_upper': obj.include_upper,
                'include_lower': obj.include_lower,
              }
        edges = copy(obj.edges)
        for var in edges:
            # Convert bin edges to lists for prettier YAML
            edges[var] = list(edges[var])
        dic['edges'] = edges
        return dumper.represent_mapping('!RecBin', dic)

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct RectangularBin from YAML files."""
        dic = loader.construct_mapping(node, deep=True)
        edges = dic['edges']
        for var in edges:
            # Convert lists back to tuples
            edges[var] = tuple(edges[var])
        dic['edges'] = edges
        return RectangularBin(**dic)

yaml.add_representer(RectangularBin, RectangularBin._yaml_representer)
yaml.add_constructor(u'!RecBin', RectangularBin._yaml_constructor)

class Binning(object):
    """A Binning is a set of Bins.

    It translates variable values to bin numbers and vice versa.
    """

    def __init__(self, **kwargs):
        """Create basic Binning.

        kwargs
        ------

        phasespace : The PhaseSpace the Binning resides in.
        bins : The list of disjoint bins on that PhaseSpace.
        """

        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise TypeError("Undefined phase space!")

        self.bins = kwargs.pop('bins', None)
        if self.bins is None:
            raise TypeError("Undefined bins!")

        if len(kwargs) > 0:
            raise TypeError("Unknown kwargs: %s"%(kwargs,))

    def get_event_bin_number(self, event):
        """Returns the bin number of the given event.

        Returns `None` if the event does not belong to any bin.

        This is a dumb method that just loops over all bins until it finds a fitting one.
        It should be replaced with something smarter for more specifig binning classes.
        """

        for i in range(len(self.bins)):
            if event in self.bins[i]:
                return i

        return None

    def get_event_bin(self, event):
        """Return the bin of the event.

        Returns `None` if the event does not fit in any bin.
        """

        nr = self.get_event_bin_number(event)
        if nr is not None:
            return self.bins[nr]
        else:
            return None

    def fill(self, event, weight=1, raise_error=False, rename={}):
        """Fill the events into their respective bins.

        Arguments
        ---------

        event: The event(s) to be filled into the binning.
               Can be either a single event or an iterable of multiple events.
        weight: The weight of the event(s).
                Can be either a scalar which is then used for all events
                or an iterable of weights for the single events.
                Default: 1
        raise_error: Raise a ValueError if an event is not in the binning.
                     Otherwise ignore the event.
                     Default: False
        rename: Dict for translating event variable names to binning variable names.
                Default: `{}`, i.e. no translation
        """

        try:
            # Numpy array?
            event = rename_fields(event, rename)
        except AttributeError:
            # Dict?
            for name in rename:
                event[rename[name]] = event[name]

        try:
            if len(event) == 0:
                # Empty iterable? Stop right here
                return
        except TypeError:
            # Not an iterable
            pass

        try:
            # Try to get bin numbers from structured numpy array
            ibins = list(map(self.get_event_bin_number, np.nditer(event)))
        except TypeError:
            # Seems like this is not a structured numpy array
            try:
                # Try to get bin numbers from any iterable of events
                ibins = list(map(self.get_event_bin_number, event))
            except TypeError:
                # We probably only have a single event
                ibins = [self.get_event_bin_number(event)]

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
                try:
                    # Try fill_index method. Should be faster if it exists.
                    self.bins.fill_index(i, w)
                except AttributeError:
                    self.bins[i].fill(w)

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

        This saves time, because the numpy array only has to be generated once.
        Other than the list of binnings to be filled, the (keyword) arguments
        are identical to the ones used by the instance method
        `fill_from_csv_file`.
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

        Arguments
        ---------

        filename : The csv file with the data. Can be a list of filenames.
        weightfield : Optional. The column with the event weights.
        weight : Optional. A single weight that will be applied to all events in the file.
                 Can be an iterable with one weight for each file if `filename` is a list.
        rename : Optional. A dict with columns that should be renamed before filling.

                    {'csv_name': 'binning_name'}

        cut_function : Optional. A function that modifies the loaded data before filling into the binning.

                            cut_function(data) = data[ data['binning_name'] > some_threshold ]

                       This is done *after* the optional renaming.
        buffer_csv_files : Optional. Save the results of loading CSV files in temporary files
                           that can be recovered if the same CSV file is loaded again. This
                           speeds up filling multiple Binnings with the same CSV-files considerably!
                           Default: False
        chunksize : Optional. Load csv file in chunks of <chunksize> rows. This reduces the memory
                    footprint of the loading operation, but can slow it down.
                    Default: 10000

        The file must be formated like this:

            first_varname,second_varname,...
            <first_value>,<second_value>,...
            <first_value>,<second_value>,...
            <first_value>,<second_value>,...
            ...

        For example:

            x,y,z
            1.0,2.1,3.2
            4.1,2.0,2.9
            3,2,1

        All values are interpreted as floats. If `weightfield` is given, that
        field will be used as weigts for the event. Other keyword arguments
        are passed on to the Binning's `fill` method. If filename is a list,
        all elemets are handled recursively.

        """

        # Actual filling is handled by static method
        Binning.fill_multiple_from_csv_file([self], *args, **kwargs)

    def reset(self, value=0., entries=0, sumw2=0.):
        """Reset all bin values."""
        for b in self.bins:
            b.value=value
            b.entries=entries
            b.sumw2=sumw2

    def get_values_as_ndarray(self, shape=None, indices=None):
        """Return the bin values as nd array.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        l = len(self.bins)
        if indices is None:
            indices = list(range(l))
        l = len(indices)

        if shape is None:
            shape = l

        arr = np.ndarray(shape=l, order='C', dtype=float) # Row-major 'C-style' array. Last variable indices vary the fastest.

        for i, ind in enumerate(indices):
            arr[i] = self.bins[ind].value

        arr.shape = shape

        return arr

    def get_entries_as_ndarray(self, shape=None, indices=None):
        """Return the number of bin entries as nd array.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        l = len(self.bins)
        if indices is None:
            indices = list(range(l))
        l = len(indices)

        if shape is None:
            shape = l

        arr = np.ndarray(shape=l, order='C', dtype=int) # Row-major 'C-style' array. Last variable indices vary the fastest.

        for i, ind in enumerate(indices):
            arr[i] = self.bins[ind].entries

        arr.shape = shape

        return arr

    def get_sumw2_as_ndarray(self, shape=None, indices=None):
        """Return the sums of squared weights as nd array.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        l = len(self.bins)
        if indices is None:
            indices = list(range(l))
        l = len(indices)

        if shape is None:
            shape = l

        arr = np.ndarray(shape=l, order='C', dtype=float) # Row-major 'C-style' array. Last variable indices vary the fastest.

        for i, ind in enumerate(indices):
            arr[i] = self.bins[ind].sumw2

        arr.shape = shape

        return arr

    def set_values_from_ndarray(self, arr):
        """Set the bin values to the values of the ndarray."""

        l = len(self.bins)
        for i in range(l):
            self.bins[i].value = arr.flat[i]

    def set_entries_from_ndarray(self, arr):
        """Set the number of bin entries to the values of the ndarray."""

        l = len(self.bins)
        for i in range(l):
            self.bins[i].entries = arr.flat[i]

    def set_sumw2_from_ndarray(self, arr):
        """Set the sums of squared weights to the values of the ndarray."""

        l = len(self.bins)
        for i in range(l):
            self.bins[i].sumw2 = arr.flat[i]

    def event_in_binning(self, event):
        """Check whether an event fits into any of the bins."""

        i = self.get_event_bin_number(event)
        if i is None:
            return False
        else:
            return True

    def __contains__(self, event):
        return self.event_in_binning(event)

    def __eq__(self, other):
        """Binnings are equal if all bins and the phase space are equal."""
        try:
            return self.bins == other.bins and self.phasespace == other.phasespace
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def _yaml_representer(dumper, obj):
        """Represent Binning in a YAML file."""
        return dumper.represent_sequence('!Binning', obj.bins)

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct Binning from YAML files."""
        bins = loader.construct_sequence(node)
        return Binning(bins=bins, phasespace=bins[0].phasespace)

yaml.add_representer(Binning, Binning._yaml_representer)
yaml.add_constructor(u'!Binning', Binning._yaml_constructor)

class _RecBinProxy(object):
    """Indexable class that returns bins as proxy for numpy arrays."""

    def __init__(self, binning):
        """Initialise the proxy and create the numpy arrays."""

        self.binning = binning
        self._value_array = np.zeros(binning._totbins, dtype=float)
        self._entries_array = np.zeros(binning._totbins, dtype=int)
        self._sumw2_array = np.zeros(binning._totbins, dtype=float)

    def fill_index(self, index, weight=1.):
        """Shortcut function to fill rectangular binnings faster."""

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

        self._value_array[index] += w
        self._entries_array[index] += n
        self._sumw2_array[index] += w2

    def __getitem__(self, index):
        """Dynamically build a RectangularBin when requested."""
        val_slice = self._value_array.reshape(-1, order='C')[index:index+1]
        ent_slice = self._entries_array.reshape(-1, order='C')[index:index+1]
        sumw2_slice = self._sumw2_array.reshape(-1, order='C')[index:index+1]
        tup = self.binning.get_bin_number_tuple(index)
        edges = dict( (v, (e[j], e[j+1])) for v,e,j in zip(self.binning.variables, self.binning._edges, tup) )
        rbin = RectangularBin(edges=edges, include_lower=not self.binning._include_upper, include_upper=self.binning._include_upper, phasespace=self.binning.phasespace, value_array=val_slice, entries_array=ent_slice, sumw2_array=sumw2_slice)
        return rbin

    def __len__(self):
        return self.binning._totbins

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        return self.binning == other.binning

class RectangularBinning(Binning):
    """Binning made exclusively out of RectangularBins"""

    def __init__(self, **kwargs):
        """Initialize RectangularBinning.

        kwargs
        ------
        binedges: Dictionary of bin edges for rectangular binning.
        include_upper: Make bins include upper edges instead of lower edges.
                       Default: False
        variables: List that determines the order of the variables.
                   Will be generated from binedges if not given.
        """

        self.binedges = kwargs.pop('binedges', None)
        if self.binedges is None:
            raise TypeError("Undefined bin edges!")
        self.binedges = dict((k, tuple(float(f) for f in v)) for k, v in self.binedges.items())

        self.variables = kwargs.pop('variables', None)
        if self.variables is None:
            self.variables = self.binedges.keys()
        self.variables = tuple(self.variables)
        self.nbins = tuple(len(self.binedges[v])-1 for v in self.variables)
        self._stepsize = [1]
        # Calculate the step size (or stride) for each variable index.
        # We use a row-major ordering (C-style).
        # The index of the later variables varies faster than the ones before:
        #
        #   (0,0) <-> 0
        #   (0,1) <-> 1
        #   (0,2) <-> 2
        #   (1,0) <-> 3
        #   ...
        #
        # _stepsize is 1 longer than variables and nbins!
        for n in reversed(self.nbins):
            self._stepsize.insert(0, self._stepsize[0] * n)
        self._stepsize = tuple(self._stepsize)
        self._totbins = self._stepsize[0]
        self._edges = tuple(self.binedges[v] for v in self.variables)

        self._include_upper = kwargs.pop('include_upper', False)

        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            # Create phasespace from binedges
            phasespace = PhaseSpace(self.variables)
            kwargs['phasespace'] = phasespace

        bins = kwargs.pop('bins', None)
        if bins is not None:
            raise TypeError("Cannot define bins of RectangularBinning! Define binedges instead.")
        else:
            # Create bin proxy
            bins = _RecBinProxy(self)
        kwargs['bins'] = bins

        Binning.__init__(self, **kwargs)

    def get_tuple_bin_number(self, i_var):
        """Translate a tuple of variable bin numbers to the linear bin number of the event.

        Turns this:

            (i_x, i_y, i_z)

        into this:

            i_bin

        The order of the indices in the tuple must conform to the order of `self.variables`.
        The the bins are ordered row-major (C-style).
        """

        if None in i_var:
            return None

        i_bin = 0
        for i,s in zip(i_var, self._stepsize[1:]):
            i_bin += s*i

        return i_bin

    def get_bin_number_tuple(self, i_bin):
        """Translate the linear bin number of the event to a tuple of single variable bin numbers.

        Turns this:

            i_bin

        into this:

            (i_x, i_y, i_z)

        The order of the indices in the tuple conforms to the order of `self.variables`.
        The bins are ordered row-major (C-style).
        """

        if i_bin is None or i_bin < 0 or i_bin >= self._totbins:
            return tuple([None]*len(self.variables))

        i_var = tuple((i_bin % t) // s for t,s in zip(self._stepsize[:-1], self._stepsize[1:]))
        return i_var

    def get_event_tuple(self, event):
        """Get the variable index tuple for a given event."""

        i_var = []
        for var in self.variables:
            edges = self.binedges[var]
            i = int(np.digitize(event[var], edges, right=self._include_upper))
            if i > 0 and i < len(edges):
                i_var.append(i-1)
            else:
                i_var.append(None)

        return i_var

    def get_event_bin_number(self, event):
        """Get the bin number for a given event."""

        tup = self.get_event_tuple(event)
        return self.get_tuple_bin_number(tup)

    def cartesian_product(self, other):
        """Create the Cartesian product of two rectangular binnings.

        The two binnings must not share any variables.
        The two binnings must have the same value of `include_upper`.
        The resulting binning is in the the variables of both binnings with the respective edges.
        """

        if self._include_upper != other._include_upper:
            raise ValueError("Both RectangularBinnings must have the same `include_upper`.")

        SA = set(self.variables)
        SB = set(other.variables)
        if len(SA & SB) > 0:
            raise ValueError("Both RectangularBinnings must not share any variables.")

        phasespace = self.phasespace * other.phasespace
        variables = list(self.variables) + list(other.variables)
        binedges = self.binedges.copy()
        binedges.update(other.binedges)

        return RectangularBinning(phasespace=phasespace, variables=variables, binedges=binedges, include_upper=self._include_upper)

    def marginalize(self, variables, reduction_function=np.sum):
        """Marginalize out the given variables and return a new RectangularBinning.

        Arguments
        ---------

        variables : Iterable of variable names to be marginalized out.
        reduction_function : Use this function to marginalize out the entries
                             over the specified variables. Must support the
                             `axis` keyword argument.
                             Default: numpy.sum

        """

        # Create new binning
        new_variables = list(self.variables)
        list(map(new_variables.remove, variables))
        new_binedges = deepcopy(self.binedges)
        list(map(new_binedges.pop, variables))
        new_includeupper = self._include_upper

        new_binning = RectangularBinning(variables=new_variables, binedges=new_binedges, include_upper=new_includeupper)

        # Find the axis numbers of the variables
        axes = tuple([self.variables.index(v) for v in variables])

        # Copy and project values
        new_values = reduction_function(self.get_values_as_ndarray(shape=self.nbins), axis=axes)
        new_entries = reduction_function(self.get_entries_as_ndarray(shape=self.nbins), axis=axes)
        new_sumw2 = reduction_function(self.get_sumw2_as_ndarray(shape=self.nbins), axis=axes)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def project(self, variables, **kwargs):
        """Project the binning onto the given variables and return a new RectangularBinning.

        The variable order of the original binning is preserved.

        Arguments
        ---------

        variables : Iterable of variable names on which to project the binning.

        Additional keyword arguments are passed on to `marginalize`.

        """

        # Which variables to remove
        rm_variables = list(self.variables)
        list(map(rm_variables.remove, variables))

        return self.marginalize(rm_variables, **kwargs)

    def slice(self, variable_slices, return_indices=False):
        """Return a new RectangularBinning containing the given variable slices

        Arguments
        ---------

        variable_slices : A dictionary specifying the bin slices of each
                          variable. Binning variables that are not part of the
                          dictionary are kept as is.  E.g. if you want the
                          slice of bin 2 in `var_A` and bins 1 through to the
                          last in `var_C`:

                              variable_slices = { 'var_A': slice(2,3),
                                                'var_C': slice(1,None) }

                          Please note that strides other than 1 are *not*
                          supported.

        return_indices : If `True`, also return the indices of the new binning:

                             new_values = binning.get_values_as_ndarray(
                                            shape=binning.nbins)[indices]
        """

        # Create new binning
        new_binedges = deepcopy(self.binedges)
        for var, sl in variable_slices.items():
            lower = new_binedges[var][:-1][sl]
            upper = new_binedges[var][1:][sl]
            new_binedges[var] = list(lower) + [upper[-1]]

        new_binning = RectangularBinning(variables=self.variables, binedges=new_binedges, include_upper=self._include_upper)

        # Find the axis numbers of the variables
        index = tuple( variable_slices[v] if v in variable_slices else slice(None) for v in self.variables )

        # Copy and project values
        new_values = self.get_values_as_ndarray(shape=self.nbins)[index]
        new_entries = self.get_entries_as_ndarray(shape=self.nbins)[index]
        new_sumw2 = self.get_sumw2_as_ndarray(shape=self.nbins)[index]

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        if return_indices:
            return new_binning, index
        else:
            return new_binning

    def rebin(self, remove_binedges):
        """Return a new RectangularBinning with the given bin edges removed.

        Arguments
        ---------

        remove_binedges : A dictionary specifying the bin edge indeices of each
                          variable that should be removed. Binning variables that are not part of the
                          dictionary are kept as is.  E.g. if you want to remove
                          bin edge 2 in `var_A` and bin edges 3, 4 and 7 in `var_C`:

                              remove_binedges = { 'var_A': [2],
                                                  'var_B': [3, 4, 7] }

                          The values of the bins adjacent to the removed bin edges
                          will be summed up in the resulting larger bin.
                          Please note that bin values are lost if the first or last
                          binedge of a variable are removed.
        """

        # Create new binning
        new_binedges = deepcopy(self.binedges)
        for var, il in remove_binedges.items():
            for i in sorted(il, reverse=True):
                L = list(new_binedges[var])
                del L[i]
                new_binedges[var] = tuple(L)

        new_binning = RectangularBinning(variables=self.variables, binedges=new_binedges, include_upper=self._include_upper)

        # Copy and add values
        old_values  = self.get_values_as_ndarray(shape=self.nbins)
        old_entries = self.get_entries_as_ndarray(shape=self.nbins)
        old_sumw2   = self.get_sumw2_as_ndarray(shape=self.nbins)
        new_values  = self.get_values_as_ndarray(shape=self.nbins)
        new_entries = self.get_entries_as_ndarray(shape=self.nbins)
        new_sumw2   = self.get_sumw2_as_ndarray(shape=self.nbins)
        for var, il in remove_binedges.items():
            ax = self.variables.index(var)
            for i in sorted(il, reverse=True):
                if i > 0 and i < self.nbins[ax]:
                    new_values[(slice(None),)*ax + (i-1, Ellipsis)] += new_values[(slice(None),)*ax + (i, Ellipsis)]
                    new_entries[(slice(None),)*ax + (i-1, Ellipsis)] += new_entries[(slice(None),)*ax + (i, Ellipsis)]
                    new_sumw2[(slice(None),)*ax + (i-1, Ellipsis)] += new_sumw2[(slice(None),)*ax + (i, Ellipsis)]
                if i < self.nbins[ax]:
                    new_values  = np.delete(new_values, i, ax)
                    new_entries  = np.delete(new_entries, i, ax)
                    new_sumw2  = np.delete(new_sumw2, i, ax)
                else:
                    new_values  = np.delete(new_values, i-1, ax)
                    new_entries  = np.delete(new_entries, i-1, ax)
                    new_sumw2  = np.delete(new_sumw2, i-1, ax)

        new_binning.set_values_from_ndarray(new_values)
        new_binning.set_entries_from_ndarray(new_entries)
        new_binning.set_sumw2_from_ndarray(new_sumw2)

        return new_binning

    def get_values_as_ndarray(self, shape=None, indices=None):
        """Return the bin values as ndarray.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        if indices is None:
            indices = slice(None, None, None)

        ret = np.array(self.bins._value_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_values_from_ndarray(self, arr):
        """Set the bin values from ndarray.

        Arguments
        ---------

        arr: Numpy ndarray containing the values.

        """

        self.bins._value_array.flat[:] = arr.flat

    def get_entries_as_ndarray(self, shape=None, indices=None):
        """Return the number of entries as ndarray.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        if indices is None:
            indices = slice(None, None, None)

        ret = np.array(self.bins._entries_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_entries_from_ndarray(self, arr):
        """Set the bin entries from ndarray.

        Arguments
        ---------

        arr: Numpy ndarray containing the values.

        """

        self.bins._entries_array.flat[:] = arr.flat

    def get_sumw2_as_ndarray(self, shape=None, indices=None):
        """Return the sums of squared weights as ndarray.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        indices: Only return the given bins.
                 Default: Return all bins.
        """

        if indices is None:
            indices = slice(None, None, None)

        ret = np.copy(self.bins._sumw2_array[indices])
        if shape is not None:
            ret = ret.reshape(shape, order='C')
        else:
            ret = ret.reshape((ret.size,), order='C')
        return ret

    def set_sumw2_from_ndarray(self, arr):
        """Set the bin sums of squared weights from ndarray.

        Arguments
        ---------

        arr: Numpy ndarray containing the values.

        """

        self.bins._sumw2_array.flat[:] = arr.flat

    def plot_ndarray(self, filename, arr, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None, reduction_function=np.sum, denominator=None, sqrt_errors=False, no_plot=False):
        """Plot a visual representation of an array containing the entries or values of the binning.

        Arguments
        ---------

        filename : The target filename of the plot.
                   If `None`, the plot fill not be saved to disk.
                   This is only useful with the `figax` option.
        arr : The array containing the data to be plotted.
        variables : `list`, list of variables to plot marginal histograms for.
                    `None`, plor marginal histograms for all variables.
                    `(list, list)`, plot 2D histograms of the cartesian product of the two variable lists.
                    `(None, None)`, plot 2D histograms of all possible variable combinations.
                    2D histograms where both variables are identical are plotted as 1D histograms.
                    Default: `None`
        divide : Divide the bin content by the bin size before plotting.
                 Default: True
        kwargs1d, kwargs2d : Additional keyword arguments for the 1D/2D histograms.
                             If the key `label` is present, a legend will be drawn.
        figax : Pair of figure and axes to be used for plotting.
                Can be used to plot multiple binnings on top of one another.
                Default: Create new figure and axes.
        reduction_function : Use this function to marginalize out variables.
                             Default: numpy.sum
        denominator : A second array can be provided as a denominator.
                      It is projected the same way `arr` is prior to dividing.
        sqrt_errors : Plot sqrt(n) error bars.
                      Default: `False`
        no_plot : Do not plot anything, just create the figure and axes.
                  Default: `False`

        Returns
        -------

        fig, ax : The figure and axis objects.

        """

        kw1d = {'drawstyle': 'steps-post'}
        if sqrt_errors:
            kw1d['linestyle'] = 'none'
        kw1d.update(kwargs1d)
        kw2d = {}
        kw2d.update(kwargs2d)

        if type(variables) is list:
            variables = (variables, None)
            sharex = False
        elif variables is None:
            variables = (self.variables, None)
            sharex = False
        elif variables == (None, None):
            variables = (self.variables, self.variables)
            sharex = 'col'
        elif type(variables[0]) is list and type(variables[1]) is list:
            sharex = 'col'

        if variables[1] is None:
            ny = len(variables[0])
            nx = 1
        else:
            ny = len(variables[0])
            nx = len(variables[1])

        if figax is None:
            fig, ax = plt.subplots(ny, nx, squeeze=False, figsize=(4*nx,4*ny), sharex=sharex)
        else:
            fig, ax = figax

        if not no_plot:
            temp_binning = deepcopy(self)
            temp_binning.set_values_from_ndarray(arr)
            if denominator is not None:
                denominator_binning = deepcopy(self)
                denominator_binning.set_values_from_ndarray(denominator)

            def make_finite(edges):
                ret = list(edges)
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
                return ret

            for i in range(ny):
                y_var = variables[0][i]
                y_edg = np.array(make_finite(self.binedges[y_var]))
                for j in range(nx):
                    if variables[1] is None:
                        x_var = y_var
                        x_edg = y_edg
                    else:
                        x_var = variables[1][j]
                        x_edg = np.array(make_finite(self.binedges[x_var]))

                    if y_var == x_var:
                        # 1D histogram

                        nn = temp_binning.project([y_var], reduction_function=reduction_function).get_values_as_ndarray()
                        if sqrt_errors:
                            yerr = np.sqrt(nn)
                        else:
                            yerr = None

                        if denominator is not None:
                            deno = denominator_binning.project([y_var], reduction_function=reduction_function).get_values_as_ndarray()
                            nn /= deno
                            if sqrt_errors:
                                yerr /= deno

                        if divide:
                            div = (y_edg[1:] - y_edg[:-1])
                            nn /= div
                            if sqrt_errors:
                                yerr /= div

                        nn = np.append(nn, nn[-1])
                        if sqrt_errors:
                            y_edg = (y_edg[:-1] + y_edg[1:]) / 2.
                            nn = nn[:-1]


                        ax[i][j].set_xlabel(y_var)

                        ax[i][j].errorbar(y_edg, nn, yerr=yerr, **kw1d)

                        # Mark open ended bins
                        if not np.isfinite(self.binedges[y_var][0]):
                            ax[i][j].axhline(nn[0], -0.1, 0.0, color='k', linestyle='dotted', clip_on=False)

                        if not np.isfinite(self.binedges[y_var][-1]):
                            ax[i][j].axhline(nn[-1], 1.0, 1.1, color='k', linestyle='dotted', clip_on=False)

                        if 'label' in kw1d:
                            ax[i][j].legend(loc='best', framealpha=0.5, prop={'size':10})

                    else:
                        # 2D histogram

                        tb = temp_binning.project([x_var, y_var], reduction_function=reduction_function)
                        arr = tb.get_values_as_ndarray(shape=tb.nbins)
                        if denominator is not None:
                            arr /= denominator_binning.project([x_var, y_var], reduction_function=reduction_function).get_values_as_ndarray(shape=tb.nbins)
                        if tb.variables[0] != y_var:
                            arr = arr.transpose()
                        arr = arr.flatten()

                        # Bin centres
                        x = np.convolve(x_edg, np.ones(2)/2, mode='valid')
                        y = np.convolve(y_edg, np.ones(2)/2, mode='valid')
                        xx = np.broadcast_to(x, (len(y),len(x))).flatten()
                        yy = np.repeat(y, len(x))

                        if divide:
                            # Bin areas
                            wx = np.diff(x_edg)
                            wy = np.diff(y_edg)
                            wxx = np.broadcast_to(wx, (len(wy),len(wx))).flatten()
                            wyy = np.repeat(wy, len(wx))
                            A = wxx * wyy

                            arr /= A

                        if i==len(variables[0])-1:
                            ax[i][j].set_xlabel(x_var)
                        if j==0:
                            ax[i][j].set_ylabel(y_var)

                        kw2d.update({'norm': LogNorm()})
                        ax[i][j].hist2d(xx, yy, weights=arr, bins=(x_edg, y_edg), **kw2d)

        fig.tight_layout()
        if filename is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                fig.savefig(filename)

        return fig, ax

    def plot_values(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        return self.plot_ndarray(filename, self.bins._value_array, variables, divide, kwargs1d, kwargs2d, figax, **kwargs)

    def plot_entries(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        return self.plot_ndarray(filename, self.bins._entries_array, variables, divide, kwargs1d, kwargs2d, figax, **kwargs)

    def plot_sumw2(self, filename, variables=None, divide=True, kwargs1d={}, kwargs2d={}, figax=None, **kwargs):
        return self.plot_ndarray(filename, self.bins._sumw2_array, variables, divide, kwargs1d, kwargs2d, figax, **kwargs)

    def __eq__(self, other):
        """Rectangular binnings are equal if the variables and edges match."""
        try:
            return (self.variables == other.variables
                    and self.binedges == other.binedges
                    and self._edges == other._edges
                    and self.nbins == other.nbins
                    and self._stepsize == other._stepsize
                    and self._totbins == other._totbins
                    and self._include_upper == other._include_upper
                   )

        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def _yaml_representer(dumper, obj):
        """Represent RectangularBinning in a YAML file."""
        dic = {}
        dic['include_upper'] = obj._include_upper
        dic['binedges'] = [ [var, list(edg)] for var, edg in zip(obj.variables, obj._edges) ]
        return dumper.represent_mapping('!RecBinning', dic)

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct RectangularBinning from YAML files."""
        dic = loader.construct_mapping(node, deep=True)
        binedges = dict(dic['binedges'])
        variables = [varedg[0] for varedg in dic['binedges']]
        return RectangularBinning(variables=variables,
                                  binedges=binedges,
                                  include_upper=dic['include_upper'])

yaml.add_representer(RectangularBinning, RectangularBinning._yaml_representer)
yaml.add_constructor(u'!RecBinning', RectangularBinning._yaml_constructor)
