from copy import copy, deepcopy
import ruamel.yaml as yaml
import re
import numpy as np
import csv

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
        value_array : A slice of a numpy array, where the value of the bin will be stored.
                      Default: None
        entries_array : A slice of a numpy array, where the number entries will be stored.
                        Default: None
        """

        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise ValueError("Undefined phase space!")

        self._value_array = kwargs.pop('value_array', None)
        if self._value_array is None:
            self._value_array = np.array([0.])

        self._entries_array = kwargs.pop('entries_array', None)
        if self._entries_array is None:
            self._entries_array = np.array([0])

        self.value = kwargs.pop('value', 0.)
        self.entries = kwargs.pop('entries', 0)

        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s"%(kwargs,))

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

    def event_in_bin(self, event):
        """Return True if the variable combination falls within the bin."""

        raise NotImplementedError("This method must be defined in an inheriting class.")

    def fill(self, weight=1.):
        """Add the weight(s) to the bin."""

        try:
            self.value += sum(weight)
            self.entries += len(weight)
        except TypeError:
            self.value += weight
            self.entries += 1

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
            raise ValueError("Edges are not defined")

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
            raise ValueError("Undefined phase space!")

        self.bins = kwargs.pop('bins', None)
        if self.bins is None:
            raise ValueError("Undefined bins!")
        else:
            # Make sure the bins are saved in a list
            self.bins = list(self.bins)

        # Check that all bins are defined on the given phase space
        for b in self.bins:
            if b.phasespace != self.phasespace:
                raise ValueError("Phase space of bin does not match phase space of binning!")

        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s"%(kwargs,))

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

    def fill(self, event, weight=1, raise_error=False):
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
        """

        try:
            # Try to get bin numbers from list of events
            ibins = map(self.get_event_bin_number, event)
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
                self.bins[i].fill(w)

    def fill_from_csv_file(self, filename, weightfield=None, **kwargs):
        """Fill the binning with events from a CSV file.

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

        All values are interpreted as floats.

        If `weightfield` is given, that field will be used as weigts for the event.

        Other keyword arguments are passed on to the Binning's `fill` method
        """

        with open(filename, 'r') as f:
            dr = csv.DictReader(f, delimiter=',', strict=True)
            for event in dr:
                for k in event:
                    # Parse the fields as floats
                    event[k] = float(event[k])

                if weightfield is None:
                    self.fill(event, **kwargs)
                else:
                    weight = event.pop(weightfield)
                    self.fill(event, weight=weight, **kwargs)

    def reset(self, value=0., entries=0):
        """Reset all bin values."""
        for b in self.bins:
            b.value=value
            b.entries=entries

    def get_values_as_ndarray(self, shape=None):
        """Return the bin values as nd array.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        """

        l = len(self.bins)

        if shape is None:
            shape = l

        arr = np.ndarray(shape=l, order='C') # Row-major 'C-style' array. Last variable indices vary the fastest.

        for i in range(l):
            arr[i] = self.bins[i].value

        arr.shape = shape

        return arr

    def get_entries_as_ndarray(self, shape=None):
        """Return the number of bin entries as nd array.

        Arguments
        ---------

        shape: Shape of the resulting array.
               Default: len(bins)
        """

        l = len(self.bins)

        if shape is None:
            shape = l

        arr = np.ndarray(shape=l, order='C') # Row-major 'C-style' array. Last variable indices vary the fastest.

        for i in range(l):
            arr[i] = self.bins[i].entries

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

        self._binedges = kwargs.pop('binedges', None)
        if self._binedges is None:
            raise ValueError("Undefined bin edges!")
        self._binedges = dict((k, tuple(v)) for k, v in self._binedges.items())

        self.variables = kwargs.pop('variables', None)
        if self.variables is None:
            self.variables = self._binedges.keys()
        self.variables = tuple(self.variables)
        self._nbins = tuple(len(self._binedges[v])-1 for v in self.variables)
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
        # _stepsize is 1 longer than variables and _nbins!
        for n in reversed(self._nbins):
            self._stepsize.insert(0, self._stepsize[0] * n)
        self._stepsize = tuple(self._stepsize)
        self._totbins = self._stepsize[0]
        self._edges = tuple(self._binedges[v] for v in self.variables)

        self._include_upper = kwargs.pop('include_upper', False)

        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            # Create phasespace from binedges
            phasespace = PhaseSpace(self.variables)
            kwargs['phasespace'] = phasespace

        bins = kwargs.pop('bins', None)
        if bins is not None:
            raise ValueError("Cannot define bins of RectangularBinning! Define binedges instead.")
        else:
            # Create bins from bin edges
            bins = []
            for i in range(self._totbins):
                tup = self.get_bin_number_tuple(i)
                edges = dict( (v, (e[j], e[j+1])) for v,e,j in zip(self.variables, self._edges, tup) )
                bins.append(RectangularBin(edges=edges, include_lower=not self._include_upper, include_upper=self._include_upper, phasespace=phasespace))
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
            edges = self._binedges[var]
            i = np.digitize(event[var], edges, right=self._include_upper)
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
        binedges = self._binedges.copy()
        binedges.update(other._binedges)

        return RectangularBinning(phasespace=phasespace, variables=variables, binedges=binedges)

    def __eq__(self, other):
        """Rectangular binnings are equal if they are equal Binnings and the variables and edges match."""
        try:
            return ( Binning.__eq__(self, other)
                    and self.variables == other.variables
                    and self._binedges == other._binedges
                    and self._edges == other._edges
                    and self._nbins == other._nbins
                    and self._stepsize == other._stepsize
                    and self._totbins == other._totbins
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
        dic['phasespace'] = obj.phasespace
        return dumper.represent_mapping('!RecBinning', dic)

    @staticmethod
    def _yaml_constructor(loader, node):
        """Reconstruct RectangularBinning from YAML files."""
        dic = loader.construct_mapping(node, deep=True)
        binedges = dict(dic['binedges'])
        variables = [varedg[0] for varedg in dic['binedges']]
        return RectangularBinning(phasespace=dic['phasespace'],
                                  variables=variables,
                                  binedges=binedges,
                                  include_upper=dic['include_upper'])

yaml.add_representer(RectangularBinning, RectangularBinning._yaml_representer)
yaml.add_constructor(u'!RecBinning', RectangularBinning._yaml_constructor)
