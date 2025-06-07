import os
import h5py
import copy
import fitsio
import numpy as np
import pixell.enmap

from . import utils
from . import io_utils


class Catalog:
    def __init__(self, cols=None, name=None, filename=None, size=0):
        r"""Represents a galaxy catalog, with one "row" per galaxy, and user-defined "columns" for RA, DEC, etc.

        A design decision: do we actually need a Catalog class? Or should we phase it out,
        in favor of ``numpy.recarray`` or ``astropy.Table``?

        Constructor args:
          - cols: dictionary (string col_name) -> (1-d numpy array).
            Optional, since you can add columns later with add_column().
          - name: catalog name (optional)
          - filename: filename, if catalog is stored on disk (optional)
          - size: number of galaxies in catalog (optional, can be inferred from array dimensions instead).

        Members:

          - ``self.size`` (integer): Number of galaxies in catalog.
          - ``self.col_names`` (list of strings): List of user-defined columns.
          - ``self.name`` (string or None): Catalog name (optional).
          - ``self.filename`` (string or None): Filename, if Catalog is stored on disk (optional).

        Additionally, for each column name (in ``self.col_names``), the Catalog contains 
        a member with the corresponding name, whose value is a 1-d array of length self.size.

        Column names are user-defined, but here are some frequently-occuring column names
        (note that I always use lower case):

          - ``self.ra_deg``: right ascension (degrees)
          - ``self.dec_deg``: declination (degrees)
          - ``self.z``: redshift
          - ``self.zerr``: redshift error (if photometric)
          - ``self.rmag``: r-band magnitude (and similarly for g-band, z-band, etc.)

        Here are some functions which return Catalogs::

          kszx.sdss.read_galaxies()
          kszx.sdss.read_randoms()
          kszx.desils_lrg.read_galaxies()
          kszx.desils_lrg.read_randoms()
          kszx.Catalog.from_h5()    # static method
        
        Here are some functions which make maps from catalogs::

          kszx.healpix_utils.map_from_catalog()   # catalog -> 2d healpix map
          kszx.pixell_utils.map_from_catalog()    # catalog -> 2d pixell maps
          kszx.grid_points()                      # catalog -> 3d map
        """

        self.size = size
        self.name = name
        self.col_names = []
        self.filename = filename

        if cols is not None:
            for (col_name, col_data) in cols.items():
                self.add_column(col_name, col_data)

    
    def add_column(self, col_name, col_data):        
        r""" Adds a new column to the catalog.

          - col_name (string)
          - col_data (1-d numpy array)
        """
        
        assert isinstance(col_name, str)
        assert len(col_name) > 0
        assert not col_name.startswith('_')
        assert col_name not in [ 'size', 'name', 'filename', 'col_names' ]
        assert col_name not in self.col_names

        col_data = np.asarray(col_data)
        assert col_data.ndim == 1

        if (len(self.col_names) > 0) or (self.size != 0):
            assert len(col_data) == self.size
        else:
            self.size = len(col_data)
        
        self.col_names.append(col_name)
        setattr(self, col_name, col_data)

    
    def get_column(self, col_name):
        """ Return the data in col_name. Useful to access the column with variable. """
        assert col_name in self.col_names
        return getattr(self, col_name)


    def remove_column(self, col_name):
        r"""Removes an existing column from the catalog."""
        
        assert isinstance(col_name, str)
        assert len(col_name) > 0
        assert not col_name.startswith('_')
        assert col_name not in [ 'size', 'name', 'filename', 'col_names' ]
        assert col_name in self.col_names

        self.col_names.remove(col_name)
        delattr(self, col_name)


    def apply_boolean_mask(self, mask, name=None, in_place=True):
        r"""Reduces the size of the Catalog, by applying a caller-specified boolean mask.
        
        This can be used to implement color cuts, redshift cuts, etc.

           - mask (1-d boolean array). Elements should be True for galaxies which are kept,
             False for galaxies which are discarded.
        
           - name (string or None). If specified, the mask fraction will be printed.

           - in_place (booelan). If False, the original Catalog is unmodified, and a
             new Catalog is returned.
        """
        
        if not in_place:
            self = self.shallow_copy()
                
        mask = np.asarray(mask)
        assert mask.dtype == bool
        if mask.shape != (self.size,):
            raise RuntimeError(f'Catalog.apply_boolean_mask(): expected {mask.shape=} to equal {(self.size,)=}')

        new_size = np.sum(mask)

        if name is not None:
            p = 100 * new_size / self.size
            print(f'{name}: {self.size} -> {new_size} [{p:.05}% retained]')
        
        for col_name in self.col_names:
            col = getattr(self, col_name)
            setattr(self, col_name, col[mask])
            assert getattr(self,col_name).shape == (new_size,)

        self.size = new_size
        return self


    def apply_redshift_cut(self, zmin, zmax, in_place=True):
        r"""Reduces the size of the catalog, by keeping objects in redshift range $z_{min} \le z \le z_{max}$.

        This is a special case of ``apply_boolean_mask()``.

           - zmin (float or None). If None, then no lower redshift limit is imposed.
           - zmax (float or None). If None, then no upper redshift limit is imposed.
           - in_place (booelan). If False, the original Catalog is unmodified, and a
             new Catalog is returned.
        """
        
        assert 'z' in self.col_names

        mask1 = (self.z >= zmin) if (zmin is not None) else np.ones(self.size)
        mask2 = (self.z <= zmax) if (zmax is not None) else np.ones(self.size)
        
        mask = np.logical_and(mask1, mask2)
        name = f'Redshift cut: {zmin=} {zmax=}'
        return self.apply_boolean_mask(mask, name=name, in_place=in_place)


    def get_xyz(self, cosmo, zcol_name='z'):
        r"""Returns shape (N,3) array containing galaxy locations in Cartesian coords (observer at origin). 

        The Catalog must define columns named ``ra_deg``, ``dec_deg``, and ``z``.
        The constructor arg is:

           - cosmo (:class:`~kszx.Cosmology`). Used to convert redshifts to distances.

        This function is super useful -- here are some use cases for the ``points`` array that it returns:
        
           - To "grid" the galaxy field, pass the ``points`` array to :func:`kszx.grid_points`.
           - To interpolate a real-space field to the galaxy locations, pass the ``points`` array 
             to :func:`kszx.interpolate_points`
           - To create a bounding box for the galaxy field, pass the ``points`` array to
             the :class:`~kszx.BoundingBox` constructor.
        
        This function is roughly equivalent to::

           # Compute radial distaance from 'z' column of catalog
           chi = kszx.Cosmology.chi(self.z)
        
           # Compute Cartesian coords from 'ra_deg' and 'dec_deg' cols of catalog.
           points = kszx.utils.ra_dec_to_xyz(self.ra_deg, self.dec_deg, r=chi)
           return points
        """

        assert 'ra_deg' in self.col_names
        assert 'dec_deg' in self.col_names
        assert zcol_name in self.col_names

        chi = cosmo.chi(z = getattr(self,zcol_name))
        xyz = utils.ra_dec_to_xyz(self.ra_deg, self.dec_deg, r=chi)
        return xyz
    

    def generate_batches(self, batchsize, verbose=True):
        r"""Splits catalog into subcatalogs no larger than 'batchsize'.

        If ``batchsize`` is None, the catalog will be "split" into a single subcatalog.

        If ``verbose`` is True, and the catalog is split into more than one batch,
        then a progress indicator will be shown.
        """

        nbatches = 1

        if batchsize is not None:
            assert batchsize > 0
            nbatches = (self.size + batchsize - 1) // batchsize

        if nbatches == 1:
            yield self
            return
        
        for b in range(nbatches):
            s = f'subcatalog {b}/{nbatches}'
            if verbose:
                print('    ', s)  # progress indicator
            if self.name is not None:
                s = f'{self.name} [{s}]'

            i = (b * self.size) // nbatches
            j = ((b+1) * self.size) // nbatches
            assert i < j

            # Subcatalogs are constructed by slicing, not "deep copy".
            yield Catalog(
                cols = { k: getattr(self,k)[i:j] for k in self.col_names },
                name = s,
                size = j-i
            )

            
    def shallow_copy(self):
        return Catalog(
            cols = { s: getattr(self,s) for s in self.col_names },
            name = self.name,
            filename = self.filename,
            size = self.size
        )


    @staticmethod
    def concatenate(catalog_list, name=None, destructive=False):
        r"""Returns a new Catalog, obtained by concatenating (merging) multiple Catalogs.

           - catalog_list (sequence of Catalogs).
           - name (string or None). Name of new Catalog (optional).
           - destructive (boolean). If True, then the input Catalogs will be destroyed to save memory
        """
        
        assert len(catalog_list) > 0
        assert all(isinstance(x, Catalog) for x in catalog_list)

        if any(set(g.col_names) != set(catalog_list[0].col_names) for g in catalog_list):
            raise RuntimeError('Catalog.concatenate(): all Catalogs must have the same column names')

        # copy.copy() is necessary if destructive=True
        col_names = copy.copy(catalog_list[0].col_names)
        cols = { }

        for k in col_names:
            cols[k] = np.concatenate([getattr(g,k) for g in catalog_list])
            if destructive:
                for g in catalog_list:
                    g.remove_column(k)
            
        return Catalog(cols=cols, name=name)


    def absorb_columns(self, src_catalog, destructive=False):
        """'Absorbs' all columns of 'src_catalog' into an existing Catalog (equivalent to calling add_column() multiple times).

        If ``destructive=True``, then all columns will be removed from ``src_catalog`` (saving some memory)."""
        
        if (self.size > 0) and (src_catalog.size > 0) and (self.size != src_catalog.size):
            raise RuntimeError('Catalog.absorb_columns(): source and destination Catalogs must have the same size')
        
        for col_name in copy.copy(src_catalog.col_names):
            col_data = getattr(src_catalog, col_name)
            if not destructive:
                col_data = np.copy(col_data)
                
            self.add_column(col_name, col_data)
            
            if destructive:
                src_catalog.remove_column(col_name)


    def make_random_subcatalog(self, new_size, name=None):
        """Returns a new Catalog, obtained by choosing a random subset (with size ``new_size``) of existing Catalog.

        This method can be used to reduce the size of a random catalog. For example::

           # Random catalong with 100x the galaxy density
           rcat_large = kszx.sdss.read_randoms('CMASS_North')                  

           # Random catalong with 10x the galaxy density
           rcat_small = rcat_large.make_random_subcatalog(rcat_large.size/10)
        """

        print(f'make_random_subcatalog: {self.name}: {self.size} -> {new_size}')

        new_size = int(new_size)
        assert new_size > 0
        assert new_size <= self.size
        
        if name is None:
            name = f'subcatalog({new_size=})'

        # This seems to be the fastest way to select 'new_size' random indices in [0,self.size).
        isrc = np.random.permutation(self.size)[:new_size]
        
        cols = { k: getattr(self,k)[isrc] for k in self.col_names }
        return Catalog(cols=cols, name=name, size=new_size)

    
    @staticmethod
    def from_h5(filename):
        r"""Reads FITS file in HDF5 format (written by ``catalog.write_h5()``), and returns a Catalog object."""
        
        print(f'Reading {filename}\n', end='')
        
        with h5py.File(filename, 'r') as f:
            catalog = Catalog(
                cols = { k: np.asarray(v) for k,v in f.items() },
                name = f.attrs.get('name', default=None),
                filename = filename,
                size = f.attrs['size']
            )

        catalog._announce_file_read()
        return catalog


    def write_h5(self, filename):
        r"""Writes a Catalog to disk, in an HDF5 file format (readable with ``Catalog.read_h5()``)."""
        
        io_utils.mkdir_containing(filename)
        print(f'Writing {filename}\n', end='')
        
        with h5py.File(filename, 'w') as f:
            f.attrs.create('size', self.size)
            
            if self.name is not None:
                f.attrs.create('name', self.name)
                
            for col_name in self.col_names:
                col = getattr(self, col_name)
                assert col.shape == (self.size,)
                f.create_dataset(col_name, data=col)

        self._announce_file_write(filename)

        
    @staticmethod
    def from_fits(filename, col_name_pairs, name=None):
        """Reads FITS file in SDSS/DESILS format, and returns a Catalog object.
        
        The ``col_name_pairs`` arg should be a list of pairs (col_name, fits_col_name).
        See sdss.py and desils_lrg.py for examples."""

        if name is None:
            name = os.path.basename(filename)
            
        print(f'Reading {filename}\n', end='')
        catalog = Catalog(name=name, filename=filename)
        
        with fitsio.FITS(filename) as f:
            for (col_name, fits_col_name) in col_name_pairs:
                catalog.add_column(col_name, f[1].read(fits_col_name))
                
        catalog._announce_file_read()
        return catalog


    @staticmethod
    def from_text_file(filename, col_names, name=None):
        """Reads text file (one row per catalog object) and returns a Catlog object.
        The ``col_names`` arg should be a list of column names. See sdss.py for examples."""

        if name is None:
            name = os.path.basename(filename)
            
        # Note: we copy the transposed array, to get contiguous 1-d arrays.
        print(f'Reading {filename}\n', end='')
        cols = [ np.copy(x) for x in np.loadtxt(filename).T ]

        if len(cols) != len(col_names):
            raise RuntimeError(f'{filename}: file has {len(cols)} columns, {len(col_names)} cols were expected')

        cols = dict(((k,v) for (k,v) in zip(col_names,cols) if k is not None))
        catalog = Catalog(cols, name=name, filename=filename)

        catalog._announce_file_read()
        return catalog


    def show(self):
        """Prints a summary of the Catalog to stdout, with one line per column showing min/mean/max."""
        
        s1 = '' if (self.name is None) else f'name={self.name}, '
        s2 = '' if (self.filename is None) else f', filename={self.filename}'        
        print(f'Catalog({s1}size={self.size}{s2})')

        for k in sorted(self.col_names):
            v = getattr(self, k)
            s = [ ]

            # Numpy operations min/mean/max fail for some dtypes (e.g. uint1).
            try: s.append(f'min={np.min(v)}')
            except: pass
            
            try: s.append(f'mean={np.mean(v)}')
            except: pass
            
            try: s.append(f'max={np.max(v)}')
            except: pass

            # Should always succeed
            s.append(f'dtype={v.dtype}')
            
            print(f"    {k}: {', '.join(s)}")
        
    
    def _announce_file_read(self):
        s1 = '' if (self.name is None) else f'{self.name}: '
        s2 = '' if (self.filename is None) else f' from {self.filename}'
        print(f'{s1}Read {self.size} galaxies{s2}, columns {self.col_names}')

    def _announce_file_write(self, filename):
        s = '' if (self.name is None) else f'{self.name}: '
        print(f'{s}Wrote {self.size} galaxies to {filename}, columns {self.col_names}')
    
    def __str__(self):
        s1 = '' if (self.name is None) else f'name={self.name}, '
        s2 = '' if (self.filename is None) else f', filename={self.filename}'
        return f'Catalog({s1}size={self.size}, col_names={self.col_names}{s2})'
