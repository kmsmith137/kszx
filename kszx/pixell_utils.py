"""
The ``kszx.pixell_utils`` module contains wrappers around the pixell library.

References:
   - https://github.com/simonsobs/pixell
   - https://pixell.readthedocs.io/en/latest
   - https://github.com/simonsobs/pixell_tutorials
"""

import numpy as np
import pixell.enmap
import pixell.enplot
import pixell.curvedsky

from . import utils
from . import io_utils

from .Catalog import Catalog


def read_map(filename):
    """Reads pixell map in FITS format, and returns a pixell.enmap.
    
    If the FITS file contains temperature + polarization, only temperature will be returned!
    """
    print(f'Reading {filename}\n', end='')
    assert filename.endswith('.fits')
    return pixell.enmap.read_map(filename)


def write_map(filename, m):
    """Writes pixell map in FITS format."""
    
    # I learned the hard way that you need this assert! If pixell.enmap.write_map() is
    # called with a numpy array (instead of a pixell ndmap), it will write a "time-bomb"
    # WCS which causes some pixell functions to fail later, with cryptic error messages.
    assert isinstance(m, pixell.enmap.ndmap)
    
    print(f'Writing {filename}\n', end='')
    assert filename.endswith('.fits')
    io_utils.mkdir_containing(filename)
    pixell.enmap.write_map(filename, m)


def plot_map(m, downgrade, nolabels=True, filename=None, title=None, **kwds):
    """Plots pixell map 'm'.

    Thin wrapper around ``pixell.enplot()``, just adding a few tweaks:

       - Make downgrade argument mandatory (usually needed in practice to avoid
         runaway heavyweight behavior). Suggest ``downgrade=20`` for ACT.

       - Make ``nolabels=True`` the default (I haven't figured out how to make pixell
         labels look reasonable).

       - Add 'filename' argument to select between show/write.

       - Add 'title' argument to display a title above the plot.

    For more kwds, see:
      https://pixell.readthedocs.io/en/latest/reference.html#pixell.enplot.plot"""

    assert isinstance(m, pixell.enmap.ndmap)

    bunch = pixell.enplot.plot(m, downgrade=downgrade, nolabels=nolabels, **kwds)

    if title is None:
        # No title: use pixell's native show/write
        if filename is None:
            pixell.enplot.show(bunch)
        else:
            print(f'Writing {filename}\n', end='')
            assert filename.endswith('.png')
            io_utils.mkdir_containing(filename)
            pixell.enplot.write(filename[:-4], bunch)
    else:
        # Title requested: use matplotlib to render the image with a proper title
        import matplotlib.pyplot as plt
        img = bunch.img if hasattr(bunch, 'img') else bunch[0].img
        aspect = img.height / img.width
        figwidth = 14
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * aspect + 0.5))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        fig.tight_layout(pad=0.5)
        if filename is None:
            plt.show()
        else:
            print(f'Writing {filename}\n', end='')
            assert filename.endswith('.png')
            io_utils.mkdir_containing(filename)
            fig.savefig(filename, dpi=80, bbox_inches='tight')
            plt.close(fig)


####################################################################################################


def _accum_catalog(ret, shape, wcs, catalog, weights, allow_outliers, normalize_sum=None):
    """Helper for map_from_catalog(). Returns sum of weights."""

    if not allow_outliers:
        # Note: if allow_outliers is False, then pixell_utils.ang2pix() returns (idec, ira).
        msg = "kszx.pixell_utils.map_from_catalog(): some points were out-of-bounds, you may want allow_outliers=True"
        idec, ira = ang2pix(shape, wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=False, outlier_errmsg=msg)
    else:
        # Note: if allow_outliers is True, then pixell_utils.ang2pix() returns (idec, ira, mask).
        idec, ira, mask = ang2pix(shape, wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=True)
        idec, ira = idec[mask], ira[mask]
        wvec_flag = (weights is not None) and (weights.ndim > 0)
        weights = weights[mask] if wvec_flag else weights

    ret_1d = np.reshape(ret, (-1,))
    ix_1d = idec*shape[1] + ira

    # Return sum of weights, after applying outlier mask, but before applying 'normalize_sum'.
    return utils.scattered_add(ret_1d, ix_1d, weights, normalize_sum=normalize_sum)
    

def map_from_catalog(shape, wcs, gcat, weights=None, rcat=None, rweights=None, normalized=True, allow_outliers=True):
    r"""Project a 3-d galaxy catalog to a 2-d pixell map.

    Function args:

      - ``shape``, ``wcs``: these args define the pixell coordinate system.

      - ``gcat`` (:class:`~kszx.Catalog`): galaxy catalog.
        Must define columns ``ra_deg``, ``dec_deg``.

      - ``weights`` (1-d numpy array, optional): per-galaxy weights.

      - ``rcat``, ``rweights`` (optional): random catalog to be subtracted.
        The rweights are renormalized so that sum(rweights) = -sum(weights).
    
      - ``normalized`` (boolean): If normalized=True, then the output map includes a
        factor ``1 / (pixel area)``. This normalization best represents a sum of delta
        functions $f(x) = \sum_j w_j \delta^2(\theta - \theta_j)$.

      - ``allow_outliers`` (boolean): If 'allow_outliers' is True, then "out-of-bounds"
        galaxies (i.e. galaxies outside the boundaries of the pixell map) will be silently
        discarded. If 'allow_outliers' is False, then out-of-bounds galaxies will raise an
        exception.

    Returns a ``pixell.enmap.ndmap``.
    """

    assert isinstance(gcat, Catalog)
    assert isinstance(rcat, Catalog) or (rcat is None)

    if (rcat is None) and (rweights is not None):
        raise RuntimeError("kszx.healpix_utils.map_from_catalog(): 'rcat' arg is None, but 'rweights' arg is not None")

    weights = utils.asarray(weights, 'kszx.healpix_utils.map_from_catalog', 'weights', dtype=float, allow_none=True)
    rweights = utils.asarray(rweights, 'kszx.healpix_utils.map_from_catalog', 'rweights', dtype=float, allow_none=True)

    assert (weights is None) or (weights.ndim == 0) or (weights.shape == (gcat.size,))
    assert (rweights is None) or (rweights.ndim == 0) or (rweights.shape == (rcat.size,))

    ret = np.zeros(shape)
    wsum = _accum_catalog(ret, shape, wcs, gcat, weights, allow_outliers)

    if rcat is not None:
        _accum_catalog(ret, shape, wcs, rcat, rweights, allow_outliers, normalize_sum = -wsum)

    if normalized:
        ret /= pixell.enmap.pixsizemap(shape, wcs)
    
    return pixell.enmap.enmap(ret, wcs=wcs)


def eval_map_on_catalog(m, catalog, pad=None, return_mask=False):
    r"""Evaluates pixell map 'm' at (ra,dec) values from catalog, and returns a 1-d array.

    Function arguments:

       - ``m`` (pixell.enmap.ndmap): 2-d map to be evaluated.
    
       - ``catalog`` (:class:`~kszx.Catalog`): galaxy catalog defining (ra, dec) values
         where map is evaluated. Must define columns ``ra_deg``, ``dec_deg``.

       - ``pad`` (boolean): Since pixell maps can be partial-sky, some of the (ra,dec)
         pairs may be "outliers", i.e. outside the map footprint. If ``pad=None`` (the default),
         this raises an exception. If 'pad' is specified (e.g. ``pad=0.0``) then outlier values
         are replaced by 'pad'.

       - ``return_mask`` (boolean): Defines what quantity this function returns.

         If ``return_mask=False`` (the default), returns a 1-d array 'map_vals'.

         If ``return_mask=True``, returns a pair ``(map_vals, mask)``, where 'mask' is
         a 1-d boolean array which is True for non-outliers, False for outliers.
    """

    assert isinstance(m, pixell.enmap.ndmap)
    assert isinstance(catalog, Catalog)

    if return_mask and (pad is None):
        raise RuntimeError("kszx.pixell_utils.eval_map_on_catalog(): if return_mask=True, then 'pad' must be specified")
    if pad is not None:
        pad = float(pad)
    
    # pixell_utils.ang2pix() is defined later in this file.
    # It returns (idec,ira) if allow_outliers is False, or (idec,ira,mask) if allow_outliers is True.

    msg = "kszx.pixell_utils.eval_map_on_catalog(): some points are out-of-bounds, you may want to specify the 'pad' arugment"
    t = ang2pix(m.shape, m.wcs, catalog.ra_deg, catalog.dec_deg, allow_outliers=(pad is not None), outlier_errmsg=msg)
    
    idec, ira = t[0], t[1]
    mask = t[2] if (len(t) > 2) else None
    map_vals = m[idec,ira]

    if pad is not None:
        map_vals = np.where(mask, map_vals, pad)

    return (map_vals, mask) if return_mask else map_vals
    

####################################################################################################


def alm2map(alm, shape, wcs):
    """Applies alm2map spherical transform, returns pixell.enmap.ndmap."""
    
    # FIXME map2alm() will blow up with a cryptic error message if it gets a bad WCS (e.g. zeroed).
    # Should find a general test for WCS badness, and add asserts throughout this file.
    
    alm = np.asarray(alm)
    assert alm.ndim == 1
    assert (alm.dtype == complex) or (alm.dtype == np.complex64)

    m = pixell.enmap.enmap(np.zeros(shape), wcs=wcs)
    pixell.curvedsky.alm2map(alm, m)
    return m
    

def map2alm(m, lmax):
    """Applies map2alm spherical transform, returns alms as 1-d complex numpy array."""
    
    assert isinstance(m, pixell.enmap.ndmap)
    assert lmax >= 0
    return pixell.curvedsky.map2alm(m, lmax=lmax)


####################################################################################################


def ang2pix(shape, wcs, ra_deg, dec_deg, allow_outliers=False, outlier_errmsg=None):
    """Given (ra,dec) values on the sky, returns the corresponding pixel indices.

    **Note:** if you are interested in this function in order to pixelize a Catalog,
    you may want to call :func:`~kszx.pixell_utils.map_from_catalog` instead.

    Function arguments:

      - ``shape``, ``wcs``: these args define the pixell coordinate system.

      - ``ra_deg``, ``dec_deg``: numpy arrays with same shape, containing (ra, dec) values.

      - ``allow_outliers`` (boolean): since pixell maps can be partial-sky, pixel indices
        can be out of bounds. In this case:
    
        If ``allow_outliers=False``, we raise an exception.

        If ``allow_outliers=True``, then we "clip" the returned idec/ira arrays
        so that they are in bounds, and return a boolean mask to indicate which
        values are valid (see below).

      - ``outlier_errmsg`` (string, optional): error message if exception is thrown.

    The return value depends on the value of ``allow_outliers``:

      - If ``allow_outliers=False``, returns a pair ``(idec, ira)``.
    
      - If ``allow_outliers=True``, returns a triple ``(idec, ira, mask)``.

    The ``(idec, ira)`` arrays are integer-valued, and contain pixel indices along
    each of the axes of the 2-d pixell map. The ``mask`` array is boolean-valued.
    
    Roughly equivalent to ``healpy.ang2pix(..., latlon=True)``.
    Note that the argument ordering is (ra, dec), consistent with healpy but not pixell!
    """

    # pixell wants a single 'coords' array, with [dec,ra] converted to radians
    coords = np.stack(np.broadcast_arrays(dec_deg, ra_deg), axis=0)   # makes copy
    coords *= (np.pi/180.)

    # pixell.enmap.sky2pix() returns a single output array, where axis 0 is {idec,ira}.
    icoords = pixell.enmap.sky2pix(
        shape,
        wcs,
        coords,
        safe = True,      # pixell does not document this argument
        corner = False,   # indicates that integer coords correspond to pixel centers (not corners)
        bcheck = (not allow_outliers)   # pixell does not document this argument
    )

    icoords = np.asarray(np.round(icoords), dtype=int)

    icoords_clipped = np.clip(
        icoords,
        a_min = 0,
        a_max = np.reshape(shape, (2,)+(1,)*(icoords.ndim-1)) - 1   # note (-1) at end
    )

    mask = np.logical_and(
        icoords[0] == icoords_clipped[0], 
        icoords[1] == icoords_clipped[1]
    )

    idec, ira = icoords_clipped   # not icoords
    
    if allow_outliers:
        return idec, ira, mask
    elif np.all(mask):
        return idec, ira

    if outlier_errmsg is None:
        outlier_errmsg = 'kszx.pixell_utils.ang2pix(): called with allow_outliers=False, and some points are out-of-bounds'
    raise RuntimeError(outlier_errmsg)


####################################################################################################


def uK_arcmin_from_ivar(ivar, max_uK_arcmin=1.0e6):
    r"""Given an inverse variance map, returns associated noise map (in uK-arcmin).

    Based on Mat's orphics library:
       https://github.com/msyriac/orphics/blob/master/orphics/maps.py

    Function args:
    
      - ``ivar`` (pixell map): inverse variance map, units (uK)^{-2}.
        (For example, the return value from :func:`~kszx.act.read_ivar()`.)

      - ``max_uK_arcmin`` (float): max allowed value in the return map (prevents
        dividing by zero).

    Return value is a pixell map with units uK-arcmin.

    Equivalent to::
    
       ps = ivar.pixsizemap()
       uK_arcmin = sqrt(ps/ivar) * (60*180/np.pi)
       return np.minimum(uK_arcmin, max_uK_arcmin)

    but regulated to avoid dividing by zero.
    """

    assert isinstance(ivar, pixell.enmap.ndmap)
    assert max_uK_arcmin > 0

    a = 60*180/np.pi
    ps = ivar.pixsizemap()
    epsilon = 0.9 * np.min(ps) * (a/max_uK_arcmin)**2
    
    uK_arcmin = a * np.sqrt(ps / np.maximum(ivar,epsilon))
    return np.minimum(uK_arcmin, max_uK_arcmin)


def fkp_from_ivar(ivar, cl0, normalize=True, return_wvar=False):
    r"""Given an inverse variance map, returns associated FKP weighting.

    Function args:

      - ``ivar`` (pixell map): inverse variance map, units (uK)^{-2}.
        (For example, the return value from :func:`~kszx.act.read_ivar()`.)

      - ``cl0`` (float): FKP weight function parameter $C_l^{(0)}$.

          - Intuitively, ``cl0`` = "fiducial signal $C_l$ at wavenumber $l$ of interest".
          - ``cl0=0`` corresponds to inverse noise weighting.
          - Large ``cl0`` corresponds to uniform weighing (but ``cl0=np.inf`` won't work).
          - I usually use ``cl0 = 0.01`` for plotting all-sky CMB temperature maps.
          - For kSZ filtering, ``cl0=3e-5`` is a reasonable choice.

      - ``normalize`` (boolean): if True, then we normalize the weight function 
        so that $\max(W(\theta))=1$.

      - ``return_wvar`` (boolean): if True, then we return $W(\theta) / \mbox{ivar}(\theta)$,
        instead of returning $W(\theta)$.
    
    Returns a pixell map.

    The FKP weighting is defined by:
    
    $$W(\theta) &= \frac{1}{C_l^{(0)} + N(\theta)} \\
    N(\theta) &\equiv \frac{\mbox{Pixel area}}{\mbox{ivar}(\theta)} \hspace{1cm} \mbox{"Local" noise power spectrum}$$

    In implementation, in order to avoid divide-by-zero for ivar=0, we compute $W(\theta)$ equivalently as:
    
    $$W(\theta) = \frac{\mbox{ivar}(\theta)}{(\mbox{pixel area}) + C_l^{(0)} \mbox{ivar}(\theta)}$$
    """
    
    assert isinstance(ivar, pixell.enmap.ndmap)
    assert ivar.ndim == 2
    assert np.all(ivar >= 0.0)
    assert cl0 >= 0.0

    wvar = 1.0 / (ivar.pixsizemap() + cl0 * ivar)
    w = wvar * ivar
    
    wmax = np.max(w)
    assert wmax > 0.0

    ret = wvar if return_wvar else w
    
    if normalize:
        ret /= wmax

    return ret


def sensitivity_curve(ivar, step, n):
    r"""Given an ivar map, computes the "sensitivity function" $S(x)$ at x = [ step, ..., n*step].
    
    We define the "sensitivity function" $S(x)$ to be the sky area (in deg^2)
    with sensitivity >= x, where x has units (uK-armcin)^{-2}. Note that $S(x)$
    is a decreasing function of x. This is useful for making a plot of sky sensitivity.

    Function args:

      - ``ivar`` (pixell map): inverse variance map, units (uK)^{-2}.
        (For example, the return value from :func:`~kszx.act.read_ivar()`.)

      - ``step``: units (uK-arcmin)^{-2}, see above.
        Recommend ``step = 1.0e-5`` for ACT.

      - ``n``: number of returned points, see above.

    Returns 1-d array ``[ S(step), S(2*step), ..., S(n*step) ]``.

    Example usage::

      step = 1.0e-5
      nbins = 3000
      ivar = kszx.act.read_ivar(150, dr=6, download=True)    
    
      x = step * np.arange(nbins)
      Sx = kszx.pixell_utils.sensitivity_curve(ivar,step,nbins)

      plt.plot(x, Sx)
      plt.yscale('log')
      plt.ylim(1, 10**5)
      plt.xlabel(r'Threshold $x$ ($\mu$K-arcmin)$^{-2}$')
      plt.ylabel(r'Sky area with sensitivity $> x$ (deg$^2$)')
    """
    
    assert isinstance(ivar, pixell.enmap.ndmap)
    ps = ivar.pixsizemap() * (180./np.pi)**2  # pixel size in deg^2
    x = (ivar / ps) / 3600.                   # sensitivity in (uK-arcmin)^{-2}
    x = np.array(x/step, dtype=int)
    x = np.minimum(x, n)
    x = np.bincount(x.reshape((-1,)), weights=ps.reshape((-1,)), minlength=n+1)
    return np.cumsum(x[::-1])[-2::-1]
