import os
import sys
import h5py
import gzip
import uuid
import shutil
import fitsio
import pickle   # FIXME cPickle?
import tarfile
import zipfile
import functools
import subprocess
import numpy as np

# Avoid name collision with io_utils.wget() below.
import wget as wget_module


def read_pickle(filename, verbose=True):
    if verbose: print(f'Reading {filename}\n', end='')
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(filename, x):
    mkdir_containing(filename)
    print(f'Writing {filename}\n', end='') 
    with open(filename, 'wb') as f:
        pickle.dump(x, f)


def read_npy(filename, verbose=True):
    if verbose: print(f'Reading {filename}\n', end='')
    return np.load(filename, allow_pickle=True)


def write_npy(filename, arr):
    mkdir_containing(filename)
    print(f'Writing {filename}\n', end='') 
    np.save(filename, arr)

    
def read_npz(filename):
    """Returns a dictionary-like object (NpzFile) which maps names to arrays.

    Some arrays may have special names 'arr_0', 'arr_1', ... 
    This happens for arrays which were positional (not keyword) arguments to write_npz().
    """

    # read_npy() and read_npz() can be the same function,
    # since np.load() works for both .npy and .npz files.
    return read_npy(filename)


def write_npz(filename, *args, **kwds):
    """Example: write_npz('file.npz', a, b, xarr=x, yarr=y), where (a,b,x,y) are numpy arrays.

    In this example, when 'file.npz' is read (with np.load() or kszx.read_npz()), the return 
    value is a dictionary-like object (NpzFile) with keys 'arr_0', 'arr_1', 'xarr', 'yarr'.
    Note that the positional arguments (a,b) get special names 'arr_0', 'arr_1'.
    """

    mkdir_containing(filename)
    print(f'Writing {filename}\n', end='')
    np.savez(filename, *args, **kwds)


####################################################################################################


@functools.cache
def get_data_dir():
    """Returns root directory for auto-downloaded kszx data. (Either $KSZX_DATA_DIR or $HOME/kszx_data.)"""

    if 'KSZX_DATA_DIR' in os.environ:
        data_dir = os.environ['KSZX_DATA_DIR']
        print(f'kszx: using environment variable $KSZX_DATA_DIR: {data_dir}')
    elif 'HOME' in os.environ:
        data_dir = os.path.join(os.environ['HOME'], 'kszx_data')
        print(f'kszx: environment variable $KSZX_DATA_DIR not defined, using {data_dir} instead')
    else:
        raise RuntimeError('kszx.get_data_dir(): neither $KSZX_DATA_DIR nor $HOME environment variables were defined?!')

    if os.path.isdir(data_dir):
        return data_dir
    
    raise RuntimeError(f'kszx: data directory {data_dir} not found.\n'
                       + f"This is the directory where kszx will auto-download and retrieve data files.\n"
                       + f"Here are some options:\n"
                       + f"  1. if you want to store data in this directory, do 'mkdir -p {data_dir}'\n"
                       + f"  2. if you want to use a different directory, do 'ln -s /some/other/dir {data_dir}\n"
                       + f"  3. if you want to use a different directory, another option is to set the $KSZX_DATA_DIR env variable\n"
                       + f"For more discussion, see: https://kszx.readthedocs.io/en/latest/intro.html#data-files")


def mkdir_containing(filename):
    assert isinstance(filename, str)
    
    d = os.path.dirname(filename)
    if (d == '') or os.path.exists(d):
        return
    
    print(f'Creating directory {d}')
    os.makedirs(d, exist_ok=True)
    

def do_download(abspath, download=False, dlfunc=None):
    """This short helper function appears in many functions which decide whether to download a file.

    Here and in other parts of kszx, the 'dlfunc' argument gives the name of a transitive caller that
    expects the file to be present, and has a 'download=False' optional argument. This information is
    only used when generating exception-text (to tell the user how to download the file).
    """
    
    if os.path.exists(abspath):
        return False
    if download:
        return True
    if dlfunc is not None:
        raise RuntimeError(f'File {abspath} not found. To auto-download, call {dlfunc}() with download=True.')
    return False


def wget(filename, url):
    if os.path.exists(filename):
        print(f'File {filename} already exists -- skipping download')
        return
    
    mkdir_containing(filename)
    print(f'Downloading {url} -> {filename}')
    
    wget_module.download(url, out=filename)
    print()   # extra newline after progress bar
    assert os.path.exists(filename)


def unpack(srcfile, expected_dstfile=None):
    """Unpacks a file ending in one of: .gz, .tar, .tgz, .tar.gz"""
    
    if not os.path.exists(srcfile):
        raise RuntimeError(f"kszx.io_utils.unpack(): source file '{srcfile}' does not exist")
    
    if srcfile.endswith('.tar'):
        print(f'Un-tarring {srcfile}')
        tar = tarfile.open(srcfile, 'r:')
        tar.extractall(path = os.path.dirname(srcfile))
        tar.close()
    
    elif srcfile.endswith('.tar.gz') or srcfile.endswith('.tgz'):
        print(f'Un-tarring and gunzipping {srcfile}')
        tar = tarfile.open(srcfile, 'r:gz')
        tar.extractall(path = os.path.dirname(srcfile))
        tar.close()
    
    elif srcfile.endswith('.gz'):
        dstfile = srcfile[:-3]
        print(f'Gunzipping {srcfile} -> {dstfile}')
        with gzip.open(srcfile, 'rb') as f_in:
            with open(dstfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    elif srcfile.endswith('.zip'):
        print(f'Unzipping {srcfile}')
        with zipfile.ZipFile(srcfile, 'r') as f:
            f.extractall(os.path.dirname(srcfile))

    else:
        raise RuntimeError(f"kszx.io_utils.unpack(): source file '{srcfile}' must end in .zip, .gz, .tgz, or .tar.gz")

    if (expected_dstfile is not None) and not os.path.exists(expected_dstfile):
        raise RuntimeError(f"kszx.io_utils.unpack(): after unpacking {srcfile=}, {expected_dstfile=} does not exist")

    
def nersc_scp_download(remote_abspath, local_abspath):
    """A temporary kludge, since ACT DR6 maps are currently on NERSC but not LAMBDA.

    Documentation by example:

      nersc_scp_download(
        remote_abspath = '/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/maps/published/act-planck_dr6.02_coadd_AA_daynight_f090_map_srcfree.fits',
        local_abspath = '/home/kmsmith/kszx_data/act/dr6/maps/published/act-planck_dr6.02_coadd_AA_daynight_f090_map_srcfree.fits'
      )
    """
    
    if os.path.exists(local_abspath):
        return

    print(f"File {local_abspath} is not currently available on the web -- the only option is a NERSC scp download.")

    mkdir_containing(local_abspath)
    tmpfile = local_abspath +  f'.tmp{uuid.uuid4().hex[:8]}'
    ssh_key = os.path.join(os.environ['HOME'], '.ssh', 'nersc')
    ssh_args = ['scp', '-i', ssh_key, f'perlmutter.nersc.gov:{remote_abspath}', tmpfile ]
    
    if not os.path.exists(ssh_key):
        raise RuntimeError('To download files from NERSC, you need to have a NERSC account, and run sshproxy.sh. '
                           + 'See https://docs.nersc.gov/connect/mfa/#sshproxy')
    
    try:
        print(f"Executing command: {' '.join(ssh_args)}")
        subprocess.run(ssh_args, check=True)
    
    except subprocess.CalledProcessError as e:
        print(f"Couldn't download from NERSC. One possible reason is that your ssh key is out of date, "
              + f"and you need to rerun sshproxy.sh. (See https://docs.nersc.gov/connect/mfa/#sshproxy). "
              + f"Error message was: {e}")

    if not os.path.exists(tmpfile):
        raise RuntimeError(f'scp command succeeded, but destination file {tmpfile} was not created?!')

    print(f'Renaming {tmpfile} -> {local_abspath}')
    os.rename(tmpfile, local_abspath)


####################################################################################################


def show_file(filename, file_type=None, file=None):
    """Prints (to stdout) a simple human-readable summary of file with given name.

    The file_type can be one of:

      pkl        python pickle
      npy        numpy array
      npz        numpy NpzFile
      h5, hdf5   HDF5 file
      fits       FITS file

    If file_type is None, then the file_type is determined from the filename in 
    a brain-dead way (e.g. if filename='x.pkl', then file_type='pkl').

    Can be invoked from command line with 'python -m kszx show <filename>'.
    """

    assert isinstance(filename, str)
        
    if file_type is None:
        b = os.path.basename(filename)
        i = b.rfind('.')
        if (i < 0) or (i >= len(b)-1):
            raise RuntimeError(f"Couldn't determine file_type from {filename=}")
        file_type = b[(i+1):]

    if file_type == 'pkl':
        x = read_pickle(filename)
        print(x, file=file)

    elif file_type == 'npy':
        x = read_npy(filename)
        print('Shape:', x.shape)
        print(x, file=file)

    elif file_type == 'npz':
        _show_npz(filename, file=file)

    elif file_type in [ 'h5', 'hdf5' ]:
        _show_h5(filename, file=file)

    elif file_type in [ 'fits' ]:
        _show_fits(filename, file=file)
            
    else:
        raise RuntimeError(f"kszx.show_file(): unsupported file_type {file_type}")
        

def _show_npz(x, file=None, indent='  '):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename or a NpzFile (class numpy.lib.npyio.NpzFile).
    """

    if isinstance(x, str):
        x = read_npz(x)

    if not isinstance(x, np.lib.npyio.NpzFile):
        raise RuntimeError('kszx._show_npz(): argument must be either string or NpzFile')

    print(x, file=file)
    
    for k,v in x.items():
        print(f'{indent}{k}: shape={v.shape}, min={np.min(v)}, max={np.max(v)}', file=file)


def _show_h5_attrs(x, prefix, file):
    for k,v in sorted(x.attrs.items()):
        print(f"{prefix}Attribute '{k}': {v}", file=file)
    

def _show_h5(x, prefix=None, file=None, indent='  '):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename, an h5py.File, or an h5py.Group.

    The default value of the 'indent' argument indents by two spaces at every level.
    """

    if prefix is None:
        prefix = ''
    if file is None:
        file = sys.stdout
    
    if isinstance(x, str):
        with h5py.File(x) as f:
            print(f"{prefix}File '{x}'", file=file)
            _show_h5(f, prefix + indent, file, indent)
        return

    _show_h5_attrs(x, prefix, file)

    # x.items() = "all groups and datasets in a single dict".
    # First pass: loop over datasets.
    
    for k,v in sorted(x.items()):
        if isinstance(v, h5py.Dataset):
            try:
                s = f', min={np.min(v)}, max={np.max(v)}'
            except:
                s = ''
            print(f"{prefix}Dataset '{k}': shape={v.shape}, dtype={v.dtype}{s}", file=file)
            _show_h5_attrs(v, prefix + indent, file)
        elif not isinstance(v, h5py.Group):
            print(f"{prefix}Unrecognized element '{k}': type={type(v)}", file=file)

    # Second pass: loop over groups

    for k,v in sorted(x.items()):
        if isinstance(v, h5py.Group):
            print(f"{prefix}Group '{k}'", file=file)
            _show_h5(v, prefix + indent, file, indent)


def _show_fits(x, file=None):
    """Intended to be called through show_file(), or via command line.

    Prints (to stdout) a simple human-readable summary of 'x', which can be
    either a filename, or a fitsio.FITS object.

    This quick-and-dirty implementation just calls __str__() functions in the
    fitsio python module (which is pretty good).
    """

    if isinstance(x, str):
        with fitsio.FITS(x) as f:
            _show_fits(f)

    elif isinstance(x, fitsio.FITS):
        for hdu in x:
            print(hdu, file=file)

    else:
        raise RuntimeError('kszx._show_fits(): argument must be a string, or of type fitsio.FITS')
