import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser('download_act')
    p.add_argument('dr', type=int, help='Either 5 or 6 (for DR5 or DR6)')

    p = subparsers.add_parser('download_desi')
    p.add_argument('survey', help='Survey name such as LRG_NGC')
    p.add_argument('-n', help='number of random files to download (default is to download all 18)')

    p = subparsers.add_parser('download_planck')

    p = subparsers.add_parser('download_sdss')
    p.add_argument('survey', help='Survey name such as CMASS_North')

    from .quijote import _sim_types, _valid_redshifts
    _sim_types_str = ', '.join(sorted(_sim_types.keys()))
    _redshifts_str = ', '.join(str(z) for z in _valid_redshifts)

    p = subparsers.add_parser(
        'download_quijote',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Download Quijote simulation data via Globus.',
        epilog=(
            f'Available sim types:\n  {_sim_types_str}\n\n'
            f'Available redshifts:\n  {_redshifts_str}\n\n'
            'By default, downloads halos, pk, and linear_pk (not snapshots).\n'
            'Use -s to also download snapshots.\n\n'
            'Examples:\n'
            '  python -m kszx download_quijote 100           # realizations 0..99\n'
            '  python -m kszx download_quijote 100 200       # realizations 100..199\n'
            '  python -m kszx download_quijote 500 -t Om_p -z 0.5\n'
            '  python -m kszx download_quijote 10 -s\n'
        ),
    )
    p.add_argument('N', type=int, nargs='+',
                   help='one int N -> download realizations 0..N-1; two ints N1 N2 -> realizations N1..N2-1')
    p.add_argument('-t', metavar='SIM_TYPE', default='fiducial', help='simulation type (default: fiducial)')
    p.add_argument('-z', metavar='REDSHIFT', type=float, default=0.0, help='redshift (default: 0)')
    p.add_argument('-s', action='store_true', help='also download snapshots')

    p = subparsers.add_parser(
        'download_websky',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Download Websky v0.4 data products from NERSC (via scp).',
        epilog=(
            'By default, downloads NOTHING; pass one or more flags to select products.\n\n'
            'Examples:\n'
            '  python -m kszx download_websky --halos\n'
            '  python -m kszx download_websky --kappa --ksz --ksz-patchy --tsz --cib 100 143 217 353 545 857\n'
            '  python -m kszx download_websky --list-cib-freqs\n'
            '  python -m kszx download_websky --all\n'
        ),
    )
    p.add_argument('--halos',      action='store_true', help='download halos.pksc (~32 GB)')
    p.add_argument('--kappa',      action='store_true', help='download kap.fits (full lensing convergence)')
    p.add_argument('--ksz',        action='store_true', help='download ksz.fits (late-time kSZ)')
    p.add_argument('--ksz-patchy', action='store_true', help='download ksz_patchy.fits (reionization kSZ)')
    p.add_argument('--tsz',        action='store_true', help='download tsz.fits (nside=8192)')
    p.add_argument('--isw',        action='store_true', help='download isw.fits')
    p.add_argument('--cmb-alm', choices=['lensed', 'unlensed', 'all'], default=None,
                   help='download CMB alm files (both seeds)')
    p.add_argument('--cib', nargs='+', default=None,
                   help="CIB frequencies in GHz, or 'all' to fetch all 45")
    p.add_argument('--list-cib-freqs', action='store_true',
                   help='list available CIB frequencies and exit')
    p.add_argument('--all', action='store_true',
                   help='shortcut for every product including the catalog (~32 GB)')
    p.add_argument('--host', default=None,
                   help="override WEBSKY_SSH_HOST (default 'perlmutter')")

    p = subparsers.add_parser('show')
    p.add_argument('filename')

    p = subparsers.add_parser('test')
    p = subparsers.add_parser('time')

    p = subparsers.add_parser('kszpipe_run')
    p.add_argument('input_dirname')
    p.add_argument('output_dirname')
    p.add_argument('-p', type=int, default=4, help='number of processes for multiprocessing Pool (default 4)')
    
    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
        sys.exit(2)
    elif args.command == 'download_act':
        from . import act
        act.download(dr=args.dr)
    elif args.command == 'download_desi':
        from . import desi
        nrfiles = int(args.n) if (args.n is not None) else None
        desi.download(args.survey, dr=1, nrfiles=nrfiles)
    elif args.command == 'download_planck':
        from . import planck
        planck.download()
    elif args.command == 'download_sdss':
        from . import sdss
        sdss.download(args.survey)
    elif args.command == 'download_quijote':
        from . import quijote
        products = ['halos', 'pk', 'linear_pk']
        if args.s:
            products.append('snapshots')
        if len(args.N) == 1:
            realizations = range(args.N[0])
        elif len(args.N) == 2:
            realizations = range(args.N[0], args.N[1])
        else:
            parser.error('download_quijote: expected 1 or 2 positional integers (got %d)' % len(args.N))
        quijote.download(args.t, realizations, products=products, redshifts=(args.z,))
    elif args.command == 'download_websky':
        from . import websky
        if args.list_cib_freqs:
            for f in websky.CIB_FREQS_GHZ:
                print(f)
            sys.exit(0)
        if args.host:
            websky.WEBSKY_SSH_HOST = args.host
        if args.all:
            args.halos      = True
            args.kappa      = True
            args.ksz        = True
            args.ksz_patchy = True
            args.tsz        = True
            args.isw        = True
            if args.cmb_alm is None: args.cmb_alm = 'all'
            if args.cib is None:     args.cib = ['all']
        cib_freqs = ()
        if args.cib is not None:
            if args.cib == ['all']:
                cib_freqs = websky.CIB_FREQS_GHZ
            else:
                cib_freqs = tuple(float(f) for f in args.cib)
        cmb_alm = () if args.cmb_alm is None else (
            ('lensed', 'unlensed') if args.cmb_alm == 'all' else (args.cmb_alm,))
        websky.download(halos=args.halos, kappa=args.kappa,
                        ksz=args.ksz, ksz_patchy=args.ksz_patchy,
                        tsz=args.tsz, isw=args.isw,
                        cib=cib_freqs, cmb_alm=cmb_alm)
    elif args.command == 'show':
        from . import io_utils
        io_utils.show_file(args.filename)
    elif args.command == 'test':
        from . import tests
        tests.run_all_tests()   # defined in kszx/tests/__init__.py
    elif args.command == 'time':
        from . import timing
        from . import cpp_kernels
        print(f'{cpp_kernels.omp_get_max_threads() = }')
        timing.time_interpolation()
        timing.time_multiply_xli_real_space()
        timing.time_multiply_xli_fourier_space()
    elif args.command == 'kszpipe_run':
        from .KszPipe import KszPipe
        kszpipe = KszPipe(args.input_dirname, args.output_dirname)
        kszpipe.run(processes=args.p)
    else:
        parser.print_help()
        sys.exit(2)
