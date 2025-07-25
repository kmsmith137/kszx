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
