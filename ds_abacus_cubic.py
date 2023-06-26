import os
import time
import yaml
import numpy as np
import argparse
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Header
from abacusnbody.hod.abacus_hod import AbacusHOD
from densitysplit.pipeline import DensitySplit
from pathlib import Path
from pycorr import TwoPointCorrelationFunction, setup_logging
from cosmoprimo.fiducial import AbacusSummit
import logging
import time
import warnings


def get_rsd_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict['LRG']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x'] + args.boxsize / 2
    y = data['y'] + args.boxsize / 2
    z = data['z'] + args.boxsize / 2
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % args.boxsize
    y_rsd = y_rsd % args.boxsize
    z_rsd = z_rsd % args.boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd


def density_split(data_positions, boxsize, boxcenter, seed=42,
    smoothing_radius=20, nquantiles=5, filter_shape='Gaussian',
    nthreads=1, method='mesh', cellsize=5.0, nquery=None):
    """Split random points according to their local density
    density."""
    ds = DensitySplit(data_positions=data_positions, boxsize=boxsize,
                      cellsize=cellsize, boxcenter=boxcenter,
                      wrap=True, nthreads=nthreads)
    np.random.seed(seed=seed)
    if nquery is None:
        nquery = nquantiles * len(data_positions)
    sampling_positions = np.random.uniform(0, boxsize, (nquery, 3))
    if method == 'mesh':
        density = ds.get_density_mesh(smoothing_radius=smoothing_radius,
                                      sampling_positions=sampling_positions,)
    elif method == 'paircount':
        density = ds.get_density_paircount(smoothing_radius=smoothing_radius,
            sampling_positions=sampling_positions, nthreads=nthreads,
            filter_shape=filter_shape)
    quantiles, quantiles_idx = ds.get_quantiles(
        nquantiles=nquantiles, return_idx=True)
    return quantiles, quantiles_idx, density


def get_distorted_positions(positions, q_perp, q_para, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the 
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    positions_ap[:, 0] /= factor_x
    positions_ap[:, 1] /= factor_y
    positions_ap[:, 2] /= factor_z
    return positions_ap

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = [boxsize/factor_x, boxsize/factor_y, boxsize/factor_z]
    return boxsize_ap

def output_mock(mock_dict, newBall, fn, tracer):
    """Save HOD catalogue to disk."""
    Ncent = mock_dict[tracer]['Ncent']
    mock_dict[tracer].pop('Ncent', None)
    cen = np.zeros(len(mock_dict[tracer]['x']))
    cen[:Ncent] += 1
    mock_dict[tracer]['cen'] = cen
    table = Table(mock_dict[tracer])
    header = Header({'Ncent': Ncent, 'Gal_type': tracer, **newBall.tracers[tracer]})
    myfits = fits.BinTableHDU(data = table, header = header)
    logger.info(f"Saving HOD to {fn}")
    myfits.writeto(fn, overwrite=True)

def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    # read the parameters 
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
        # Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1 # a lot of this is a placeholder for something more suited for multi-tracer
    ngal_dict = Ball.compute_ngal(Nthread = nthread)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)
    return mock_dict

def setup_hod(config):
    print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params)
    newBall.params['Lbox'] = args.boxsize
    # parameters to fit
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return newBall, param_mapping, param_tracer, data_params


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    logger = logging.getLogger('ds_abacus_cubic')
    setup_logging(level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--nthreads", type=int, default=256)
    parser.add_argument("--boxsize", type=int, default=2000)
    parser.add_argument("--cellsize", type=float, default=5.0)
    parser.add_argument("--redshift", type=float, default=0.5)
    parser.add_argument("--split", type=str, default='z')
    parser.add_argument("--filter_shape", type=str, default='Gaussian')
    parser.add_argument("--smoothing_radius", type=float, default=10)
    parser.add_argument("--nquantiles", type=int, default=5)
    parser.add_argument("--nquery", type=int, default=None)
    parser.add_argument("--save_hod", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_quantiles", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_density", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_clustering", action=argparse.BooleanOptionalAction)
    parser.add_argument("--outdir", type=str, default="/pscratach/sd/c/cuesta/ds_boss/")
    parser.add_argument("--hod_config_dir", type=str, default="/pscratch/sd/c/cuesta/ds_boss/code/")
    parser.add_argument('--hod_params_dir', type=str, default="/pscratch/sd/c/cuesta/ds_boss/parameters/HOD/")
    parser.add_argument('--quantiles_clustering', help="List of quantiles to calculate the clustering for", type=int, nargs='*', default=None)

    args = parser.parse_args()

    overwrite = True
    redges = np.hstack([np.arange(0, 5, 1),
                        np.arange(7, 30, 3),
                        np.arange(31, 155, 5)])
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    if args.quantiles_clustering is None:
        quantiles_clustering = list(range(args.nquantiles))
    else:
        quantiles_clustering = args.quantiles_clustering

    # HOD configuration
    dataset = 'bossprior'
    config_fn = Path(args.hod_config_dir, f'hod_config_{dataset}.yaml')
    logger.info(f'Loading HOD configuration from {config_fn}')
    config = yaml.safe_load(open(config_fn))

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)
    redshift = args.redshift

    for cosmo in range(args.start_cosmo, args.start_cosmo + args.n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        # calculate distortion parameters
        q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
        q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
        q = q_perp**(2/3) * q_para**(1/3)
        logger.info(f'q_perp = {q_perp:.3f}, q_para = {q_para:.3f}, q = {q:.3f}')

        hods_dir = Path(args.hod_params_dir, dataset)
        hods_fn = hods_dir / f'hod_parameters_{dataset}_c{cosmo:03}.csv'
        logger.info(f'Loading HOD parameters from {hods_fn}')
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(args.start_phase, args.start_phase + args.n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)

            for hod in range(args.start_hod, args.start_hod + args.n_hod):
                if not overwrite:
                    # if output files exist, skip to next iteration
                    cross_fn = Path(
                        args.outdir / f'ds_cross_multipoles/HOD/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/',
                        f'ds_cross_multipoles_{args.split}split_{args.filter_shape.lower()}'\
                        f'_Rs{args.smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                    )
                    auto_fn = Path(
                        args.outdir / f'ds_auto_multipoles/HOD/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/',
                        f'ds_auto_multipoles_{args.split}split_{args.filter_shape.lower()}'\
                        f'_Rs{args.smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                    )
                    if os.path.exists(cross_fn) and os.path.exists(auto_fn):
                        logger.info(f'c{cosmo:03} ph{phase:03} hod{hod} already exists')
                        continue

                hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, args.nthreads)

                if args.save_hod:
                    output_dir = Path(args.outdir / f'HOD/{dataset}/',
                                     f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(output_dir,
                                     f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}.fits')
                    output_mock(hod_dict, newBall, output_fn, 'LRG',)

                x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(hod_dict)

                cross_los = []
                auto_los = []
                for los in ['x', 'y', 'z']:
                    if args.split == 'z':
                        xpos = x_rsd if los == 'x' else x
                        ypos = y_rsd if los == 'y' else y
                        zpos = z_rsd if los == 'z' else z
                    else:
                        xpos, ypos, zpos = x, y, z

                    data_positions = np.c_[xpos, ypos, zpos]

                    data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                                q_perp=q_perp, q_para=q_para)
                    boxsize_ap = np.array(get_distorted_box(boxsize=args.boxsize, q_perp=q_perp, q_para=q_para,
                                                            los=los))
                    boxcenter_ap = boxsize_ap / 2

                    start_time = time.time()
                    quantiles_ap, quantiles_idx, density = density_split(
                        data_positions=data_positions_ap, boxsize=boxsize_ap,
                        boxcenter=boxcenter_ap, cellsize=args.cellsize, seed=phase,
                        filter_shape=args.filter_shape, smoothing_radius=args.smoothing_radius,
                        nquantiles=args.nquantiles, nquery=args.nquery, nthreads=args.nthreads,)
                    logger.info(f'Density split took {time.time() - start_time:.2f} seconds')

                    if args.save_density:
                        cout = {'density': density, 'quantiles_idx': quantiles_idx}
                        output_dir = Path(args.outdir, 'density_pdf/HOD/{dataset}/',
                                          f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(output_dir,
                                         f'density_pdf_{args.split}split_{args.filter_shape.lower()}_NQ{args.nquantiles}'\
                                         f'_Rs{args.smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy')
                        logger.info(f'Saving density pdf to {output_fn}')
                        np.save(output_fn, cout)

                    if args.save_clustering:
                        logger.info(f'Computing cross-correlation function for quantiles {quantiles_clustering}')
                        start_time = time.time()
                        cross_ds = []
                        for i in range(args.nquantiles):
                            if i in quantiles_clustering:
                                result = TwoPointCorrelationFunction(
                                    'smu', edges=edges, data_positions1=quantiles_ap[i],
                                    data_positions2=data_positions_ap, los=los,
                                    engine='corrfunc', boxsize=boxsize_ap, nthreads=args.nthreads,
                                    compute_sepsavg=False, position_type='pos',
                                )
                                s, multipoles = result(ells=(0, 2, 4), return_sep=True)
                            else:
                                multipoles = np.zeros((3, len(redges) - 1))
                            cross_ds.append(multipoles)
                        cross_los.append(cross_ds)
                        logger.info(f'CCF took {time.time() - start_time:.2f} seconds')

                        logger.info(f'Computing auto-correlation function for quantiles {quantiles_clustering}')
                        start_time = time.time()
                        auto_ds = []
                        for i in range(args.nquantiles):
                            if i in quantiles_clustering:
                                result = TwoPointCorrelationFunction(
                                    'smu', edges=edges, data_positions1=quantiles_ap[i],
                                    los=los, engine='corrfunc', boxsize=boxsize_ap, nthreads=args.nthreads,
                                    compute_sepsavg=False, position_type='pos'
                                )
                                s, multipoles = result(ells=(0, 2, 4), return_sep=True)
                            else:
                                multipoles = np.zeros((3, len(redges) - 1))
                            auto_ds.append(multipoles)
                        auto_los.append(auto_ds)
                        logger.info(f'ACF took {time.time() - start_time:.2f} seconds')

                if args.save_clustering:
                    cross_los = np.asarray(cross_los)
                    auto_los = np.asarray(auto_los)

                    cout = {'s': s, 'multipoles': cross_los}
                    output_dir = Path(args.outdir,
                                        f'ds_cross_multipoles/HOD/{dataset}/'
                                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(output_dir,
                                        f'ds_cross_multipoles_{args.split}split_{args.filter_shape.lower()}'\
                                        f'_NQ{args.nquantiles}_Rs{args.smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy')
                    logger.info(f'Saving cross-correlation function to {output_fn}')
                    np.save(output_fn, cout)

                    cout = {'s': s, 'multipoles': auto_los}
                    output_dir = Path(args.outdir,
                                        f'ds_auto_multipoles/HOD/{dataset}/',
                                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(output_dir,
                                        f'ds_auto_multipoles_{args.split}split_{args.filter_shape.lower()}'\
                                        f'_NQ{args.nquantiles}_Rs{args.smoothing_radius}_c{cosmo:03}_ph{phase:03}_hod{hod}.npy')
                    logger.info(f'Saving auto-correlation function to {output_fn}')
                    np.save(output_fn, cout)
