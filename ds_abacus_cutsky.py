from densitysplit.pipeline import DensitySplit
from pycorr import TwoPointCorrelationFunction, setup_logging
from pyrecon import utils
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.utils import DistanceToRedshift
from pathlib import Path
import logging
import numpy as np
import argparse
import time

def density_split(data_positions, randoms_positions, cellsize=5.0,
    seed=42, smoothing_radius=10, nquantiles=5, boxpad=1.1,
    data_weights=None, random_weights=None):
    """Split random query points according to the local density."""
    ds = DensitySplit(data_positions=data_positions,
                      randoms_positions=randoms_positions,
                      data_weights=data_weights,
                      randoms_weights=random_weights,
                      cellsize=cellsize, boxpad=boxpad)
    nsample = 5 * len(data_positions)
    np.random.seed(seed=seed)
    idx = np.random.randint(len(randoms_positions), size=nsample)
    sampling_positions = randoms_positions[idx]
    density = ds.get_density_mesh(smoothing_radius=smoothing_radius,
                                  sampling_positions=sampling_positions,)
    quantiles, quantiles_idx = ds.get_quantiles(
        nquantiles=nquantiles, return_idx=True)
    return quantiles, quantiles_idx, density

def read_cutsky(data_fn, zmin=0.45, zmax=0.6, is_random=False,
    add_fkp=False, P0=2e4):
    """Read Patchy mocks."""
    data = np.genfromtxt(data_fn)
    mask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    ra = data[mask, 0]
    dec = data[mask, 1]
    redshift = data[mask, 2]
    weights = np.ones(len(ra))
    if add_fkp:
        raise NotImplementedError('FKP weights not implemented yet.')
    return ra, dec, redshift, weights

logger = logging.getLogger('ds_abacus_cutsky')


parser = argparse.ArgumentParser()
parser.add_argument('--start_phase', type=int, default=0)
parser.add_argument('--n_phase', type=int, default=1)
parser.add_argument('--start_hod', type=int, default=0)
parser.add_argument('--n_hod', type=int, default=1)
parser.add_argument("--start_cosmo", type=int, default=0)
parser.add_argument("--n_cosmo", type=int, default=1)
parser.add_argument("--zmin", type=float, default=0.45)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--nquantiles", type=int, default=5)
parser.add_argument("--smoothing_radius", type=float, default=10)
parser.add_argument("--hemisphere", type=str, default='NGC')
parser.add_argument("--save_quantiles", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_density", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_clustering", action=argparse.BooleanOptionalAction)
parser.add_argument('--add_fkp', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
setup_logging()

fid_cosmo = AbacusSummit(0)
phases = list(range(args.start_phase, args.start_phase + args.n_phase))
hods = list(range(args.start_hod, args.start_hod + args.n_hod))
cosmos = list(range(args.start_cosmo, args.start_cosmo + args.n_cosmo))
R1R2 = None

randoms_dir = Path('/pscratch/sd/c/cuesta/ds_boss/HOD/bossprior/randoms/')
randoms_fn = randoms_dir / f'randoms_cutsky.txt'
logger.info(f'Reading randoms: {randoms_fn}')
randoms_ra, randoms_dec, randoms_redshift, randoms_weights = read_cutsky(
    randoms_fn, zmin=args.zmin, zmax=args.zmax, is_random=True, add_fkp=args.add_fkp)
randoms_dist = fid_cosmo.comoving_radial_distance(randoms_redshift)
randoms_positions = utils.sky_to_cartesian(
    dist=randoms_dist, ra=randoms_ra, dec=randoms_dec)

for phase in phases:
    for hod in hods:
        for cosmo in cosmos:
            start_time = time.time()
            data_dir = Path(f'/pscratch/sd/c/cuesta/ds_boss/HOD/bossprior/AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500')
            data_fn = data_dir / f'AbacussSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}_cutsky.txt'
            logger.info(f'Reading data: {data_fn}')
            data_ra, data_dec , data_redshift, data_weights = read_cutsky(
                data_fn, zmin=args.zmin, zmax=args.zmax, is_random=False, add_fkp=args.add_fkp)
            data_dist = fid_cosmo.comoving_radial_distance(data_redshift)
            data_positions = utils.sky_to_cartesian(dist=data_dist, ra=data_ra, dec=data_dec)

            logger.info('Computing density split')
            quantiles_positions, quantiles_idx, density = density_split(
                data_positions=data_positions,
                randoms_positions=randoms_positions,
                boxpad=1.1,
                cellsize=5.0,
                seed=42,
                smoothing_radius=args.smoothing_radius,
                nquantiles=args.nquantiles,
            )

            if args.save_quantiles:
                d2r = DistanceToRedshift(fid_cosmo.comoving_radial_distance)
                quantiles_positions_sky = []
                for quantile in quantiles_positions:
                    quantiles_dist, quantiles_ra, quantiles_dec = utils.cartesian_to_sky(quantile)
                    quantiles_redshift = d2r(quantiles_dist)
                    quantiles_positions_sky.append(
                        np.c_[quantiles_ra, quantiles_dec, quantiles_redshift]
                    )
                output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_quantiles/abacus_cutsky/')
                Path.mkdir(output_dir, parents=True, exist_ok=True)
                output_fn = output_dir / \
                    f'ds_quantiles_abacus_cutsky_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                    f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_ph{phase:04}.npy'
                np.save(output_fn, quantiles_positions_sky)

            if args.save_density:
                output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_density/abacus_cutsky/')
                Path.mkdir(output_dir, parents=True, exist_ok=True)
                output_fn = output_dir / \
                    f'ds_density_abacus_cutsky_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                    f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_ph{phase:04}.npy'
                cout = {'density': density, 'quantiles_idx': quantiles_idx}
                np.save(output_fn, cout)

            if args.save_clustering:
                logger.info('Computing clustering')
                redges = np.hstack([np.arange(0, 5, 1), np.arange(7, 30, 3), np.arange(31, 155, 5)])
                muedges = np.linspace(-1, 1, 241)
                edges = (redges, muedges)

                for corr in ['cross', 'auto']:
                    multipoles_ds = []
                    for ds in range(5):
                        if corr == 'cross':
                            result = TwoPointCorrelationFunction(
                                'smu', edges=edges,
                                data_positions1=quantiles_positions[ds],
                                data_positions2=data_positions,
                                randoms_positions1=randoms_positions,
                                randoms_positions2=randoms_positions,
                                data_weights2=data_weights,
                                los='midpoint', engine='corrfunc', nthreads=256,
                                estimator='landyszalay', compute_sepsavg=False,
                                position_type='pos', R1R2=R1R2,
                            )
                        else:
                            result = TwoPointCorrelationFunction(
                                'smu', edges=edges,
                                data_positions1=quantiles_positions[ds],
                                randoms_positions1=randoms_positions,
                                los='midpoint', engine='corrfunc', nthreads=256,
                                estimator='landyszalay', compute_sepsavg=False,
                                position_type='pos', R1R2=R1R2,
                            )
                        R1R2 = result.R1R2
                        s, multipoles = result(ells=(0, 2, 4), return_sep=True)
                        multipoles_ds.append(multipoles)
                    multipoles_ds = np.asarray(multipoles_ds)

                    cout = {'s': s, 'multipoles': multipoles_ds}
                    output_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/ds_{corr}_multipoles/abacus_cutsky/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    if args.add_fkp:
                        output_fn = output_dir / \
                            f'ds_{corr}_multipoles_abacus_cutsky_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                            f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_FKP_ph{phase:04}.npy'
                    else:
                        output_fn = output_dir / \
                            f'ds_{corr}_multipoles_abacus_cutsky_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                            f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_ph{phase:04}.npy'
                    logger.info(f'Saving to disk: {output_fn}')
                    np.save(output_fn, cout)

            print(f'Phase {phase} done in {time.time() - start_time:.2f} s')
                        