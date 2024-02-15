from densitysplit.pipeline import DensitySplit
from pycorr import TwoPointCorrelationFunction, setup_logging
from utils import sky_to_cartesian, cartesian_to_sky, DistanceToRedshift
from cosmoprimo.fiducial import AbacusSummit
from pathlib import Path
import fitsio
import numpy as np
import argparse
import logging


def density_split(data_positions, randoms_positions, cellsize=5.0, seed=42,
    smoothing_radius=10, nquantiles=5, boxpad=1.1, data_weights=None,
    randoms_weights=None, data_positions_pad=None, randoms_positions_pad=None,
    data_weights_pad=None, randoms_weights_pad=None,):
    """
    Split a collection of query points in quantiles according to the local density.
    """
    if data_positions_pad is None:
        data_positions_pad = data_positions
        randoms_positions_pad = randoms_positions
        data_weights_pad = data_weights
        randoms_weights_pad = randoms_weights
    ds = DensitySplit(data_positions=data_positions_pad,
                      randoms_positions=randoms_positions_pad,
                      data_weights=data_weights_pad,
                      randoms_weights=randoms_weights_pad,
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


def read_cmass(filename, distance, zmin=0.45, zmax=0.6, is_random=False, weight_type=None):
    """Read CMASS LSS catalogues."""
    data = fitsio.read(filename)
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    redshift = data[mask]['Z']
    weights = np.ones(len(ra))
    if 'FKP' in weight_type:
        weights *= data[mask]['WEIGHT_FKP']
    if not is_random:
        if 'default' in weight_type:
            weights *= data[mask]['WEIGHT_SYSTOT'] * (data[mask]['WEIGHT_CP']
                    + data[mask]['WEIGHT_NOZ'] - 1)
    dist = distance(redshift)
    positions = sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights


logger = logging.getLogger('ds_cmass')

parser = argparse.ArgumentParser()
parser.add_argument("--zmin", type=float, default=0.45)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--nquantiles", type=int, default=5)
parser.add_argument("--smoothing_radius", type=float, default=10)
parser.add_argument("--region", type=str, default='NGC')
parser.add_argument("--save_quantiles", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_density", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_clustering", action=argparse.BooleanOptionalAction)
parser.add_argument("--use_quantiles_randoms", action=argparse.BooleanOptionalAction)
parser.add_argument("--add_redshift_padding", action=argparse.BooleanOptionalAction)
parser.add_argument("--zpad", type=float, default=0.2)
parser.add_argument("--weight_type", type=str, default=None)
parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction)
parser.add_argument("--nthreads", type=int, default=1)

args = parser.parse_args()
setup_logging()

# cosmology definitions
cosmo = AbacusSummit(0)
distance = cosmo.comoving_radial_distance
d2r = DistanceToRedshift(distance)

# some practical definitions
region = 'North' if args.region == 'NGC' else 'South'
weight_type = '' if args.weight_type is None else f'_{args.weight_type}'
padding_type = '' if not args.add_redshift_padding else f'_zpad{args.zpad}'
quantiles_randoms = []
R1R2_cross = [None] * args.nquantiles
R1R2_auto = [None] * args.nquantiles
data_positions_pad = None
randoms_positions_pad = None
data_weights_pad = None
randoms_weights_pad = None
gpu = True if args.use_gpu else False
flags = f'{args.region}_zmin{args.zmin}_zmax{args.zmax}_nquantiles{args.nquantiles}'\
        f'_smoothing{args.smoothing_radius}{weight_type}{padding_type}'

data_dir = Path('/pscratch/sd/e/epaillas/ds_boss/CMASS/')
data_fn = data_dir / f'galaxy_DR12v5_CMASS_{region}.fits.gz'
logger.info(f'Reading {data_fn}')
data_positions, data_weights = read_cmass(distance=distance, filename=data_fn,
    zmin=args.zmin, zmax=args.zmax, is_random=False, weight_type=args.weight_type)

randoms_dir = Path('/pscratch/sd/e/epaillas/ds_boss/CMASS/')
randoms_fn = randoms_dir / f'random0_DR12v5_CMASS_{region}.fits.gz'
logger.info(f'Reading {randoms_fn}')
randoms_positions, randoms_weights = read_cmass(distance=distance, filename=randoms_fn,
    zmin=args.zmin, zmax=args.zmax, is_random=True, weight_type=args.weight_type)

if args.add_redshift_padding:
    zmin_pad = args.zmin - args.zpad
    zmax_pad = args.zmax + args.zpad
    logger.info(f'Reading padded data: {data_fn}')
    data_positions_pad, data_weights_pad = read_cmass(
        filename=data_fn, distance=distance, zmin=zmin_pad, zmax=zmax_pad,
        weight_type=args.weight_type,)
    logger.info(f'Reading padded randoms: {randoms_fn}')
    randoms_positions_pad, randoms_weights_pad = read_cmass(
        filename=randoms_fn, distance=distance, zmin=zmin_pad, zmax=zmax_pad,
        weight_type=args.weight_type, is_random=True,)

logger.info('Computing density split')
quantiles_positions, quantiles_idx, density = density_split(
    data_positions=data_positions,
    randoms_positions=randoms_positions,
    data_weights=data_weights,
    randoms_weights=randoms_weights,
    data_positions_pad=data_positions_pad,
    randoms_positions_pad=randoms_positions_pad,
    data_weights_pad=data_weights_pad,
    randoms_weights_pad=randoms_weights_pad,
    boxpad=1.1, cellsize=5.0, seed=42,
    smoothing_radius=args.smoothing_radius,
    nquantiles=args.nquantiles,
)

for ds, quantile in enumerate(quantiles_positions):
    if args.use_quantiles_randoms:
        logger.info(f'Generating randoms for Q{ds} matching its n(z)')
        # distances (redshifts) are drawn from the measured quintiles
        dist, ra, dec = cartesian_to_sky(quantile)
        nrandoms = 50 * len(dist)
        idx = np.random.randint(len(dist), size=nrandoms)
        qrand_dist = dist[idx]
        # angles are drawn from the galaxy randoms
        randoms_dist, randoms_ra, randoms_dec = cartesian_to_sky(randoms_positions)
        qrand_ra = randoms_ra[idx]
        qrand_dec = randoms_dec[idx]
        quantiles_randoms.append(sky_to_cartesian(dist=qrand_dist, ra=qrand_ra, dec=qrand_dec))
    else:
        logger.info(f'Generating randoms for Q{ds} by cloning the galaxy randoms')
        # use the same randoms as for the galaxies
        quantiles_randoms.append(randoms_positions)

if args.save_quantiles:
    quantiles_positions_sky = []
    for quantile in quantiles_positions:
        quantiles_dist, quantiles_ra, quantiles_dec = cartesian_to_sky(quantile)
        quantiles_redshift = d2r(quantiles_dist)
        quantiles_positions_sky.append(
            np.c_[quantiles_ra, quantiles_dec, quantiles_redshift]
        )
    output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_quantiles/CMASS/')
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    output_fn = output_dir / f'ds_quantiles_CMASS_{flags}.npy'
    logger.info(f'Saving quantiles positions to {output_fn}')
    np.save(output_fn, quantiles_positions_sky)

    if args.use_quantiles_randoms:
        quantiles_positions_sky = []
        for quantile in quantiles_randoms:
            quantiles_dist, quantiles_ra, quantiles_dec = cartesian_to_sky(quantile)
            quantiles_redshift = d2r(quantiles_dist)
            quantiles_positions_sky.append(
                np.c_[quantiles_ra, quantiles_dec, quantiles_redshift]
            )
        output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_quantiles/CMASS/')
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        output_fn = output_dir / \
            f'ds_randoms_CMASS_{flags}.npy'
        logger.info(f'Saving quantiles randoms to {output_fn}')
        np.save(output_fn, quantiles_positions_sky)

if args.save_density:
    output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_density/CMASS/')
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    output_fn = output_dir / f'density_CMASS_{flags}.npy'
    logger.info(f'Saving density to {output_fn}')
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
                    randoms_positions1=quantiles_randoms[ds],
                    randoms_positions2=randoms_positions,
                    data_weights2=data_weights,
                    randoms_weights2=randoms_weights,
                    los='midpoint', engine='corrfunc',
                    estimator='landyszalay', compute_sepsavg=False,
                    position_type='pos', R1R2=R1R2_cross[ds],
                    nthreads=args.nthreads, gpu=gpu,
                )
                if args.use_quantiles_randoms:
                    R1R2_cross[ds] = result.R1R2
                else:
                    R1R2_cross = [result.R1R2 for _ in range(args.nquantiles)]
            else:
                result = TwoPointCorrelationFunction(
                    'smu', edges=edges,
                    data_positions1=quantiles_positions[ds],
                    randoms_positions1=quantiles_randoms[ds],
                    los='midpoint', engine='corrfunc',
                    estimator='landyszalay', compute_sepsavg=False,
                    position_type='pos', R1R2=R1R2_auto[ds],
                    nthreads=args.nthreads, gpu=gpu,
                )
                if args.use_quantiles_randoms:
                    R1R2_auto[ds] = result.R1R2
                else:
                    R1R2_auto = [result.R1R2 for _ in range(args.nquantiles)]
            s, multipoles = result(ells=(0, 2, 4), return_sep=True)
            multipoles_ds.append(multipoles)
        multipoles_ds = np.asarray(multipoles_ds)

        cout = {'s': s, 'multipoles': multipoles_ds}
        output_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/ds_{corr}_multipoles/CMASS/')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = output_dir / \
        f'ds_{corr}_multipoles_CMASS_{flags}.npy'
        logger.info(f'Saving multipoles to {output_fn}')
        np.save(output_fn, cout)

            