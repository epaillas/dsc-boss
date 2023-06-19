from densitysplit.pipeline import DensitySplit
from pycorr import TwoPointCorrelationFunction, setup_logging
from pyrecon import utils
from cosmoprimo.fiducial import Cosmology
from cosmoprimo.utils import DistanceToRedshift
from pathlib import Path
import fitsio
import numpy as np
import argparse

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

def read_cmass(data_fn, zmin=0.45, zmax=0.6, is_random=False, add_fkp=False):
    """Read CMASS LSS catalogues."""
    data = fitsio.read(data_fn)
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    redshift = data[mask]['Z']
    weights = np.ones(len(ra))
    if add_fkp:
        weights *= data[mask]['WEIGHT_FKP']
    if not is_random:
        weights *= data[mask]['WEIGHT_SYSTOT'] * (data[mask]['WEIGHT_CP']
                   + data[mask]['WEIGHT_NOZ'] - 1)
    return ra, dec, redshift, weights

parser = argparse.ArgumentParser()
parser.add_argument("--zmin", type=float, default=0.45)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--nquantiles", type=int, default=5)
parser.add_argument("--smoothing_radius", type=float, default=10)
parser.add_argument("--hemisphere", type=str, default='North')
parser.add_argument("--save_quantiles", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_density", action=argparse.BooleanOptionalAction)
parser.add_argument("--save_clustering", action=argparse.BooleanOptionalAction)
parser.add_argument('--add_fkp', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
setup_logging()

cosmo = Cosmology(
    Omega_m=0.307115,
    Omega_b=0.048,
    sigma8=0.8288,
    n_s=0.9611,
    h=0.677,
    engine='class'
)

data_dir = Path('/pscratch/sd/e/epaillas/ds_boss/CMASS/')
data_fn = data_dir / f'galaxy_DR12v5_CMASS_{args.hemisphere}.fits.gz'
data_ra, data_dec , data_redshift, data_weights = read_cmass(
    data_fn, zmin=args.zmin, zmax=args.zmax, is_random=False, add_fkp=args.add_fkp)
data_dist = cosmo.comoving_radial_distance(data_redshift)
data_positions = utils.sky_to_cartesian(dist=data_dist, ra=data_ra, dec=data_dec)

randoms_dir = Path('/pscratch/sd/e/epaillas/ds_boss/CMASS/')
randoms_fn = randoms_dir / f'random0_DR12v5_CMASS_{args.hemisphere}.fits.gz'
randoms_ra, randoms_dec, randoms_redshift, randoms_weights = read_cmass(
    randoms_fn, zmin=args.zmin, zmax=args.zmax, is_random=True, add_fkp=args.add_fkp)
randoms_dist = cosmo.comoving_radial_distance(randoms_redshift)
randoms_positions = utils.sky_to_cartesian(
    dist=randoms_dist, ra=randoms_ra, dec=randoms_dec)

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
    d2r = DistanceToRedshift(cosmo.comoving_radial_distance)
    quantiles_positions_sky = []
    for quantile in quantiles_positions:
        quantiles_dist, quantiles_ra, quantiles_dec = utils.cartesian_to_sky(quantile)
        quantiles_redshift = d2r(quantiles_dist)
        quantiles_positions_sky.append(
            np.c_[quantiles_ra, quantiles_dec, quantiles_redshift]
        )
    output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_quantiles/CMASS/')
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    output_fn = output_dir / f'quantiles_DR12v5_CMASS_{args.hemisphere}.npy'
    np.save(output_fn, quantiles_positions_sky)

if args.save_density:
    output_dir = Path('/pscratch/sd/e/epaillas/ds_boss/ds_density/CMASS/')
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    output_fn = output_dir / f'density_DR12v5_CMASS_{args.hemisphere}.npy'
    cout = {'density': density, 'quantiles_idx': quantiles_idx}
    np.save(output_fn, cout)

if args.save_clustering:
    redges = np.hstack([np.arange(0, 5, 1), np.arange(7, 30, 3), np.arange(31, 155, 5)])
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    R1R2 = None
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
        output_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/ds_{corr}_multipoles/CMASS/')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if args.add_fkp:
            output_fn = output_dir / \
                f'ds_{corr}_multipoles_DR12v5_CMASS_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_FKP.npy'
        else:
            output_fn = output_dir / \
                f'ds_{corr}_multipoles_DR12v5_CMASS_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
                f'_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}.npy'
        np.save(output_fn, cout)

            