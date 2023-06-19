from densitysplit.pipeline import DensitySplit
from pycorr import TwoPointCorrelationFunction, setup_logging
from pyrecon import utils
from cosmoprimo.fiducial import Cosmology
from cosmoprimo.utils import DistanceToRedshift
from pathlib import Path
import numpy as np
import argparse
import time


def read_patchy(data_fn, zmin=0.45, zmax=0.6, is_random=False,
    add_fkp=False, P0=2e4):
    """Read Patchy mocks."""
    data = np.genfromtxt(data_fn)
    mask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    ra = data[mask, 0]
    dec = data[mask, 1]
    redshift = data[mask, 2]
    weights = np.ones(len(ra))
    if add_fkp:
        if is_random:
            weights *= 1 / (1 + P0 * data[mask, 3])
        else:
            weights *= 1 / (1 + P0 * data[mask, 4])
    if is_random:
        weights *= (data[mask, 5] * data[mask, 6])
    else:
        weights *= (data[mask, 6] * data[mask, 7])
    return ra, dec, redshift, weights


parser = argparse.ArgumentParser()
parser.add_argument('--start_phase', type=int, default=1)
parser.add_argument('--n_phase', type=int, default=1)
parser.add_argument("--zmin", type=float, default=0.45)
parser.add_argument("--zmax", type=float, default=0.6)
parser.add_argument("--hemisphere", type=str, default='NGC')
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
phases = list(range(args.start_phase, args.start_phase + args.n_phase))
R1R2 = None

randoms_dir = Path('/pscratch/sd/e/epaillas/ds_boss/Patchy/')
randoms_fn = randoms_dir / f'Patchy-Mocks-Randoms-DR12{args.hemisphere}-COMPSAM_V6C_x50.dat'
randoms_ra, randoms_dec, randoms_redshift, randoms_weights = read_patchy(
    randoms_fn, zmin=args.zmin, zmax=args.zmax, is_random=True, add_fkp=args.add_fkp)
randoms_dist = cosmo.comoving_radial_distance(randoms_redshift)
randoms_positions = utils.sky_to_cartesian(
    dist=randoms_dist, ra=randoms_ra, dec=randoms_dec)

for phase in phases:
    start_time = time.time()
    data_dir = Path('/pscratch/sd/e/epaillas/ds_boss/Patchy/')
    data_fn = data_dir / f'Patchy-Mocks-DR12{args.hemisphere}-COMPSAM_V6C_{phase:04}.dat'
    data_ra, data_dec , data_redshift, data_weights = read_patchy(
        data_fn, zmin=args.zmin, zmax=args.zmax, is_random=False, add_fkp=args.add_fkp)
    data_dist = cosmo.comoving_radial_distance(data_redshift)
    data_positions = utils.sky_to_cartesian(dist=data_dist, ra=data_ra, dec=data_dec)

    redges = np.hstack([np.arange(0, 5, 1), np.arange(7, 30, 3), np.arange(31, 155, 5)])
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    result = TwoPointCorrelationFunction(
        'smu', edges=edges,
        data_positions1=data_positions,
        randoms_positions1=randoms_positions,
        los='midpoint', engine='corrfunc', nthreads=256,
        estimator='landyszalay', compute_sepsavg=False,
        position_type='pos', R1R2=R1R2,
    )
    R1R2 = result.R1R2
    s, multipoles = result(ells=(0, 2, 4), return_sep=True)

    cout = {'s': s, 'multipoles': multipoles}
    output_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/tpcf_multipoles/Patchy/')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.add_fkp:
        output_fn = output_dir / \
            f'tpcf_multipoles_Patchy_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
            f'_FKP_ph{phase:04}.npy'
    else:
        output_fn = output_dir / \
            f'tpcf_multipoles_Patchy_{args.hemisphere}_zmin{args.zmin}_zmax{args.zmax}'\
            f'_ph{phase:04}.npy'
    np.save(output_fn, cout)

print(f'Phase {phase} done in {time.time() - start_time:.2f} s')
            
