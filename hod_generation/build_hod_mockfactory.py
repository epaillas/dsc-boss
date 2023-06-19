import os
import time
import yaml
import numpy as np
import argparse
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from astropy.io.fits import Header
from abacusnbody.hod.abacus_hod import AbacusHOD
from densitysplit.pipeline import DensitySplit
from pathlib import Path
from pypower import setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.cosmology import Cosmology
from scipy.interpolate import RectBivariateSpline
from scipy import special
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def get_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict['LRG']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    return x, y, z, vx, vy, vz


def output_mock(mock_dict, fn,):
    """Save HOD catalogue to disk."""
    x, y, z, vx, vy, vz = get_positions(hod_dict)
    cout = np.c_[x, y, z, vx, vy, vz]
    print(f'nden = {len(cout)/boxsize**3} (h/Mpc)^3')
    np.savetxt(output_fn, cout)

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
    newBall.params['Lbox'] = boxsize
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=0)
    parser.add_argument("--n_phase", type=int, default=1)

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    setup_logging(level='WARNING')
    overwrite = True
    nthreads = 256
    save_mock = True
    boxsize = 2000
    redshift = 0.5

    # HOD configuration
    dataset = 'bossprior'
    config_dir = './'
    config_fn = Path(config_dir, f'hod_config_{dataset}.yaml')
    config = yaml.safe_load(open(config_fn))

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        hods_dir = Path(f'/pscratch/sd/c/cuesta/ds_boss/parameters/HOD/{dataset}/')
        hods_fn = hods_dir / f'hod_parameters_{dataset}_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)

            for hod in range(start_hod, start_hod + n_hod):
                print(f'c{cosmo:03} ph{phase:03} hod{hod}')

                hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, nthreads)

                if save_mock:
                    output_dir = Path(f'/global/homes/e/epaillas/carolscratch/ds_boss/HOD/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(
                        output_dir,
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}.txt'
                    )
                    output_mock(hod_dict, output_fn,)
