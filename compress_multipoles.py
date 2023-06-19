import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--start_phase', type=int, default=1)
parser.add_argument('--n_phase', type=int, default=1)
parser.add_argument('--region', type=str, default='NGC')
parser.add_argument('--statistic', type=str, default='ds_cross')
parser.add_argument('--zmin', type=float, default=0.45)
parser.add_argument('--zmax', type=float, default=0.6)
parser.add_argument('--smoothing_radius', type=float, default=10)
parser.add_argument('--nquantiles', type=int, default=5)
parser.add_argument('--data_class', type=str, default='Patchy')
parser.add_argument('--dataset', type=str, default='bossprior')

args = parser.parse_args()

phases = np.arange(args.start_phase, args.start_phase + args.n_phase)
multipoles_phases = []
for phase in phases:
    data_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/{args.statistic}_multipoles/{args.data_class}/{args.dataset}')
    if 'ds' in args.statistic:
        data_fn = data_dir / f'{args.statistic}_multipoles_{args.data_class}_{args.region}_zmin{args.zmin}'\
            f'_zmax{args.zmax}_zsplit_gaussian_NQ{args.nquantiles}_Rs{args.smoothing_radius}_ph{phase:04}.npy'
    else:
        data_fn = data_dir / f'{args.statistic}_multipoles_{args.dataset}_{args.region}_zmin{args.zmin}'\
            f'_zmax{args.zmax}_ph{phase:04}.npy'

    data = np.load(data_fn, allow_pickle=True).item()
    s = data['s']
    multipoles = data['multipoles']
    multipoles_phases.append(multipoles)
multipoles_phases = np.asarray(multipoles_phases)

cout = {'s': s, 'multipoles': multipoles_phases}
Path(args.outdir).mkdir(parents=True, exist_ok=True)
if 'ds' in args.statistic:
    output_fn = Path(args.outdir) / f'{args.statistic}_multipoles_zsplit_Rs10_{args.region.lower()}_landyszalay.npy'
else:
    output_fn = Path(args.outdir) / f'{args.statistic}_multipoles_{args.region.lower()}_landyszalay.npy'
np.save(output_fn, cout)
