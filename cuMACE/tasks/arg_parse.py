import argparse
import ast 

def cuMACE_argparse():
    parser = argparse.ArgumentParser(
        description="Parse command line arguments to set variables."
    )

    parser.add_argument(
        "--train_set",
        type=str,
        default='train.xyz',
        help="train_set (default: train.xyz)"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default='cuMACE',
        help="prefix (default: cuMACE)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./save/',
        help="save_dir (default: ./save/)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="layers (default: 1)"
    )
    parser.add_argument(
        "--max_l",
        type=int,
        default=2,
        help="Maximum l (default: 2)"
    )
    parser.add_argument(
        "--max_L",
        type=int,
        default=0,
        help="Maximum L (default: 0)"
    )
    parser.add_argument(
        "--hidden_irreps_type",
        type=str,
        default='sph-like',
        help="hidden_irreps_type (default: sph-like)"
    )
    parser.add_argument(
        "--max_body_order",
        type=int,
        default=3,
        help="Max body order (default: 3)"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Cutoff value (default: 5.0)"
    )

    parser.add_argument(
        "--n_atom_basis",
        type=int,
        default=8,
        help="Number of atom basis (default: 8)"
    )
    parser.add_argument(
        "--hidden_dim",
        nargs="+",  # allows multiple values after --hidden_dim
        type=int,
        default=[32, 32, 32],
        help="Hidden dimension(s) (default: 32 32 32)"
    )
    parser.add_argument(
        "--avg_num_neighbors",
        type=float,
        default=10.0,
        help="avg_num_neighbors (default: 10.0)"
    )
    parser.add_argument(
        "--atomic_energies",
        type=ast.literal_eval,  # Use literal_eval to safely evaluate Python literals
        default={},
        help="Dictionary of atomic energies in Python literal format. "
    )
    parser.add_argument(
        "--atomic_scale",
        type=ast.literal_eval,  # Use literal_eval to safely evaluate Python literals
        default={},
        help="Dictionary of atomic energies in Python literal format. "
    )
    parser.add_argument(
        "--energyscale",
        type=float,
        default=1.0,
        help="Energy scale (default: 1.0)"
    )
    parser.add_argument(
        "--energy_weight",
        type=float,
        default=1.0,
        help="energy_weight (default: 1.0)"
    )
    parser.add_argument(
        "--force_weight",
        type=float,
        default=100.0,
        help="force_weight (default: 100.0)"
    )
        
    parser.add_argument(
        "--energy2mev",
        type=float,
        default=None,
        help="Conversion factor energy->meV (default: 1000 / energyscale)"
    )
    parser.add_argument(
        "--force2mevA",
        type=float,
        default=None,
        help="Conversion factor force->meV/A (default: 1000 / energyscale)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size (default: 5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="weight_decay (default: 1e-6)"
    )
    parser.add_argument(
        "--lr_start",
        type=float,
        default=0.01,
        help="lr_start (default: 0.01)"
    )
    parser.add_argument(
        "--lr_stop",
        type=float,
        default=1e-6,
        help="lr_stop (default: 1e-6)"
    )
    parser.add_argument(
        "--trainable_rbf",
        action="store_true",
        default=False,
        help="trainable_rbf (default: false)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=321,
        help="seed (default: 321)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="patience (default: 50)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5000,
        help="num_epochs (default: 5000)"
    )
    parser.add_argument(
        "--n_rbf",
        type=int,
        default=8,
        help="n_rbf (default: 8)"
    )
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=0.1,
        help="valid_fraction (default: 0.1)"
    )
    
    args = parser.parse_args()
    

    return args
