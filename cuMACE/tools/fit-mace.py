import sys
sys.path.append('../../')
import cuMACE
import torch.nn as nn
import torch
import sys 
import argparse


args=cuMACE.tasks.cuMACE_argparse()

# Now assign them back to local variables (if you want shorter names)
prefix=args.prefix
save_dir =args.save_dir 
train_set=args.train_set 
max_l = args.max_l
max_L = args.max_L
layers = args.layers
max_body_order = args.max_body_order
cutoff = args.cutoff
n_atom_basis = args.n_atom_basis
hidden_dim = args.hidden_dim
avg_num_neighbors = args.avg_num_neighbors
hidden_irreps_type = args.hidden_irreps_type
energyscale = args.energyscale
energy2mev = args.energy2mev
force2mevA = args.force2mevA
energy_weight = args.energy_weight 
force_weight = args.force_weight
batch_size = args.batch_size
trainable_rbf = args.trainable_rbf
weight_decay = args.weight_decay
lr_start = args.lr_start
lr_stop = args.lr_stop
atomic_energies = args.atomic_energies
atomic_scale = args.atomic_scale
#
seed = args.seed
valid_fraction = args.valid_fraction
patience = args.patience
num_epochs = args.num_epochs
n_rbf = args.n_rbf
# You can now use these variables in your code
print("-------------------------------------------")
print("=== input Variables ===")
print("prefix:", prefix)
print("save_dir:", save_dir)
print("train_set:", train_set)
print("max_l (of Ylm & A):", max_l)
print("max_L (of B):", max_L)
print("irreps type (of B):", hidden_irreps_type)
print("layers:", layers)
print("max_body_order (A^n):", max_body_order)
print("cutoff (Ang):", cutoff)
print("n_atom_basis:", n_atom_basis)
print("hidden_dim:", hidden_dim)
print("energyscale:", energyscale)
print("energy2mev:", energy2mev)
print("force2mevA:", force2mevA)
print("batch_size:", batch_size)
print("trainable_rbf:", trainable_rbf)
print("weight_decay:", weight_decay)
print("lr_start:", lr_start)
print("lr_stop:", lr_stop)
print("atomic_energies:", atomic_energies)
print("atomic_scale:", atomic_scale)
print("-------------------------------------------")

zs = list(atomic_energies.keys())

collection = cuMACE.tasks.get_dataset_from_xyz(train_path=train_set,
                                 valid_fraction=valid_fraction,
                                 seed=seed,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces'},
                                 atomic_energies=atomic_energies, # avg
                                 energyscale = energyscale,  
                                 )
train_loader = cuMACE.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )
valid_loader = cuMACE.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=10,
                              )

device = 'cuda'
hidden = f"{hidden_dim[0]}"
for i in range(1,len(hidden_dim)):
    hidden = hidden + f"{hidden_dim[i]}" 
radial_basis = cuMACE.modules.BesselRBF(cutoff=cutoff, n_rbf=n_rbf, trainable=False)
cutoff_func = cuMACE.modules.PolynomialCutoff(cutoff=cutoff)

model = cuMACE.models.MACE_model_forcefield( zs = zs, 
                           n_atom_basis=n_atom_basis,
                           cutoff_func = cutoff_func, 
                           radial_basis = radial_basis, 
                           max_l = max_l, 
                           max_body_order = max_body_order,
                           max_L = max_L,
                           layers = layers, 
                           hidden_dim = hidden_dim,
                           hidden_irreps_type = hidden_irreps_type,
                           atomic_scale = atomic_scale,
                           avg_num_neighbors =   avg_num_neighbors, 
                           )

model.to(device)
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print("-------------------------------------------")
print("Total number of parameters = ", total_params)
print("-------------------------------------------")
prefix = prefix+f'_MACE_layers={layers}-L={max_L}-'+hidden_irreps_type+f'-l={max_l}-b={max_body_order}-n={n_atom_basis}-rc={cutoff}-bs={batch_size}-hd='+hidden+f'-lr={lr_start:.1e}-{lr_stop:.1e}-L2={weight_decay:.1e}-trainable_rbf={trainable_rbf}-params={total_params}'

# Define loss function and optimizer
energy_loss_fn = nn.MSELoss()
force_loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start, amsgrad = True, weight_decay=weight_decay)

#lr scheduler 
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=patience )

# setup Trainer 
trainer = cuMACE.tasks.Trainer(                 
                  model = model,
                  optimizer = optimizer,
                  scheduler = scheduler, 
                  energy_loss_fn = energy_loss_fn,
                  force_loss_fn = energy_loss_fn,
                  multiplier = {'energy': energy_weight, 'forces':force_weight},
                  energy2mev = energy2mev,
                  force2mevA= force2mevA, 
                  device='cuda')

# load model, it will detect whether it exists 
trainer.load_best_model(save_dir, prefix)

# training process 
trainer.fit_ema(train_loader=train_loader,
            valid_loader = valid_loader,
            num_epochs = num_epochs,
            save_dir = save_dir, 
            prefix = prefix, 
            lr_threshold = lr_stop,
            )
