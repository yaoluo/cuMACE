import dataclasses
import logging
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
from ase import Atoms
from ase.io import read
from ..tools.torch_geometric import DataLoader
from ..data import AtomicData
import torch_geometric 
import torch 
__all__ = ["load_data_loader","get_dataset_from_rMD17", "get_dataset_from_xyz", "random_train_valid_split","measure_shiftscale"]

@dataclasses.dataclass
class SubsetAtoms:
    train: Atoms
    valid: Atoms 
    test: Atoms
    cutoff: float
    data_key: Dict
    atomic_energies: Dict 
    energyscale: float=1.0

def load_data_loader(
    collection: SubsetAtoms,
    data_type: str, # ['train', 'valid', 'test']
    batch_size: int,
):

    allowed_types = ['train', 'valid', 'test']
    if data_type not in allowed_types:
        raise ValueError(f"Input value must be one of {allowed_types}, got {data_type}")

    cutoff = collection.cutoff
    data_key = collection.data_key
    atomic_energies = collection.atomic_energies

    if data_type == 'train':
        loader = DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies, energyscale=collection.energyscale)
                for atoms in collection.train
            ],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif data_type == 'valid':
        loader = DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies, energyscale=collection.energyscale)
                for atoms in collection.valid
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
    elif data_type == 'test':
        loader = DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies, energyscale=collection.energyscale)
                for atoms in collection.test
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
    return loader

def get_dataset_from_rMD17( 
    name: str,
    cutoff: float,
    train_number: int = 950 , 
    valid_number: int = 50,
    seed: int = 1234,
    atomic_energies: Dict[int, float] = None) -> SubsetAtoms:

    n_extract = train_number + valid_number
    data = torch_geometric.datasets.MD17(root='./', name= name)
    # load 1000 frames with equal time difference 
    step = int(len(data) / n_extract)
    if step * n_extract > len(data):
        step -= 1
    
    all_configs = [] 
    E_list = [] 
    for i in range(n_extract):
        i_frame = i * step 
        mol = Atoms(numbers=data[i_frame].z, positions=data[i_frame].pos, pbc = False)
        mol.set_array("forces", data[i_frame].force.numpy())
        mol.info["energy"] = data[i_frame].energy[0]  # Example energy value
        all_configs.append(mol)  
        natoms = len(data[i_frame].force)
        E_list.append(data[i_frame].energy[0]/natoms)
             
    print('# of atoms = ', natoms)
    print('mean E per atoms = ',np.mean(np.asarray(E_list)))
    valid_fraction = valid_number / n_extract
    train_configs, valid_configs = random_train_valid_split(
            all_configs, valid_fraction, seed
        )
    
    data_key={'energy': 'energy', 'forces':'forces'}
    return (
        SubsetAtoms(train=train_configs, valid=valid_configs, test=valid_configs, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies)
    )

def get_dataset_from_xyz(
    train_path: str,
    cutoff: float,
    valid_path: str = None,
    valid_fraction: float = 0.1,
    test_path: str = None,
    seed: int = 1234,
    data_key: Dict[str, str] = None,
    atomic_energies: Dict[int, float] = None,
    energyscale: Dict[int, float] = None,  
) -> SubsetAtoms:
    """Load training and test dataset from xyz file"""
    all_train_configs = read(train_path, ":")


    if not isinstance(all_train_configs, list):
        all_train_configs = [all_train_configs]
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        valid_configs = read(valid_path, ":")
        if not isinstance(valid_configs, list):
            valid_configs = [valid_configs]
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        test_configs = read(test_path, ":")
        if not isinstance(test_configs, list):
            test_configs = [test_configs]
        logging.info(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetAtoms(train=train_configs, valid=valid_configs, test=test_configs, cutoff=cutoff, data_key=data_key, atomic_energies=atomic_energies, energyscale=energyscale)
    )

def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )

def measure_shiftscale(train_loader):
    Zset = [] 
    for batch in train_loader:
       data = batch.to_dict()
    
       Z = data['atomic_numbers'].numpy()
       Zset.append(Z)
    Zset = np.unique(np.asarray(Zset).astype(int))
    print('unique atomic number = ',Zset)   

    Fmean = {Z: 0 for Z in Zset}    


    Etotal = 0 
    Fset = {Z: 0 for Z in Zset}
    Nset = {Z: 0 for Z in Zset} 

    for batch in train_loader:
       data = batch.to_dict()
    
       Z = data['atomic_numbers'].numpy()
       E = data['energy'].numpy()
       F = data['forces'].numpy()
       Etotal = Etotal + np.sum(E) 
       for i in range(len(Z)):
          Nset[Z[i]] = Nset[Z[i]] + 1 
          Fmean[Z[i]] = Fmean[Z[i]] + F[i]  

    Fmean = {Z:Fmean[Z]/Nset[Z] for Z in Zset}  


    for batch in train_loader:
       data = batch.to_dict()
       Z = data['atomic_numbers'].numpy()
       F = data['forces'].numpy()
       for i in range(len(Z)):
          Fset[Z[i]] = Fset[Z[i]] + np.mean((F[i,:] - Fmean[Z[i]])**2)
    

    Scale = {Z: 0 for Z in Zset}
    for Z in Zset:
       Scale[Z] = np.sqrt(Fset[Z] / Nset[Z])
    E_mean = Etotal / np.sum(list(Nset.values())) 
    Shift = {Z: E_mean for Z in Zset}

    return Shift, Scale
