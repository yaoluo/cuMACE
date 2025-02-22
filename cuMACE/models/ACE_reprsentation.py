import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional, List, Any

import torch_geometric.utils
from ..tools import get_edge_vectors_and_lengths, etp_l1m1_kl2m2_2_kl3m3
from ..modules import NodeEncoder
import cuequivariance as cue 
import cuequivariance_torch as cuet 
import torch_geometric 


class ACE_reprsenttaion(nn.Module):
   def __init__(
        self,
        atom_onehot: nn.Module, 
        n_atom_basis: int,
        cutoff_func: nn.Module, 
        radial_basis: nn.Module,
        max_l: int,
        irreps_out: cue.Irreps,
        max_body_order = 2,
        device='cuda',
        dtype = torch.float32, 
        hidden_dim: List[int] = [32,32,32]
   ):
      """
      Args:
          n_atom_basis: number of features to describe atomic environments.
              This determines the size of each embedding vector; i.e. embeddings_dim.
          edge_coding: layer for encoding edge type
          cutoff: cutoff radius
          radial_basis: layer for expanding interatomic distances in a basis set
          n_radial_basis: number of radial embedding dimensions
          cutoff_fn: cutoff function
          cutoff: cutoff radius
          max_l: the maximum l considered in the angular basis
          max_body_order: the maximum correlation order
          irreps_out: the irreps of the output representation of the atomic structure 
      """
      super().__init__()
      

      
      self.n_atom_basis = n_atom_basis 
      self.max_l = max_l
      self.max_body_order = max_body_order
      self.num_elements = atom_onehot.num_classes
      self.atom_onehot = atom_onehot
      self.device = device 
      self.irreps_out = irreps_out 

      
      self.cutoff_func = cutoff_func
      self.radial_basis = radial_basis
      
      self.W_z2k = torch.nn.Linear(in_features=self.num_elements, out_features=self.n_atom_basis, bias=False)


      self.radial_MLP_for_l = nn.ModuleList([
         torch.nn.Sequential(
                           nn.Linear(self.radial_basis.n_rbf, hidden_dim[0]),
                           nn.SiLU(),
                           nn.Linear(hidden_dim[0], hidden_dim[1]),
                           nn.SiLU(),
                           nn.Linear(hidden_dim[1], self.n_atom_basis)
                           )
                           for i in range(self.max_l+1)])
   
      #spherical harmonic 
      self.sph = cuet.SphericalHarmonics(ls=[l for l in range(self.max_l+1)])
      
      # SymmetricContraction
      irreps_str = ""
      for l in range(self.max_l+1):
         if l%2==0:
            if irreps_str == "":
               irreps_str = f"{self.n_atom_basis}x{l}e"
            else:
               irreps_str = irreps_str + f"+{self.n_atom_basis}x{l}e"
         else:
            if irreps_str == "":
               irreps_str = f"{self.n_atom_basis}x{l}o"
            else:
               irreps_str = irreps_str + f"+{self.n_atom_basis}x{l}o"
      print(irreps_str)
      self.irreps_in = cue.Irreps("O3", irreps_str)

      self.MixA = cuet.Linear( self.irreps_in,
                               self.irreps_in,
                               layout=cue.ir_mul,
                               internal_weights = True,
                               dtype=dtype, 
                               device=device 
                              )

      # irreps_in^3 -> irreps_out with generlized CG
      self.SymContraction = cuet.SymmetricContraction(
            self.irreps_in, self.irreps_out, 
            contraction_degree=self.max_body_order, 
            num_elements=self.num_elements, 
            layout=cue.ir_mul, 
            dtype=dtype, 
            device=device
            )
      return 
   
   def forward(self, data: Dict[str, torch.Tensor]):
      
      sender = data['edge_index'][0]
      receiver = data['edge_index'][1]
   
      # [1.1] map Z to a vector, the node embedding 
      indices, onehot_Z = self.atom_onehot(data['atomic_numbers'])
      Z_node = self.W_z2k(onehot_Z)
      # W_kz [# of nodes, n_atom_basis]
      Z_sender = Z_node[sender]

      #print(f'Z_sender,size = {Z_sender.shape}')
      # initial edge attri
      edge_vec, edge_length = get_edge_vectors_and_lengths(data['positions'], data['edge_index'], data['shifts'], normalize=True)
      cutoff =  self.cutoff_func(edge_length)
      radial_basis = self.radial_basis(edge_length) * cutoff
      # radial_embedding is tensor [*, # of radial basis]
      Rl = [self.radial_MLP_for_l[l]( radial_basis ) for l in range(self.max_l+1)]
      Rlm = torch.cat([ Rl[l].unsqueeze(-1).expand(-1, -1, 2*l+1) for l in range(self.max_l+1)], dim = -1)
      
      # Ylm of size (n_edge, \sum_{l=0}^max_l 2l+1)
      Ylm = self.sph(edge_vec) 
      #print(f'Rlm.size() = {Rlm.size()},Ylm.size() = {Ylm.size()}, Z_sender.shape = {Z_sender.shape}')
      
      #compose R_l, Ylm, R_l to A 
      # A^{(1)}_{i, k l_1 m_1} = \sum_{j \in \mathcal{N}(i)} R^{(1)}_{k l_1} (r_{ji}) Y_{l_1}^{m_1} (\hat{r}_{ji}) W^{(1)}_{k z_j}.
      edge_attr = torch.einsum('aki,ai,ak->aik',Rlm,Ylm,Z_sender)
      #print(f'edge_attr.size() [n_edge, dim of all lm, k] = {edge_attr.size()}')

      node_feat_A = self.MixA( torch_geometric.utils.scatter(src=edge_attr, index=receiver, dim=0, reduce='sum').reshape(-1, self.irreps_in.dim) )
      #print(f'node_feat_A.size = {node_feat_A.size()}')
      
      # symmetrize A^n -> B0_n   
      B =  self.SymContraction(node_feat_A, indices)

      return B, radial_basis, Ylm
   


