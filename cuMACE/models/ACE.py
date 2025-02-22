import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional, List, Any

import torch_geometric.utils
from ..tools import get_edge_vectors_and_lengths, etp_l1m1_kl2m2_2_kl3m3,irreps_sph_upto_l,irreps_upto_l,ResMLP
from ..modules import NodeEncoder
import cuequivariance as cue 
import cuequivariance_torch as cuet 
import torch_geometric 
from .ACE_reprsentation import ACE_reprsenttaion
from .interation_block import cuEquivInteraction
from torch_scatter import scatter
class ACE(nn.Module):
   def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        #cutoff: float,
        cutoff_func: nn.Module, 
        radial_basis: nn.Module,
        max_l: int,
        max_body_order = 2,
        device='cuda',
        dtype = torch.float32, 
        #timeit: bool = False,
        #keep_node_features_A: bool = False,
        #forward_features: List[str] = [],
   ):
      """
      Args:
          zs: list of atomic numbers
          n_atom_basis: number of features to describe atomic environments.
              This determines the size of each embedding vector; i.e. embeddings_dim.
          edge_coding: layer for encoding edge type
          cutoff: cutoff radius
          radial_basis: layer for expanding interatomic distances in a basis set
          n_radial_basis: number of radial embedding dimensions
          cutoff_fn: cutoff function
          cutoff: cutoff radius
          max_l: the maximum l considered in the angular basis
          max_nu: the maximum correlation order
          num_message_passing: number of message passing layers
          avg_num_neighbors: average number of neighbors per atom, used for normalization
      """
      super().__init__()
      
      self.zs = zs 
      self.n_atom_basis = n_atom_basis 
      self.max_l = max_l
      self.max_body_order = max_body_order
      self.num_elements = len(zs)
      self.node_onehot = NodeEncoder(self.zs)
      self.W_z2k = torch.nn.Linear(in_features=self.num_elements, out_features=self.n_atom_basis, bias=False)
      self.device = device 
      #self.raidal_embed = radial_basis
      
      self.cutoff_func = cutoff_func
      self.radial_basis = radial_basis
      
      self.radial_MLP_for_l = nn.ModuleList([
         torch.nn.Sequential(
                           nn.Linear(self.radial_basis.n_rbf, self.n_atom_basis),
                           nn.SiLU(),
                           nn.Linear(self.n_atom_basis, self.n_atom_basis),
                           nn.SiLU(),
                           nn.Linear(self.n_atom_basis, self.n_atom_basis)
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
      #irreps_str = ""
      #for l in range(self.max_l+1):
      #   irreps_str = irreps_str + f"{self.n_atom_basis}x{l}e"
      #   irreps_str = irreps_str + f"{self.n_atom_basis}x{l}o"
      #irreps_out = cue.Irreps("O3", irreps_str)
      irreps_out = cue.Irreps("O3", f"{self.n_atom_basis}x0e")

      # irreps_in^3 -> irreps_out with generlized CG
      self.SymContraction = cuet.SymmetricContraction(
            self.irreps_in, irreps_out, 
            contraction_degree=self.max_body_order, 
            num_elements=self.num_elements, 
            layout=cue.ir_mul, 
            dtype=dtype, 
            device=device
            )
      
      # invariant feature to energy 
      self.readout_E = torch.nn.Sequential(  
               nn.Linear(self.n_atom_basis, 1),
               )
      self.readout_E0 = torch.nn.Sequential(  
               nn.Linear(3, 1)
               )
      return 
   
   def forward(self, data: Dict[str, torch.Tensor]):
      
      positions = data['positions']
      positions.requires_grad_(True)

      sender = data['edge_index'][0]
      receiver = data['edge_index'][1]
   
      # [1.1] map Z to a vector, the node embedding 
      indices, onehot_Z = self.node_onehot(data['atomic_numbers'])

      
      # W_kz [# of nodes, n_atom_basis]
      Z_sender = self.W_z2k(onehot_Z)[sender]

      #print(f'Z_sender,size = {Z_sender.shape}')
      # initial edge attri
      edge_vec, edge_length = get_edge_vectors_and_lengths(positions, data['edge_index'], data['shifts'], normalize=True)
      cutoff =  self.cutoff_func(edge_length)
      # radial_embedding is tensor [*, # of radial basis]
      Rl = [self.radial_MLP_for_l[l](self.radial_basis(edge_length) * cutoff ) for l in range(self.max_l+1)]
      Rlm = torch.cat([ Rl[l].unsqueeze(-1).expand(-1, -1, 2*l+1) for l in range(self.max_l+1)], dim = -1)
      
      # Ylm of size (n_edge, \sum_{l=0}^max_l 2l+1)
      Ylm = self.sph(edge_vec) 
      #print(f'Rlm.size() = {Rlm.size()},Ylm.size() = {Ylm.size()}, Z_sender.shape = {Z_sender.shape}')
      
      #compose R_l, Ylm, R_l to A 
      # A^{(1)}_{i, k l_1 m_1} = \sum_{j \in \mathcal{N}(i)} R^{(1)}_{k l_1} (r_{ji}) Y_{l_1}^{m_1} (\hat{r}_{ji}) W^{(1)}_{k z_j}.
      edge_attr = torch.einsum('aki,ai,ak->aik',Rlm,Ylm,Z_sender)
      #print(f'edge_attr.size() [n_edge, dim of all lm, k] = {edge_attr.size()}')

      node_feat_A = self.MixA( scatter(src=edge_attr, index=receiver, dim=0, reduce='sum').reshape(-1, self.irreps_in.dim) )
      #print(f'node_feat_A.size = {node_feat_A.size()}')
      
      # symmetrize A^n -> B0_n   
      B =  self.SymContraction(node_feat_A, indices)

      #print(B)
      Energy_atom = self.readout_E(B).squeeze(-1) + self.readout_E0(onehot_Z).squeeze(-1)
   
      #print(f'B.size() = {B.size()}, Energy_atom.size = {Energy_atom.size()}')
      #print('Energy_atom = ',Energy_atom)
      
      Energy_bacth = scatter(src=Energy_atom, index=data['batch'], dim=0, reduce='sum')
      #print('E_bacth = ',Energy_bacth)
      #print(f'E_bacth.size() = {E_bacth.size()}')

      Energy_total = torch.sum(Energy_bacth)
      Forces = -torch.autograd.grad(
            outputs=Energy_total,
            inputs=positions,
            retain_graph=True ,
            create_graph=True  # if you need higher-order derivatives
        )[0]
      return Energy_bacth, Forces 
   
   def check_invariance(self, data: Dict[str, torch.Tensor]):
      E0,_ = self.forward(data)

      # rotate 
      from e3nn import o3
      rot = o3.rand_matrix().to(data['positions'].device)

      data['positions'] = torch.einsum('ix,yx',data['positions'], rot)
      data['shifts'] = torch.einsum('ix,yx',data['shifts'], rot)

      E0_rot, _ = self.forward(data) 

      print(E0)
      print(E0_rot)

      if (torch.max(torch.abs(E0-E0_rot))>1e-5):
         raise ValueError('Energy is not invariant!')
      else:
         print(f'Energy is invariant, err = {torch.abs(torch.abs(E0-E0_rot))}')
         
      return 

class ACE_model(nn.Module):
   def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff_func: nn.Module, 
        radial_basis: nn.Module,
        max_l: int,
        max_body_order = 2,
        device='cuda',
        dtype = torch.float32
   ):
      """
      Args:
          zs: list of atomic numbers
          n_atom_basis: number of features to describe atomic environments.
              This determines the size of each embedding vector; i.e. embeddings_dim.
          edge_coding: layer for encoding edge type
          cutoff: cutoff radius
          radial_basis: layer for expanding interatomic distances in a basis set
          n_radial_basis: number of radial embedding dimensions
          cutoff_fn: cutoff function
          cutoff: cutoff radius
          max_l: the maximum l considered in the angular basis
          max_nu: the maximum correlation order
          num_message_passing: number of message passing layers
          avg_num_neighbors: average number of neighbors per atom, used for normalization
      """
      super().__init__()

      self.zs = zs 
      self.num_elements = len(zs)
      self.atom_onehot =  NodeEncoder(self.zs)
      self.n_atom_basis = n_atom_basis 
      self.max_l = max_l
      self.max_body_order = max_body_order
      self.device = device 

      self.irreps_out = cue.Irreps("O3", f"{self.n_atom_basis}x0e")
      self.B_feature = ACE_reprsenttaion(irreps_out = self.irreps_out, 
                                         atom_onehot = self.atom_onehot,
                                         n_atom_basis = n_atom_basis,
                                         cutoff_func = cutoff_func,
                                         radial_basis = radial_basis,
                                         max_l = max_l,
                                         max_body_order = max_body_order, 
                                         device = device,
                                         dtype = dtype)
      self.readout_E = nn.Linear(self.n_atom_basis, 1)
      self.readout_E0 = nn.Linear(self.num_elements, 1)
               
      return 
   
   def forward(self, data: Dict[str, torch.Tensor]):
      
      data['positions'].requires_grad_(True)
      

      sender = data['edge_index'][0]
      receiver = data['edge_index'][1]
   
      # [1.1] map Z to a vector, the node embedding 
      indices, onehot_Z = self.atom_onehot(data['atomic_numbers'])

      B,_,_ =  self.B_feature(data)

      #print(B)
      Energy_atom = self.readout_E(B).squeeze(-1) + self.readout_E0(onehot_Z).squeeze(-1)
   
      Energy_bacth = scatter(src=Energy_atom, index=data['batch'], dim=0,reduce='sum')

      Energy_total = torch.sum(Energy_bacth)
      Forces = -torch.autograd.grad(
            outputs=Energy_total,
            inputs=data['positions'],
            retain_graph=True ,
            create_graph=True  
        )[0]
      return Energy_bacth, Forces 
   
   def check_invariance(self, data: Dict[str, torch.Tensor]):
      E0,_ = self.forward(data)

      # rotate 
      from e3nn import o3
      rot = o3.rand_matrix().to(data['positions'].device)

      data['positions'] = torch.einsum('ix,yx',data['positions'], rot)
      data['shifts'] = torch.einsum('ix,yx',data['shifts'], rot)

      E0_rot, _ = self.forward(data) 

      print(E0)
      print(E0_rot)

      if (torch.max(torch.abs(E0-E0_rot))>1e-5):
         raise ValueError('Energy is not invariant!')
      else:
         print(f'Energy is invariant, err = {torch.abs(torch.abs(E0-E0_rot))}')
         
      return 

class MACE_model(nn.Module):
   def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff_func: nn.Module, 
        radial_basis: nn.Module,
        max_l: int,
        max_body_order: int= 2,
        max_L: int=0,  
        device='cuda',
        dtype = torch.float32,
        dropout_ratio = 0.1, 
        hidden_dim: List[int] = [32,32,32],
        ):
      """
      Args:
          zs: list of atomic numbers
          n_atom_basis: number of features to describe atomic environments.
              This determines the size of each embedding vector; i.e. embeddings_dim.
          edge_coding: layer for encoding edge type
          cutoff: cutoff radius
          radial_basis: layer for expanding interatomic distances in a basis set
          n_radial_basis: number of radial embedding dimensions
          cutoff_fn: cutoff function
          cutoff: cutoff radius
          max_l: the maxium rotational order for spherical harmonics 
          max_body_order: the maximum correlation order
          max_L: the maxium rotational order for B features 
          num_message_passing: number of message passing layers
          avg_num_neighbors: average number of neighbors per atom, used for normalization
      """
      super().__init__()

      self.zs = zs 
      self.num_elements = len(zs)
      self.atom_onehot =  NodeEncoder(self.zs)
      
      self.n_atom_basis = n_atom_basis 
      self.max_l = max_l
      self.max_L = max_L
      self.max_body_order = max_body_order
      self.device = device 

      self.h0 = nn.Linear(in_features=self.num_elements, out_features=self.n_atom_basis)

      # irreps 
      self.irreps_B = irreps_upto_l(self.max_L, multiplicity=self.n_atom_basis)
      self.irreps_A = irreps_upto_l(self.max_l, multiplicity=self.n_atom_basis)
      self.irreps_sph = irreps_sph_upto_l(self.max_l)

      print(f"irreps_A = {self.irreps_A}")
      print(f"irreps_B = {self.irreps_B}")
      print(f"irreps_sph = {self.irreps_sph}")

      self.B_feature = ACE_reprsenttaion(irreps_out = self.irreps_B, 
                                         atom_onehot = self.atom_onehot,
                                         n_atom_basis = n_atom_basis,
                                         cutoff_func = cutoff_func,
                                         radial_basis = radial_basis,
                                         max_l = max_l,
                                         max_body_order = max_body_order, 
                                         device = device,
                                         dtype = dtype,
                                         hidden_dim = hidden_dim)
      # equivariant linear mixing 
      self.ELinear_h0_2_h1 = cuet.Linear(irreps_in=f'{self.n_atom_basis}x0e',
                  irreps_out=self.irreps_B,
                  layout=cue.ir_mul)
      self.ELinear_m1_2_h1 = cuet.Linear(irreps_in=self.irreps_B,
                  irreps_out=self.irreps_B,
                  layout=cue.ir_mul)
      self.ELinear_h1_2_h1p = cuet.Linear(irreps_in=self.irreps_B,
                  irreps_out=self.irreps_B,
                  layout=cue.ir_mul)
      
      self.etp_edge = etp_l1m1_kl2m2_2_kl3m3(self.irreps_sph, self.irreps_B, self.irreps_A, internal_weight=False)
      self.radial_MLP =torch.nn.Sequential(
                           nn.Linear(radial_basis.n_rbf, hidden_dim[0]),
                           nn.SiLU(),
                           nn.Linear(hidden_dim[0], hidden_dim[1]),
                           nn.SiLU(),
                           nn.Linear(hidden_dim[1], hidden_dim[2]),
                           #nn.Dropout(p=dropout_ratio),  #10% drop out 
                           nn.SiLU(),
                           nn.Linear(hidden_dim[2], self.etp_edge.weight_numel)
                           )
                         
      self.ELinear_A2_mixing = cuet.Linear(irreps_in=self.irreps_A,
                  irreps_out=self.irreps_A,
                  layout=cue.ir_mul)
      # sym contract 
      # irreps_out^3 -> irreps_out with generlized CG
      self.SymContraction = cuet.SymmetricContraction(
            self.irreps_A, self.irreps_B, 
            contraction_degree=self.max_body_order, 
            num_elements=self.num_elements, 
            layout=cue.ir_mul, 
            dtype=dtype, 
            device=device
            )
      
      # equivariant linear mixing 
      self.ELinear_h1_2_h2 = cuet.Linear(irreps_in=self.irreps_B,
                  irreps_out=self.irreps_B,
                  layout=cue.ir_mul)
      
      self.ELinear_m2_2_h2 = cuet.Linear(irreps_in=self.irreps_B,
                  irreps_out=self.irreps_B,
                  layout=cue.ir_mul)

      
      self.readout_E_layer0 = nn.Linear(self.num_elements, 1)
      self.readout_E_layer1 = nn.Linear(self.n_atom_basis, 1)
      self.readout_E_layer2 = torch.nn.Sequential(  
                                 nn.Linear(self.n_atom_basis, 16),
                                 nn.SiLU(),
                                 nn.Linear(16, 1),
                                 )
      return 

   def message_passing(self,
               data: Dict[str, torch.Tensor], 
               radial_basis,
               Ylm,
               node_features, 
               ):
      # radial_embedding is tensor [*, # of radial basis]

      Rl = self.radial_MLP(radial_basis)
      edge_message = self.etp_edge.contract(Ylm, node_features ,Rl)
      return edge_message
   
   def forward(self, data: Dict[str, torch.Tensor]):
      
      data['positions'].requires_grad_(True)
      sender = data['edge_index'][0]
      receiver = data['edge_index'][1]

      # [1.1] map Z to a vector, the node embedding 
      indices, onehot_Z = self.atom_onehot(data['atomic_numbers'])

      h0 = self.h0(onehot_Z)
      # the first layer 
      m1, radial_basis, Ylm =  self.B_feature(data)
      #print(f'm0.size = {m1.size()}, Ylm shape = {Ylm.size()}')

      # linear mix the message and state 
      h1 = self.ELinear_h0_2_h1(h0) + self.ELinear_m1_2_h1(m1) 
      #print(f'h1 = {h1.shape}')
      #print(h1)
      h1p = self.ELinear_h1_2_h1p(h1)
      edge_message = self.message_passing( data, radial_basis, Ylm, h1p[sender])
      A2 = self.ELinear_A2_mixing(scatter(src=edge_message, index=receiver, dim=0, reduce='sum'))
      #A2^n -> B2
      B2 = self.SymContraction(A2, indices)
      # tensor product to form the new A feature, m0 
      h2 = self.ELinear_h1_2_h2(h1) + self.ELinear_m2_2_h2(B2) 
      #print(f'h2.size = {h2.size()}')

      # readout energy 
      E_layer0 = self.readout_E_layer0(onehot_Z).squeeze(-1)
      E_layer1 = self.readout_E_layer1(h1[:,:self.n_atom_basis]).squeeze(-1)
      E_layer2 = self.readout_E_layer2(h2[:,:self.n_atom_basis]).squeeze(-1)

      Energy_bacth = scatter(src=E_layer0+E_layer1+E_layer2, index=data['batch'], reduce='sum')

      Energy_total = torch.sum(Energy_bacth)
      Forces = -torch.autograd.grad(
            outputs=Energy_total,
            inputs=data['positions'],
            retain_graph=True ,
            create_graph=True  # if you need higher-order derivatives
        )[0]
      
      return Energy_bacth, Forces 

   def check_invariance(self, data: Dict[str, torch.Tensor]):

      E0,_ = self.forward(data)

      # rotate 
      from e3nn import o3
      rot = o3.rand_matrix().to(data['positions'].device)

      data['positions'] = torch.einsum('ix,yx',data['positions'], rot)
      data['shifts'] = torch.einsum('ix,yx',data['shifts'], rot)

      E0_rot, _ = self.forward(data) 

      print(E0)
      print(E0_rot)

      if (torch.max(torch.abs(E0-E0_rot))>1e-3):
         print(f'err = {torch.abs(torch.abs(E0-E0_rot))}')
         raise ValueError('Energy is not invariant!')
      else:
         print(f'Energy is invariant, err = {torch.abs(torch.abs(E0-E0_rot))}')
         
      return 

class MACE_model_forcefield(nn.Module):
   def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff_func: nn.Module, 
        radial_basis: nn.Module,
        max_l: int,
        max_body_order: int= 2,
        max_L: int=0,  
        layers: int = 1, 
        device='cuda',
        dtype = torch.float32,
        dropout_ratio = 0.1, 
        hidden_dim: List[int] = [32,32,32],
        hidden_irreps_type = 'sph-like', 
        atomic_scale = None,
        avg_num_neighbors = 10.0, 
        ):
      """
      Args:
          zs: list of atomic numbers
          n_atom_basis: number of features to describe atomic environments.
              This determines the size of each embedding vector; i.e. embeddings_dim.
          edge_coding: layer for encoding edge type
          cutoff: cutoff radius
          radial_basis: layer for expanding interatomic distances in a basis set
          n_radial_basis: number of radial embedding dimensions
          cutoff_fn: cutoff function
          cutoff: cutoff radius
          max_l: the maxium rotational order for spherical harmonics 
          max_body_order: the maximum correlation order
          max_L: the maxium rotational order for B features 
          num_message_passing: number of message passing layers
          avg_num_neighbors: average number of neighbors per atom, used for normalization
      """
      super().__init__()

      self.zs = zs 
      self.num_elements = len(zs)
      self.cutoff_func = cutoff_func
      self.radial_basis = radial_basis
      self.dtype = dtype
      self.dropout_ratio = dropout_ratio 
      self.atom_onehot =  NodeEncoder(self.zs)
      self.n_atom_basis = n_atom_basis 
      self.max_l = max_l
      self.max_L = max_L
      self.max_body_order = max_body_order
      self.layers = layers 
      self.device = device 
      self.hidden_irreps_type = hidden_irreps_type

      #self.atomic_scale = atomic_scale
      if atomic_scale is None:
         atomic_scale = torch.ones(self.num_elements, dtype=dtype, device=device)
      else:
         atomic_scale = torch.tensor(list(atomic_scale.values()), dtype=dtype, device=device)
      self.register_buffer(
            "atomic_scale",
            torch.tensor(atomic_scale, dtype=torch.get_default_dtype()),
        )
      self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
      
      self.initial_node_feature = nn.Linear(in_features=self.num_elements, out_features=self.n_atom_basis)
      self.sph = cuet.SphericalHarmonics(ls=[l for l in range(self.max_l+1)])


      # [1] set up input, intermediate, output irreps of the equiv interaction blocks
      self.irreps_sph = irreps_sph_upto_l(max_l=self.max_l)
      # irreps of the input and out of each interaction block 
      self.irreps_input = [ irreps_sph_upto_l(max_l=0,multiplicity=self.n_atom_basis)] # initial input are atomic embedding nx0e 
      self.irreps_intermediate = [ irreps_sph_upto_l(max_l=self.max_l, multiplicity=self.n_atom_basis) ] # 

      if self.layers==1:
         self.irreps_output = [ irreps_sph_upto_l(max_l=0, multiplicity=self.n_atom_basis) ]
      else:
         if self.hidden_irreps_type=='sph-like':
            self.irreps_output = [ irreps_sph_upto_l(max_l=self.max_L, multiplicity=self.n_atom_basis) ]   
         else:
            self.irreps_output = [ irreps_upto_l(max_l=self.max_L, multiplicity=self.n_atom_basis) ]      
      # setup the 
      for layer in range(1,self.layers):
         self.irreps_input.append(self.irreps_output[-1])
         self.irreps_intermediate.append(irreps_upto_l(max_l=self.max_l, multiplicity=self.n_atom_basis))
         if layer == self.layers-1:
            self.irreps_output.append(irreps_sph_upto_l(max_l=0, multiplicity=self.n_atom_basis))
         else:
            self.irreps_output.append(self.irreps_output[-1])

      # [2] cuEquivInteraction
      self.Interactions = nn.ModuleList([
         cuEquivInteraction(
            num_elements = self.num_elements,
            max_body_order = self.max_body_order, 
            node_irreps_in = self.irreps_input[i],
            node_irreps_intermediate = self.irreps_intermediate[i],
            node_irreps_out = self.irreps_output[i],
            edge_sph_irreps = self.irreps_sph,
            edge_length_dim = self.radial_basis.n_rbf,
            device = self.device,
            dtype = self.dtype,
            dropout_ratio = self.dropout_ratio,
            hidden_dim = hidden_dim, 
            avg_num_neighbors = avg_num_neighbors) 
            for i in range(self.layers)
            ])
      
      # [3] energyreadout 
      readout = [] 
      for i in range(self.layers):
         if i==self.layers-1:
            readout.append(ResMLP([self.n_atom_basis,16,1]))
         else:
            readout.append(nn.Linear(self.n_atom_basis, 1))
            
      self.E_readout = nn.ModuleList(readout)
      self.E_for_atomtype = nn.Linear(self.num_elements, 1)
      self.scale_for_atomtype = nn.Linear(self.num_elements, 1)

      return 

   def forward(self, data: Dict[str, torch.Tensor]):
      
      data['positions'].requires_grad_(True)
      sender = data['edge_index'][0]
      receiver = data['edge_index'][1]

      # [1] initial node, edge features 
      indices, node_attrs = self.atom_onehot(data['atomic_numbers'])
      # 
      node_feats = self.initial_node_feature(node_attrs)
      edge_vec, edge_length = get_edge_vectors_and_lengths(data['positions'], data['edge_index'], data['shifts'], normalize=True)
      cutoff =  self.cutoff_func(edge_length)
      #
      edge_feats = self.radial_basis(edge_length) * cutoff
      #
      edge_attrs = self.sph(edge_vec) 

      # readout energy 
      Energy = self.E_for_atomtype(node_attrs).squeeze(-1)
      atomic_scale = self.scale_for_atomtype(node_attrs).squeeze(-1)
      for layer in range(self.layers):
         node_feats = self.Interactions[layer](
            sender,
            receiver, 
            indices, 
            node_attrs,
            node_feats,
            edge_feats,
            edge_attrs
         )
         #x = self.E_readout[layer](node_feature[:,:self.n_atom_basis])
         #print(f'x.size = {x.size()}, Energy.size = {Energy.size()}')
         Energy = Energy + self.E_readout[layer](node_feats[:,:self.n_atom_basis]).squeeze(-1)
      Energy = Energy * atomic_scale[indices] 

      Energy_bacth = scatter(src=Energy, index=data['batch'], dim=0, reduce='sum')
      Energy_total = torch.sum(Energy_bacth)
      Forces = -torch.autograd.grad(
            outputs=Energy_total,
            inputs=data['positions'],
            retain_graph=True ,
            create_graph=True  # if you need higher-order derivatives
        )[0]
      
      return Energy_bacth, Forces 

   def check_invariance(self, data: Dict[str, torch.Tensor]):

      E0,_ = self.forward(data)
      # rotate 
      from e3nn import o3
      rot = o3.rand_matrix().to(data['positions'].device)
      data['positions'] = torch.einsum('ix,yx',data['positions'], rot)
      data['shifts'] = torch.einsum('ix,yx',data['shifts'], rot)
      E0_rot, _ = self.forward(data) 

      print(E0)
      print(E0_rot)
      if (torch.max(torch.abs(E0-E0_rot))>1e-3):
         print(f'err = {torch.abs(torch.abs(E0-E0_rot))}')
         raise ValueError('Energy is not invariant!')
      else:
         print(f'Energy is invariant, err = {torch.abs(torch.abs(E0-E0_rot))}')
      return 
   
