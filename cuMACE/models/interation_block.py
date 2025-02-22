import torch
from torch import nn
from typing import Callable, Dict, Sequence, Optional, List, Any
from ..tools import get_edge_vectors_and_lengths, etp_l1m1_kl2m2_2_kl3m3,ResMLP,SkipMLP,scatter_sum
from ..modules import NodeEncoder
import cuequivariance as cue 
import cuequivariance_torch as cuet 
#import torch_geometric 
from torch_scatter import scatter
from e3nn.util.jit import compile_mode

@compile_mode("script")
class cuEquivInteraction(nn.Module):
   def __init__(
        self,
        num_elements: int,
        max_body_order: int, 
        node_irreps_in: cue.Irreps,
        node_irreps_intermediate: cue.Irreps,
        node_irreps_out: cue.Irreps,
        edge_sph_irreps: cue.Irreps,  
        edge_length_dim: int, 
        device='cuda',
        dtype = torch.float32,
        dropout_ratio = 0.1, 
        hidden_dim: List[int] = [32,32,32],
        avg_num_neighbors=10.0,
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
          max_l: the maxium rotational order for intermediate rep, A
          max_body_order: the maximum correlation order
          max_L: the maxium rotational order for B features 
          num_message_passing: number of message passing layers
          avg_num_neighbors: average number of neighbors per atom, used for normalization
      """
      super().__init__()

      self.num_elements = num_elements
      self.max_body_order = max_body_order
      self.device = device 
      self.node_irreps_in = node_irreps_in
      self.node_irreps_intermediate = node_irreps_intermediate
      self.node_irreps_out = node_irreps_out
      self.edge_sph_irreps = edge_sph_irreps
      self.edge_length_dim = edge_length_dim
      self.dropout_ratio = dropout_ratio
      self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
      

      # equivariant linear mixing
      self.ELinear_h1_2_h1p = cuet.Linear(
                                    irreps_in=self.node_irreps_in,
                                    irreps_out=self.node_irreps_in,
                                    layout=cue.ir_mul 
                                    )
      # equivariant tensor product : Ylm x input_k -> output_k, parallelized for different multiplicity 
      self.etp_edge = etp_l1m1_kl2m2_2_kl3m3(
                                    self.edge_sph_irreps,
                                    self.node_irreps_in,
                                    self.node_irreps_intermediate, 
                                    internal_weight=False 
                                    )
      print(f'edge_length_dim = {edge_length_dim}')
      # parameters in equivariant tensor product
      neuron_dim = [edge_length_dim]
      for x in hidden_dim:
         neuron_dim.append(x)
      neuron_dim.append(self.etp_edge.weight_numel)
      #self.radial_MLP = SkipMLP(neuron_dim = neuron_dim)
      self.radial_MLP = ResMLP(neuron_dim = neuron_dim)
      # A_{ir,k} = W_{kk'}A_{ir,k'}, k loops over the multiplicity 

      self.ELinear_A_mixing = cuet.FullyConnectedTensorProduct(
            irreps_in1=self.node_irreps_intermediate,
            irreps_in2=f"{self.num_elements}x0e",
            irreps_out=self.node_irreps_intermediate,
            layout=cue.ir_mul)
      
      
      # sym contract 
      # irreps_out^3 -> irreps_out with generlized CG
      self.SymContraction = cuet.SymmetricContraction(
            self.node_irreps_intermediate, self.node_irreps_out, 
            contraction_degree=self.max_body_order, 
            num_elements=self.num_elements, 
            layout=cue.ir_mul, 
            dtype=dtype, 
            device=device
            )
      # equivariant linear mixing: L -> L, k -> k' 
      self.ELinear_m2_2_h2 = cuet.Linear(irreps_in=self.node_irreps_out,
                  irreps_out=self.node_irreps_out,
                  layout=cue.ir_mul)
      self.ELinear_h1_2_h2 = cuet.Linear(irreps_in=self.node_irreps_in,
                  irreps_out=self.node_irreps_out,
                  layout=cue.ir_mul)
      
      return 
      
   def forward( self, 
                sender: torch.tensor, 
                receiver: torch.tensor, 
                indices: torch.tensor,
                node_attrs: torch.tensor,
                node_feature_in: torch.tensor,
                edge_length_embed: torch.tensor, 
                edge_sph_embed: torch.tensor ) -> torch.Tensor:

      # node_feature_in: (ir, k) -> (ir, k')
      h1p = self.ELinear_h1_2_h1p(node_feature_in)
      # weight for each path in etp (ir1,ir2,ir3,k), k for multiplicity 
      #print(f'edge_length_embed.size = {edge_length_embed.size()}')
      Rl = self.radial_MLP(edge_length_embed)
      # R Ylm x node_feature_in -> A on the dege 
      edge_message = self.etp_edge.contract(edge_sph_embed, h1p[sender], Rl)
      # aggregate the message on the edge to the node 
      message_A = scatter(src=edge_message, index=receiver, dim=0, reduce='sum') / self.avg_num_neighbors
      linear_message_A = self.ELinear_A_mixing(message_A, node_attrs)
      # A^n -> B2, n = 1, 2,.., self.max_body_order 
      nonlinear_message_B = self.SymContraction(linear_message_A, indices)
      # residual update 
      node_feature_out = self.ELinear_h1_2_h2(node_feature_in) + self.ELinear_m2_2_h2(nonlinear_message_B) 

      return node_feature_out 
