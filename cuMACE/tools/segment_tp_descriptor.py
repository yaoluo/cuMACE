import torch
import torch.nn as nn 
import cuequivariance as cue
import cuequivariance_torch as cuet
import cuequivariance.segmented_tensor_product as stp
import itertools 
from e3nn.util.jit import compile_mode
from typing import Union


def irreps_sph_upto_l(max_l:int, multiplicity: int=1):
    irreps_str = ""
    for l in range(max_l+1):
       if l%2==0:
          if irreps_str == "":
             irreps_str = f"{multiplicity}x{l}e"
          else:
             irreps_str = irreps_str + f"+{multiplicity}x{l}e"
       else:
          if irreps_str == "":
             irreps_str = f"{multiplicity}x{l}o"
          else:
             irreps_str = irreps_str + f"+{multiplicity}x{l}o"
    irreps = cue.Irreps("O3", irreps_str)
    return irreps

def irreps_upto_l(max_l:int, multiplicity:int):
    irreps_str = ""
    for l in range(max_l+1):
       if irreps_str == "":
          irreps_str = f"{multiplicity}x{l}e+{multiplicity}x{l}o"
       else:
          irreps_str = irreps_str + f"+{multiplicity}x{l}e+{multiplicity}x{l}o"
    irreps = cue.Irreps("O3", irreps_str)
    return irreps

def descriptor_channelwise_stp(irreps_in1, irreps_in2, irreps_out):
    """
    Build the descriptor for the tensor product of irreps_in1 and irreps_in2, 
    and decompose it to irreps_out using Clebsch-Gordan coefficients.

    Parameters:
    irreps_in1 (list of tuples): A list where each tuple contains a multiplicity and an irreducible representation (irrep) for the first input tensor X_{ir1,k}.
    irreps_in2 (list of tuples): A list where each tuple contains a multiplicity and an irrep for the second input tensor Y_{ir2,k}.
    irreps_out (list of tuples): A list where each tuple contains a multiplicity and an irrep for the output tensor Z_{ir3,k}.
      The multiplicity is the same for different irreps.

    Returns:
    d (SegmentedTensorProduct): The segmented tensor product object configured with the specified irreps.
    CGC (torch.Tensor): A tensor containing the Clebsch-Gordan coefficients C_{ir1,ir2,ir3}.

    Example:
         irreps_in1 = cue.Irreps("O3", "1x0e + 1x1o + 1x2o")
         irreps_in2 = cue.Irreps("O3", "1x0e + 1x1o + 1x2o")
         irreps_out = cue.Irreps("O3", "1x0e + 1x1o + 1x2o")
    """

    
    d = stp.SegmentedTensorProduct.from_subscripts("ijk,iu,ju,ku")
    # 
    d_W = stp.SegmentedTensorProduct.from_subscripts("i,ia,ia")

    for mul, ir in irreps_in1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps_in2:
        d.add_segment(2, (ir.dim, mul))
    for mul, ir in irreps_out:
        d.add_segment(3, (ir.dim, mul))

    print(d)
    N = 0
    weight = []
    CG_path = [] 
    iCG = 0 
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)), (i3, (mul3, ir3)) in itertools.product(
        enumerate(irreps_in1),enumerate(irreps_in2), enumerate(irreps_out)
    ):
        if mul1 != mul2 or mul2 != mul3:
            print(f'irreps_in1 = {irreps_in1}')
            print(f'irreps_in2 = {irreps_in2}')
            print(f'irreps_out = {irreps_out}')
            raise ValueError('multiplicity not consistent')
        
        CG = cue.O3.clebsch_gordan(ir1,ir2,ir3)
        if len(CG)!=0:
            #print(CG)
            NCG = len(CG.reshape(-1))
            N += NCG
            for x in CG.reshape(-1):
                weight.append(x)
            CG_path.append([ir1,ir2,ir3])
            #print(i1,i2,i3)
            d.add_path(None, i1, i2, i3, c=1.0)

            d_W.add_segment(0, (1,))
            d_W.add_segment(1, (1,NCG))
            d_W.add_segment(2, (1,NCG))
            #print(d_W, iCG)
            d_W.add_path(iCG,iCG,iCG,c=1.0)
            iCG += 1
    CGC = torch.tensor(weight).reshape(1,-1)
    #for 

    return d, d_W, CGC  

def descriptor_stp_l1m1_kl2m2_2_kl3m3(irreps_sph, irreps_in, irreps_out):
    """
    Build the descriptor for the tensor product of irreps_in1 and irreps_in2, 
    and decompose it to irreps_out using Clebsch-Gordan coefficients.

    Parameters:
    irreps_sph (list of tuples): A list where each tuple contains a multiplicity=1 and an irreducible representation (irrep) for the first input tensor X_{ir1,k}.
    irreps_in2 (list of tuples): A list where each tuple contains a multiplicity=M and an irrep for the second input tensor Y_{ir2,k}.
    irreps_out (list of tuples): A list where each tuple contains a multiplicity=M and an irrep for the output tensor Z_{ir3,k}.
      
    Returns:
    d (SegmentedTensorProduct): The segmented tensor product object configured with the specified irreps.
    CGC (torch.Tensor): A tensor containing the Clebsch-Gordan coefficients C_{ir1,ir2,ir3}.

    Example:
         irreps_in1 = cue.Irreps("O3", "1x0e + 1x1o + 1x2o")
         irreps_in2 = cue.Irreps("O3", "2x0e + 2x1o + 2x2o")
         irreps_out = cue.Irreps("O3", "2x0e + 2x1o + 2x2o")
    """

    
    d = stp.SegmentedTensorProduct.from_subscripts("abcu,a,bu,cu")
    # 
    d_W = stp.SegmentedTensorProduct.from_subscripts("i,u,iu")
    #
    d_W2 = []

    for mul, ir in irreps_sph:
        d.add_segment(1, (ir.dim, ))
    for mul, ir in irreps_in:
        d.add_segment(2, (ir.dim, mul))
    for mul, ir in irreps_out:
        d.add_segment(3, (ir.dim, mul))

    print(d)
    N = 0
    weight = []
    CG_path = [] 
    iCG = 0 
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)), (i3, (mul3, ir3)) in itertools.product(
        enumerate(irreps_sph),enumerate(irreps_in), enumerate(irreps_out)
    ):
        
        if mul1 !=1 or mul2 != mul3:
            print(f'irreps_in1 = {irreps_sph}')
            print(f'irreps_in2 = {irreps_in}')
            print(f'irreps_out = {irreps_out}')
            raise ValueError('multiplicity not consistent')
        
        CG = cue.O3.clebsch_gordan(ir1,ir2,ir3)
        if len(CG)!=0:
            #print(CG)
            NCG = len(CG.reshape(-1))
            N += NCG
            for x in CG.reshape(-1):
                weight.append(x)
            CG_path.append([ir1,ir2,ir3])
            #print(i1,i2,i3)
            d.add_path(None, i1, i2, i3, c=1.0)

            d_W.add_segment(0, (NCG,))
            d_W.add_segment(1, (mul2,))
            d_W.add_segment(2, (NCG,mul2))
            d_W.add_path(iCG,iCG,iCG,c=1.0)
            iCG += 1

            #d_W2.append([ir1,ir2,ir3])
    CGC = torch.tensor(weight).reshape(1,-1)
    #for 

    return d, d_W, CGC  

@compile_mode("script")
class etp_l1m1_kl2m2_2_kl3m3(nn.Module):
    def __init__(self, 
                 irreps_sph, 
                 irreps_in, 
                 irreps_out,
                 internal_weight = False, 
                 device='cuda', 
                 dtype=torch.float32):
        super().__init__()

        d, d_W, CGC = descriptor_stp_l1m1_kl2m2_2_kl3m3(irreps_sph, irreps_in, irreps_out)
        self.tp = cuet.TensorProduct(d, device=device)
        self.weight_map = cuet.TensorProduct(d_W, device=device)
        self.weight_numel = d_W.operands[1].size
        
        CGC = CGC.reshape(1,-1)
        
        self.register_buffer('CGC', CGC.clone().to(device = device,dtype=dtype))
        self.internal_weight = internal_weight
        if(internal_weight):
            self.weight = nn.Parameter(torch.randn(1,d_W.operands[1].size, device=device))
        return 
    
    def forward(self, 
                Ylm: torch.tensor,
                hklm: torch.tensor,
                weight: Union[None, torch.tensor])->torch.tensor:
        #
        if weight is None:
            if self.internal_weight: 
                CGC_w = self.weight_map(self.CGC, self.weight)
            else:
                raise ValueError('no internal_weight, please pass in weight')
        else:
            CGC_w = self.weight_map(self.CGC, weight)
        x = self.tp(CGC_w, Ylm, hklm) 
        return x
    
    def contract(self, 
                 Ylm: torch.tensor, 
                 hklm: torch.tensor, 
                 weight: torch.tensor) -> torch.tensor:
        CGC_w = self.weight_map(self.CGC, weight)
        return self.tp(CGC_w, Ylm, hklm) 


    