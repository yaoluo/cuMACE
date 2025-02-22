from torch import nn
import torch 
from e3nn.util.jit import compile_mode


#@torch.compile
class IdentityNN(nn.Module):
   def __init__(self,
                ):
      super().__init__()
   def forward(self, x):
      return x
   
#@torch.compile
class AddModule(nn.Module):
    def __init__(self, f, g):
        super(AddModule, self).__init__()
        self.f = f
        self.g = g

    def forward(self, x):
        return self.f(x) + self.g(x)
   
@compile_mode("script")
class ResMLP(nn.Module):
   def __init__(self, 
                neuron_dim, 
                ):
      super().__init__()

      self.layers = len(neuron_dim)-1
      net = [] 
      for i in range( self.layers ):
         if i==self.layers-1:
            # output layer 
            x = nn.Linear(
               in_features=neuron_dim[i],
               out_features=neuron_dim[i+1])
         else:
            # hidden layers with nonlinearity 
            x=nn.Sequential(
               nn.Linear(  in_features=neuron_dim[i],
                           out_features=neuron_dim[i+1] ),
               nn.SiLU())
         
         if i!=self.layers-1:
            if neuron_dim[i] == neuron_dim[i+1]:
               y = IdentityNN() 
            else:
               y = nn.Linear(
                  in_features=neuron_dim[i],
                  out_features=neuron_dim[i+1], bias=False)
         if i!=self.layers-1:
            net.append(AddModule(x,y))
         else:
            net.append(x)
      self.f = nn.Sequential(*net)
   def forward(self, x):
      return self.f(x)
   
class SkipMLP(nn.Module):
   def __init__(self, 
                neuron_dim, 
                ):
      super().__init__()

      self.layers = len(neuron_dim)-1
      net = [] 
      skip_net = []
      for i in range( self.layers ):
         if i!=self.layers-1:
            net.append( nn.Sequential(
                        nn.Linear(  in_features=neuron_dim[i],
                                    out_features=neuron_dim[i+1] ),
                        nn.SiLU()))
         
         if neuron_dim[i]==neuron_dim[-1]:
            skip_net.append( IdentityNN() )
         else:
            skip_net.append( nn.Linear(  in_features=neuron_dim[i],
                                    out_features=neuron_dim[-1] ))
         
      self.f = nn.ModuleList(net)
      self.skip_f = nn.ModuleList(skip_net)
   def forward(self, x):
      out = 0 
      for layer in range(self.layers-1):
         out = out + self.skip_f[layer](x)
         x = self.f[layer](x)
      out = out + self.skip_f[-1](x)
      return out 
