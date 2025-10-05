import numpy as np
import torch
import math
from clustpy.deep.neural_networks._abstract_autoencoder import _AbstractAutoencoder


class FullyConnectedBlockCompressed(torch.nn.Module):
    """
    Feed Forward Neural Network Block with optional low-rank and Hadamard-decomposition reparameterization,
    except for the first and last layer (which remain standard Linear layers).
    """

    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = None, bias: bool = True,
                 output_fn: torch.nn.Module = None, low_rank_factors: int = None, 
                 hadamard_factors: int = None, low_rank: int = None, rank: int = None,
                 multiplier: int = 1):
        super(FullyConnectedBlockCompressed, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        self.low_rank = low_rank
        self.rank = rank
        self.multiplier = multiplier

        layer_positions = []
        fc_block_list = []

        for i in range(len(layers) - 1):
            layer_positions.append(len(fc_block_list))

            # --- Decide whether to compress ---
            is_first = (i == 0)
            is_last = (i == len(layers) - 2)

            if (not is_first) and (not is_last):  
                # Apply compression only for intermediate layers
                if hadamard_factors is not None and hadamard_factors!=1:
                    if self.rank is None: 
                        thisrank = np.max([10, np.min([int(np.sqrt(layers[i])), 
                                                       int(np.sqrt(layers[i + 1]))])]) * multiplier
                    else: 
                        thisrank = self.rank
                    fc_block_list.append(HadamardLowRankLinear(
                        layers[i], layers[i + 1],
                        rank=thisrank,
                        num_factors=hadamard_factors,
                        bias=self.bias
                    ))
                elif hadamard_factors ==1:
                    #thisrank = np.max([10, np.min([int(np.sqrt(layers[i])), 
                    #                                   int(np.sqrt(layers[i + 1]))])]) * multiplier
                    
                    thisrank = np.max([10, int(np.min([layers[i], layers[i + 1]])*multiplier)  ] )
                    fc_block_list.append(LowRankLinear(layers[i], layers[i + 1], 
                                                       rank=thisrank, bias=self.bias))
                else:
                    fc_block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=self.bias))
            else:
                # First and last layers are always standard Linear
                fc_block_list.append(torch.nn.Linear(layers[i], layers[i + 1], bias=self.bias))

            # Optional BatchNorm
            if self.batch_norm:
                fc_block_list.append(torch.nn.BatchNorm1d(layers[i + 1]))

            # Optional Dropout
            if self.dropout is not None:
                fc_block_list.append(torch.nn.Dropout(self.dropout))

            # Activation (only if not last layer)
            if not is_last:
                if self.activation_fn is not None:
                    fc_block_list.append(self.activation_fn())
            else:
                if self.output_fn is not None:
                    fc_block_list.append(self.output_fn())

        self.block = torch.nn.Sequential(*fc_block_list)
        self.layer_positions = layer_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class LowRankLinear(torch.nn.Module):
    '''
    implements standard low-rank reparameterization of a given autencoder weight matrix for compression
    '''
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.U = torch.nn.Parameter(torch.empty(out_features, rank))
        self.V = torch.nn.Parameter(torch.empty(rank, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None

        # Initialize
        torch.nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.U @ self.V
        return torch.nn.functional.linear(x, weight, self.bias)



class HadamardLowRankLinear(torch.nn.Module):
    '''
    implements Hadamard-decomposition reparametrization of an autoencoder weight matrix for compression 
    '''
    def __init__(self, in_features, out_features, rank, num_factors=2, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_factors = num_factors
        self.bias_flag = bias

        self.U_list = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(out_features, rank))
            for _ in range(num_factors)
        ])
        self.V_list = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(rank, in_features))
            for _ in range(num_factors)
        ])

        if self.bias_flag:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.U_list)):

            # Initialization affects the convergence stability for our parameterization
            fan = torch.nn.init._calculate_correct_fan(self.U_list[i], mode='fan_in')
            gain = torch.nn.init.calculate_gain('relu', 0)
            std_u = gain / np.sqrt(fan)

            fan = torch.nn.init._calculate_correct_fan(self.V_list[i], mode='fan_in')
            std_v = gain / np.sqrt(fan)

            torch.nn.init.normal_(self.U_list[i], 0, std_u)
            torch.nn.init.normal_(self.V_list[i], 0, std_v)

            #torch.nn.init.kaiming_uniform_(U, a=math.sqrt(5))
            #torch.nn.init.kaiming_uniform_(V, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = None
        for U, V in zip(self.U_list, self.V_list):
            factor = U @ V  # shape: [out_features, in_features]
            weight = factor if weight is None else weight * factor  # Hadamard product

        return torch.nn.functional.linear(x, weight, self.bias)


class HadamardAutoencoder(_AbstractAutoencoder):
    '''
    implements the constrained (compressed) autoencoder used in Khatri-Rao clustering 
    '''
    def __init__(self, layers: list, batch_norm: bool = False, dropout: float = None,
                 activation_fn: torch.nn.Module = torch.nn.LeakyReLU, bias: bool = True, decoder_layers: list = None,
                 decoder_output_fn: torch.nn.Module = None, work_on_copy: bool = True,
                 random_state: np.random.RandomState | int = None, rank: int = None, hadamard_factors:int=2,  multiplier: int=1):
        super().__init__(work_on_copy, random_state)
        if decoder_layers is None:
            decoder_layers = layers[::-1]
        if (layers[-1] != decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {layers[-1]} and {decoder_layers[0]} respectively.")
        if (layers[0] != decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {layers[0]} and {decoder_layers[-1]} respectively.")
        # Initialize encoder 
        self.encoder = FullyConnectedBlockCompressed(layers=layers, batch_norm=batch_norm, dropout=dropout,
                                           activation_fn=activation_fn, bias=bias,
                                           output_fn=None, hadamard_factors=hadamard_factors, low_rank=rank, multiplier=multiplier)
 
        # Inverts the list of layers to make symmetric version of the encoder
        self.decoder = FullyConnectedBlockCompressed(layers=decoder_layers, batch_norm=batch_norm, dropout=dropout,
                                           activation_fn=activation_fn, bias=bias,
                                           output_fn=decoder_output_fn, hadamard_factors=hadamard_factors, low_rank=rank,  multiplier=multiplier)
         
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.encoder.layers[0]:
            raise ValueError("Input layer of the encoder ({0}) does not match input sample ({1})".format(self.encoder.layers[0],
                                                                                            x.shape[1]))
        embedded = self.encoder(x)
        return embedded
    
    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        if embedded.shape[1] != self.decoder.layers[0]:
            raise ValueError("Input layer of the decoder does not match input sample")
        decoded = self.decoder(embedded)
        return decoded
