import torch
from torch.nn import Parameter, ModuleList, BatchNorm1d
from Graph_Attention_Layer import *

class Binding_Affinity_Predictor(torch.nn.Module):
    def __init__(self, in_channels, num_gnn_layers, num_linear_layers, linear_out_channels):
        super(Binding_Affinity_Predictor, self).__init__()
        self.in_channels = in_channels
        
        self.num_gnn_layers = num_gnn_layers
        self.gated_graph_attention_layers = ModuleList()
        #self.batch_norms = ModuleList()
        for layer_id in range(num_gnn_layers):
            self.gated_graph_attention_layers.append(Gate_Augmented_Graph_Attention_Layer(in_channels, in_channels, True))
        #    self.batch_norms.append(BatchNorm1d(in_channels))

        self.pooling_layer = global_add_pool

        self.num_linear_layers = num_linear_layers
        self.linear_layers = ModuleList()
        for layer_id in range(num_linear_layers):
            if layer_id == 0:
                self.linear_layers.append(Linear(self.in_channels, linear_out_channels[0]))
                continue

            self.linear_layers.append(Linear(linear_out_channels[layer_id - 1], linear_out_channels[layer_id]))

        self.out = Linear(linear_out_channels[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linear_layers:
            layer.reset_parameters()

        self.out.reset_parameters()

    def forward(self, batched_data):
        x, edge_index_1, edge_index_2, edge_weight = batched_data["x"], batched_data["edge_index_1"], batched_data["edge_index_2"], batched_data["edge_weight"]

        for index, layer in enumerate(self.gated_graph_attention_layers):
            if index == 0:
                x_1 = layer(x, edge_index_1)
         #       x_1 = self.batch_norms[index](x_1) 
                x_2 = layer(x, edge_index_2, edge_weight)
         #       x_2 = self.batch_norms[index](x_2)
                continue
        
            x_1 = layer(x_1, edge_index_1)
            #x_1 = self.batch_norms[index](x_1) 
            x_2 = layer(x_2, edge_index_2, edge_weight)
            #x_2 = self.batch_norms[index](x_2)

        node_representation = x_2 - x_1
        graph_representation = self.pooling_layer(node_representation, batched_data.batch)

        for layer in self.linear_layers:
            graph_representation = layer(graph_representation)
            graph_representation = relu(graph_representation)

        binding_affinity = self.out(graph_representation)

        return torch.nan_to_num(binding_affinity)        

        #return binding_affinity