import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import dgl
import numpy as np

from .gatedgcn_layer import GatedGCNLayer

class SemanticLabelEncoder(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        node_feat_dim = net_params['emb_dim']
        num_unique_nodes = net_params['num_unique_nodes']
        # node_weights = nn.init.xavier_uniform_(torch.empty(g.number_of_nodes(), node_feat_dim), gain=nn.init.calculate_gain('tanh'))
        node_weights = nn.init.xavier_uniform_(torch.empty(num_unique_nodes, node_feat_dim), gain=2.0)
        self.node_unique_embeddings = torch.nn.Embedding.from_pretrained(node_weights, freeze=False, padding_idx=None)

        edge_feat_dim = net_params['emb_dim']
        num_unique_edges = net_params['num_unique_edges']
        # edge_weights = nn.init.xavier_uniform_(torch.empty(num_unique_edges, edge_feat_dim), gain=nn.init.calculate_gain('tanh'))
        edge_weights = nn.init.xavier_uniform_(torch.empty(num_unique_edges, edge_feat_dim), gain=2.0)
        self.edge_unique_embeddings = torch.nn.Embedding.from_pretrained(edge_weights, freeze=False, padding_idx=None)

    def _get_inputs(self, node_inputs, edge_inputs):

        edge_embeddings = self.edge_unique_embeddings(edge_inputs)
        node_embeddings = self.node_unique_embeddings(node_inputs)

        return node_embeddings, edge_embeddings

    def _get_outputs(self, inputs, task):

        if "CTA" in task:
            node_embeddings = self.node_unique_embeddings(inputs)
            return node_embeddings

        if "CPA" in task:
            edge_embeddings = self.edge_unique_embeddings(inputs)
            return edge_embeddings

class GCNNet(nn.Module):

    def __init__(self, g, net_params):
        super().__init__()

        self.g = g

        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']

        self.input_encoder = SemanticLabelEncoder({
            'emb_dim': net_params['emb_dim'],
            'num_unique_nodes': g.number_of_nodes(),
            'num_unique_edges': net_params['p_vocab_size']
        })

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList( [ GatedGCNLayer(hidden_dim, hidden_dim, dropout, batch_norm, residual) for _ in range(n_layers - 1)] )
        self.layers.append( GatedGCNLayer(hidden_dim, out_dim, dropout, batch_norm, residual) )

        self.node_outputs = None
        self.edge_outputs = None

    def _get_inputs(self, node_inputs, edge_inputs):

        node_embeddings = self.input_encoder.node_unique_embeddings(node_inputs)
        edge_embeddings = self.input_encoder.edge_unique_embeddings(edge_inputs)

        return node_embeddings, edge_embeddings

    def _get_outputs(self, inputs, task):

        if "CTA" in task:
            node_embeddings = self.node_outputs[inputs,:]
            return node_embeddings

        if "CPA" in task:
            edge_embeddings = self.edge_outputs[inputs,:]
            return edge_embeddings

    def forward(self, g, h, e):

        h = self.in_feat_dropout(h)
        e = self.in_feat_dropout(e)

        h_before = h

        # GCN with node/edge updates
        for conv in self.layers:
            h, e = conv(g, h, e)

        self.node_outputs = h_before
        self.edge_outputs = e

        return self.node_outputs, self.edge_outputs