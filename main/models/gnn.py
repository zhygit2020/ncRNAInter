import torch
import torch.nn as nn

class apply_node_func(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, norm=None):
        super(apply_node_func, self).__init__()
        self.fc_neigh = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.activation = activation
        self.norm = norm
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=nn.init.calculate_gain('relu')) 

    def forward(self, node):
        h_neigh = node.data['neigh'] # NodeBatch
        h_neigh = self.fc_neigh(h_neigh) # nodeupdte
        if self.activation is not None:
            h_neigh = self.activation(h_neigh)
        if self.norm is not None:
            h_neigh = self.norm(h_neigh)
        return {'activation': h_neigh}


class NodeSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, node_num, id2nodeidx, activation=None, norm=None, dropout=0.0):
        super(NodeSAGE, self).__init__()
        self.n_layers = n_layers
        self.node_num = node_num
        self.id2nodeidx = id2nodeidx
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList() # Holds submodules in a list
        self.layers.append(apply_node_func(in_feats=in_feats, out_feats=n_hidden, activation=activation, norm=norm))
        for _ in range(n_layers - 1):
            self.layers.append(apply_node_func(in_feats=n_hidden, out_feats=n_hidden, activation=activation, norm=norm))
        # submodule of nodeupdate，consists of n_layers' apply_node_func module

    def message_func(self, edges): # edge:dgl.EdgeBatch
        m = edges.src['h'] * edges.data['edgeweight']
        return {'m': m, 'edgeweight': edges.data['edgeweight']}

    def reduce_func(self, nodes): # node:dgl.NodeBatch
        # remove the validation edge information of which nodes.mailbox['edgeweight']==0
        with torch.no_grad():
            effective_message_nums = torch.count_nonzero(nodes.mailbox['edgeweight'],dim=1) #tensor([5, 4, 5, 5]) 
        neigh = torch.div(torch.sum(nodes.mailbox['m'], dim=1), effective_message_nums)
        return {'neigh': neigh}

    def forward(self, blocks:'list of dgl.MFGs', x :'first_layer_nodes_features', tasktype : 'task type, str'): # EdgeSampler data
        blocks[0].srcdata['activation'] = x
        for i, layer in enumerate(self.layers):
            h = blocks[i].srcdata.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            blocks[i].srcdata['h'] = h
            blocks[i].update_all(message_func=self.message_func, reduce_func=self.reduce_func, apply_node_func=layer)
            if i<= len(blocks)-2:
                blocks[i+1].srcdata['activation'] = blocks[i].dstdata['activation']
        updated_nfea = blocks[-1].dstdata['activation']
        return updated_nfea

class EdgePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W1 = nn.Linear(2 * in_features, in_features) # edge classifier after node cat
        self.W2 = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.W1.weight, gain=nn.init.calculate_gain('relu')) 
        self.actifun1 = nn.ReLU()
        self.actifun2 = nn.Softmax(dim=1)

    def apply_edges(self, edges):
        data = torch.cat([edges.src['x'], edges.dst['x']], 1)
        data = self.actifun1(self.W1(data))
        logits = self.actifun2(self.W2(data))[:,1].unsqueeze(-1)
        return {'logits': logits}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope(): # local_scope avoid reflecting to the original graph
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['logits']

class GNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, node_num, id2nodeidx, activation, dropout):
        super().__init__()
        # two part of whole model
        self.updater = NodeSAGE(in_feats=in_feats,
                                n_hidden=n_hidden,
                                n_layers=n_layers,
                                node_num=node_num,
                                id2nodeidx=id2nodeidx,
                                activation=activation,
                                norm=None,
                                dropout=dropout)
        self.predictor = EdgePredictor(num_classes=n_classes, in_features=n_hidden)

    def forward(self, edge_subgraph, blocks, x, tasktype):
        upnfea = self.updater(blocks, x, tasktype)
        return self.predictor(edge_subgraph, upnfea)
