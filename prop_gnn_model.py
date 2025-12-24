import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


def make_edge_mlp(edge_feat_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(edge_feat_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )

class MoleculeGINE(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, out_dim=4):
        super().__init__()

        self.edge_mlp = make_edge_mlp(edge_feat_dim, hidden_dim)

        #GINE
        self.conv1 = GINEConv(nn.Linear(node_feat_dim, hidden_dim), edge_dim=hidden_dim)
        self.conv2 = GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=hidden_dim)

        #multi-head
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(out_dim)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        #map edge feature via mlp
        edge_emb = self.edge_mlp(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_emb))
        x = F.relu(self.conv2(x, edge_index, edge_emb))

        #pooling
        x = global_mean_pool(x, batch)

        #multi-head prediction
        #run pooled vector x on each head
        head_outputs = [head(x) for head in self.heads]

        #concatenate
        out = torch.cat(head_outputs, dim=-1) #[B, 4]
        
        return out