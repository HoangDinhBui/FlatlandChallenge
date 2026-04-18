"""
Minimal standalone GAT implementation for loading GNNPsPPO model.
No dependency on torch_geometric - pure PyTorch.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalGATConv(nn.Module):
    """
    Minimal Graph Attention Network layer compatible with torch_geometric's GATConv state_dict.
    Implements the core GAT attention mechanism using only PyTorch.
    """

    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        # Linear transformation (matches torch_geometric's GATConv.lin)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters (matches torch_geometric's att_src, att_dst)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Bias
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        x: [N, in_channels]
        edge_index: [2, E] (source, target)
        returns: [N, heads * out_channels] if concat else [N, out_channels]
        """
        N = x.size(0)
        H = self.heads
        C = self.out_channels

        # Linear transform: [N, H*C]
        x_transformed = self.lin(x)  # [N, H*C]
        x_transformed = x_transformed.view(N, H, C)  # [N, H, C]

        # Compute attention scores
        # alpha_src = (x_transformed * att_src).sum(dim=-1) -> [N, H]
        alpha_src = (x_transformed * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_transformed * self.att_dst).sum(dim=-1)  # [N, H]

        if edge_index.size(1) == 0:
            # No edges - just return transformed features
            out = x_transformed.view(N, H * C) if self.concat else x_transformed.mean(dim=1)
            return out + self.bias

        src, dst = edge_index[0], edge_index[1]  # [E]

        # Attention coefficients: e_ij = LeakyReLU(alpha_src_i + alpha_dst_j)
        alpha = alpha_src[src] + alpha_dst[dst]  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax over neighbors
        # For each destination node, softmax over all source nodes
        alpha_max = torch.zeros(N, H, device=x.device).fill_(-1e9)
        alpha_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, H), alpha, reduce='amax', include_self=True)
        alpha = alpha - alpha_max[dst]
        alpha = torch.exp(alpha)

        alpha_sum = torch.zeros(N, H, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), alpha)
        alpha_sum = alpha_sum.clamp(min=1e-12)

        alpha = alpha / alpha_sum[dst]  # [E, H]

        # Dropout on attention
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)

        # Message passing: aggregate source features weighted by attention
        messages = x_transformed[src] * alpha.unsqueeze(-1)  # [E, H, C]

        out = torch.zeros(N, H, C, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, C), messages)

        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)

        return out + self.bias


class TreeObsToGraph:
    """
    Convert TreeObs flat array to graph format.
    Each node in the observation tree = 1 graph node.
    """
    N_FEATURES = 11

    def __call__(self, flat_obs):
        if flat_obs is None:
            return (
                torch.zeros((1, self.N_FEATURES)),
                torch.zeros((2, 0), dtype=torch.long)
            )

        total_features = len(flat_obs)
        n_nodes = max(1, total_features // self.N_FEATURES)
        remainder = total_features % self.N_FEATURES

        if remainder != 0:
            flat_obs = np.concatenate([flat_obs, np.zeros(self.N_FEATURES - remainder)])
            n_nodes = len(flat_obs) // self.N_FEATURES

        node_features = flat_obs[:n_nodes * self.N_FEATURES].reshape(n_nodes, self.N_FEATURES)
        node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)

        edges_src, edges_dst = [], []
        for i in range(1, n_nodes):
            parent = (i - 1) // 4
            edges_src.extend([parent, i])
            edges_dst.extend([i, parent])

        x = torch.FloatTensor(node_features)

        if len(edges_src) > 0:
            edge_index = torch.LongTensor([edges_src, edges_dst])
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return x, edge_index


class StandaloneGNNPsPPO(nn.Module):
    """
    Standalone GNN Actor-Critic that matches the GNNPsPPO architecture
    but uses MinimalGATConv instead of torch_geometric.
    """

    def __init__(self, action_size=5, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.action_size = action_size
        self.masking_value = torch.tensor(-1e+8)
        self.softmax = nn.Softmax(dim=-1)

        node_feat_dim = 11

        # GAT Encoder
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            MinimalGATConv(node_feat_dim, hidden_dim // n_heads,
                           heads=n_heads, dropout=0.1, concat=True)
        )
        for _ in range(n_layers - 1):
            self.gat_layers.append(
                MinimalGATConv(hidden_dim, hidden_dim // n_heads,
                               heads=n_heads, dropout=0.1, concat=True)
            )

        # Actor head
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

        # Critic head
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.graph_converter = TreeObsToGraph()

    def _encode_single(self, obs_np):
        """Encode a single observation through GNN."""
        x, edge_index = self.graph_converter(obs_np)
        x = x.to(next(self.parameters()).device)
        edge_index = edge_index.to(next(self.parameters()).device)

        for gat in self.gat_layers:
            x = torch.relu(gat(x, edge_index))

        # Global mean pooling
        embedding = x.mean(dim=0, keepdim=True)
        return embedding

    def act(self, observation, action_mask=None):
        """
        Get action from observation.
        observation: numpy array (flat)
        action_mask: list or numpy array of 0/1
        returns: action index
        """
        self.eval()
        with torch.no_grad():
            embedding = self._encode_single(observation)
            logits = self.fc_actor(embedding).squeeze(0)

            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.bool)
                logits = torch.where(mask, logits, self.masking_value)

            probs = self.softmax(logits)
            action = torch.multinomial(probs, 1).item()

        return action

    def get_value(self, observation):
        """Get critic value for observation."""
        self.eval()
        with torch.no_grad():
            embedding = self._encode_single(observation)
            value = self.fc_critic(embedding).item()
        return value

    def load_checkpoint(self, path):
        """Load model from checkpoint file."""
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state_dict)
        self.eval()
        print(f"[Model] Loaded checkpoint from {path}")
        return True
