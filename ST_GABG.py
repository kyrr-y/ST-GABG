import torch
from joblib import dump, load
from torch_geometric.loader import DataLoader
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ====================== 全局配置 ======================
torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 数据集定义 ======================
class GraphSequenceDataset(Dataset):
    def __init__(self, graph_data, sequence_data):
        assert len(graph_data) == len(sequence_data), "Graph data and sequence data must have the same length."
        self.graph_data = graph_data
        self.sequence_data = sequence_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, index):
        graph = self.graph_data[index]
        sequence = self.sequence_data[index, :-1].astype(np.float32)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        return graph, sequence

# ====================== 数据加载 ======================
def dataloader(batch_size):
    train_graph_data = load('./dataresult/train_graph_data')
    train_signal_sequences = load('./dataresult/train_set')
    val_graph_data = load('./dataresult/val_graph_data')
    val_signal_sequences = load('./dataresult/val_set')
    test_graph_data = load('./dataresult/test_graph_data')
    test_signal_sequences = load('./dataresult/test_set')

    train_dataset = GraphSequenceDataset(train_graph_data, train_signal_sequences.values)
    val_dataset = GraphSequenceDataset(val_graph_data, val_signal_sequences.values)
    test_dataset = GraphSequenceDataset(test_graph_data, test_signal_sequences.values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

# ====================== 注意力层 ======================
class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector

# ====================== 模型结构 ======================
class GCNBiGRUGATTModel(nn.Module):
    def __init__(self, gcn_input_channels, gcn_convLayers, bigru_input_dim, hidden_layer_sizes, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # BiGRU
        self.num_layers = len(hidden_layer_sizes)
        self.bigru_layers = nn.ModuleList()
        self.bigru_layers.append(nn.GRU(bigru_input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        for i in range(1, self.num_layers):
            self.bigru_layers.append(nn.GRU(hidden_layer_sizes[i-1]*2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))

        self.globalAttention = GlobalAttention(hidden_layer_sizes[-1])

        # GCN
        self.convLayers = gcn_convLayers
        self.input_channels = gcn_input_channels
        self.gcn_layers = nn.ModuleList()
        for hidden_channels in gcn_convLayers:
            self.gcn_layers.append(GCNConv(self.input_channels, hidden_channels))
            self.input_channels = hidden_channels

        # 门控融合
        self.projection_gcn = nn.Linear(256, 512)
        self.projection_gatt = nn.Linear(512, 512)
        self.gate = nn.Sequential(nn.Linear(512+512, 512), nn.Sigmoid())
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_layer_sizes[-1] + gcn_convLayers[-1], num_classes)

    def forward(self, data, signal_sequence):
        batch_size = signal_sequence.size(0)
        bigru_out = signal_sequence.view(batch_size, 32, 32)

        # BiGRU
        hidden = []
        for bigru in self.bigru_layers:
            bigru_out, hidden = bigru(bigru_out)
        gatt_features = self.globalAttention(hidden[-1], bigru_out)

        # GCN
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcnconv in self.gcn_layers:
            x = gcnconv(x, edge_index)
            x = F.relu(x)
        gcn_features = global_mean_pool(x, batch)

        # 门控融合
        projected_gcn = self.projection_gcn(gcn_features)
        projected_gatt = self.projection_gatt(gatt_features)
        concat_features = torch.cat((projected_gcn, projected_gatt), dim=1)
        gate_values = self.gate(concat_features)
        combined_features = gate_values * projected_gcn + (1 - gate_values) * projected_gatt

        outputs = self.classifier(combined_features)
        return outputs