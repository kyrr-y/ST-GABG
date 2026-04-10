import torch
from ST_GABG import GCNBiGRUGATTModel  # 用你真实的模型！

# ====================== 必须和训练完全一样的参数 ======================
gcn_input_channels = 2
gcn_convLayers = (64, 128, 256)
bigru_input_dim = 32
hidden_layer_sizes = [64, 128, 256]
num_classes = 10

# ====================== 加载真实模型 ======================
model = GCNBiGRUGATTModel(
    gcn_input_channels,
    gcn_convLayers,
    bigru_input_dim,
    hidden_layer_sizes,
    num_classes
)

# 加载权重
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# ====================== 构造正确输入（你的模型需要两个参数） ======================
from torch_geometric.data import Data

# 构造图数据
x = torch.randn(1024, 2)      # 特征维度=2
edge_index = torch.randint(0, 1024, (2, 512))  # 图边
data = Data(x=x, edge_index=edge_index)

# 构造序列数据 (1,1024)
signal_sequence = torch.randn(1, 1024)

# ====================== 导出 ONNX ======================
torch.onnx.export(
    model,
    (data, signal_sequence),   # 你的模型需要两个输入！
    "model.onnx",
    opset_version=16,
    input_names=["data", "signal_sequence"],
    output_names=["output"]
)

print("✅ 导出成功！model.onnx 已生成")