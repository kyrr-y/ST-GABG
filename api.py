import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# 导入你的模型（假设 ST_GABG.py 在同一目录）
from ST_GABG import GCNBiGRUGATTModel, device

app = FastAPI(title="ST-GABG 模型 API 服务")

# ====================== 模型参数（和训练完全一致）======================
gcn_input_channels = 2
gcn_convLayers = (64, 128, 256)
bigru_input_dim = 32
hidden_layer_sizes = [64, 128, 256]
num_classes = 10

# ====================== 加载模型 ======================
try:
    model = GCNBiGRUGATTModel(
        gcn_input_channels,
        gcn_convLayers,
        bigru_input_dim,
        hidden_layer_sizes,
        num_classes
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("✅ 模型加载成功 —— 无训练，纯推理！")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

# ====================== 图构建函数（与训练时一致）======================
def make_NearestNeighbors(signals, m_k=5):
    """KNN 邻居计算"""
    nn = NearestNeighbors(n_neighbors=m_k+1, metric='euclidean')
    nn.fit(signals)
    distances, indices = nn.kneighbors(signals)
    return distances[:, 1:], indices[:, 1:]   # 排除自环

def get_edge_indexs(signals, indices, distances):
    """构建边和边权重"""
    edge_index = []
    edge_weights = []
    for i in range(len(signals)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                edge_index.append([i, j])
                edge_weights.append(distances[i][j_idx])
    edge_index = np.array(edge_index).T
    edge_weights = np.array(edge_weights)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    return edge_index, edge_weights

def temporal_feature_augmentation(signal):
    """时间步特征增强：信号标准化 + 时间步标准化"""
    signal = signal.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_signal = scaler.fit_transform(signal)
    time_steps = np.arange(len(signal)).reshape(-1, 1)
    time_scaler = StandardScaler()
    scaled_time = time_scaler.fit_transform(time_steps)
    return np.hstack((scaled_signal, scaled_time))

def build_graph_from_signal(signal, m_k=5):
    """根据一维信号构建图数据"""
    features = temporal_feature_augmentation(signal)          # shape (1024, 2)
    distances, indices = make_NearestNeighbors(features, m_k)
    edge_index, edge_attr = get_edge_indexs(features, indices, distances)
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # 全局池化需要 batch 信息，单个图直接设为全 0
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

# ====================== 测试接口 ======================
@app.get("/")
def home():
    return {
        "status": "运行成功 ✅",
        "model": "ST-GABG 已就绪",
        "接口文档": "/docs"
    }

# ====================== 预测接口 ======================
class InferInput(BaseModel):
    signal: list   # 长度 1024

@app.post("/predict")
def predict(input: InferInput):
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    try:
        signal = np.array(input.signal, dtype=np.float32)
        if len(signal) != 1024:
            raise HTTPException(status_code=400, detail="信号长度必须为 1024")
        
        # 1. 动态构建图数据
        graph_data = build_graph_from_signal(signal, m_k=5)
        graph_data = graph_data.to(device)
        
        # 2. 准备信号张量
        signal_tensor = torch.from_numpy(signal).float().to(device).view(1, 32, 32)
        
        # 3. 推理
        with torch.no_grad():
            logits = model(graph_data, signal_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            "code": 200,
            "predicted_class": pred_class,
            "confidence": confidence,
            "status": "success"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()   # 打印详细堆栈到日志
        raise HTTPException(status_code=500, detail=str(e))

# ====================== 启动服务（用于直接运行）======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)