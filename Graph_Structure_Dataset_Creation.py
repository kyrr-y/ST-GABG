from joblib import dump, load

# 加载数据
train_set = load('./dataresult/train_set') 
val_set = load('./dataresult/val_set') 
test_set = load('./dataresult/test_set') 
print(train_set.shape)  # 二维数组，行代表样本数量， 列代表信号长度和标签
print(val_set.shape)
print(test_set.shape)
from joblib import dump, load
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch_geometric.data import Data



def make_NearestNeighbors(signals, m_k=5):
    """改进后的KNN函数，返回节点距离和索引"""
    nn = NearestNeighbors(n_neighbors=m_k+1, metric='euclidean')  # 多找一个邻居用于排除自环
    nn.fit(signals)
    distances, indices = nn.kneighbors(signals)
    return distances[:, 1:], indices[:, 1:]  # 排除第一个自环邻居

def get_edge_indexs(signals, indices, distances):
    """改进后的边构建函数，添加边权重并排除自环边"""
    edge_index = []
    edge_weights = []
    for i in range(len(signals)):
        for j_idx, j in enumerate(indices[i]):
            if i != j:  # 排除自环边
                edge_index.append([i, j])
                edge_weights.append(distances[i][j_idx])
    
    edge_index = np.array(edge_index).T
    edge_weights = np.array(edge_weights)
    
    # 转换为tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    return edge_index, edge_weights

def temporal_feature_augmentation(signal):
    """时间步特征增强：添加标准化后的时间步作为第二维特征"""
    # 信号标准化
    signal_scaler = StandardScaler()
    scaled_signal = signal_scaler.fit_transform(signal)
    
    # 生成时间步并标准化
    time_steps = np.arange(len(signal)).reshape(-1, 1)
    time_scaler = StandardScaler()
    scaled_time = time_scaler.fit_transform(time_steps)
    
    # 合并特征
    return np.hstack((scaled_signal, scaled_time))

def make_graph_dataset(dataframe, m_k):
   
    x_data = dataframe.iloc[:,0:-1].values
    y_labels = dataframe.iloc[:,-1].values

    dataset = []  
    for index in range(dataframe.shape[0]):
        # 原始信号处理
        raw_signal = x_data[index].reshape(-1, 1)
        
        # 特征增强：信号标准化 + 时间步特征
        features = temporal_feature_augmentation(raw_signal)
        
        # 构建KNN关系
        distances, indices = make_NearestNeighbors(features, m_k=m_k)
        
        # 构建边和边权重
        edge_index, edge_attr = get_edge_indexs(features, indices, distances)
        
        # 创建Data对象
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y_labels[index], dtype=torch.long)
        )
        dataset.append(data)
    
    return dataset

def cross_validate_k(train_set, k_candidates):
    """K值交叉验证函数（需配合具体模型使用）"""
    best_k = k_candidates[0]
    best_score = -np.inf
    
    for k in k_candidates:
        print(f"\n--- 正在验证K={k} ---")
        fold_scores = []
        
        # 创建K折交叉验证
        kf = KFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
            # 生成训练/验证子集
            train_sub = train_set.iloc[train_idx]
            val_sub = train_set.iloc[val_idx]
            
            # 生成图数据
            train_graph = make_graph_dataset(train_sub, k)
            val_graph = make_graph_dataset(val_sub, k)
            
            # ----------------------
            # 这里需要添加模型训练和验证代码
            # 示例伪代码：
            # model = GNNModel()
            # train(model, train_graph)
            # score = evaluate(model, val_graph)
            # fold_scores.append(score)
            # ----------------------
            
            print(f"Fold {fold+1} 完成")
        
        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    
    print(f"\n最佳K值: {best_k} (得分: {best_score:.4f})")
    return best_k

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":

    # 步骤1：通过交叉验证选择最佳K值（需要实现模型部分）
    k_candidates = [3, 5, 7, 9]
    # best_k = cross_validate_k(train_set, k_candidates)
    best_k = 5  # 假设交叉验证选择的结果
    
    # 步骤2：使用最佳K值生成最终数据集
    print(f"\n使用最佳K值 {best_k} 生成图数据集...")
    train_graph_data = make_graph_dataset(train_set, best_k)
    val_graph_data = make_graph_dataset(val_set, best_k)
    test_graph_data = make_graph_dataset(test_set, best_k)

    # 保存数据
    dump(train_graph_data, './dataresult/train_graph_data')
    dump(val_graph_data, './dataresult/val_graph_data')
    dump(test_graph_data, './dataresult/test_graph_data')
    
    print("图数据保存完成！")

