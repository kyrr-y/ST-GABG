import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import matplotlib
from joblib import dump
from ST_GABG import (
    device,
    dataloader,
    GCNBiGRUGATTModel
)

matplotlib.rc("font", family='Microsoft YaHei')

# ====================== 超参数 ======================
batch_size = 32
epochs = 100
gcn_convLayers = (64, 128, 256)
gcn_input_channels = 2
num_classes = 10
hidden_layer_sizes = [64, 128, 256]
bigru_input_dim = 32
learn_rate = 0.003

# ====================== 加载数据 ======================
train_loader, val_loader, test_loader = dataloader(batch_size)

# ====================== 模型 ======================
model = GCNBiGRUGATTModel(
    gcn_input_channels,
    gcn_convLayers,
    bigru_input_dim,
    hidden_layer_sizes,
    num_classes
)

# ====================== 损失 & 优化器 ======================
loss_function = nn.CrossEntropyLoss(reduction='sum').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

# ====================== 参数统计 ======================
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f'Total: {sum(params)}')

count_parameters(model)

# ====================== 训练函数 ======================
def model_train(batch_size, epochs, train_loader, val_loader, model, optimizer, loss_function, device):
    model = model.to(device)
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size
    best_accuracy = 0.0

    train_loss, train_acc = [], []
    validate_acc, validate_loss = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        loss_epoch = 0.
        correct_epoch = 0

        for batchdata in train_loader:
            graph_data, signal_sequence = batchdata
            graph_data, signal_sequence = graph_data.to(device), signal_sequence.to(device)
            optimizer.zero_grad()
            y_pred = model(graph_data, signal_sequence)
            predicted_labels = torch.argmax(y_pred, dim=1)
            correct_epoch += (predicted_labels == graph_data.y).sum().item()
            loss = loss_function(y_pred, graph_data.y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)

        print(f'Epoch: {epoch+1:2} train_Loss: {loss_epoch/train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')

        # 验证
        with torch.no_grad():
            model.eval()
            loss_validate = 0.
            correct_validate = 0
            for valbatch in val_loader:
                graph_data, signal_sequence = valbatch
                graph_data, signal_sequence = graph_data.to(device), signal_sequence.to(device)
                pre = model(graph_data, signal_sequence)
                predicted_labels = torch.argmax(pre, dim=1)
                correct_validate += (predicted_labels == graph_data.y).sum().item()
                loss = loss_function(pre, graph_data.y)
                loss_validate += loss.item()

            val_accuracy = correct_validate / val_size
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            print(f'Epoch: {epoch+1:2} val_Loss:{loss_validate/val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')

            scheduler.step(loss_validate / val_size)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), "best_model.pth")

    print("best_accuracy :", best_accuracy)

# ====================== 启动训练 ======================
if __name__ == "__main__":
    model_train(batch_size, epochs, train_loader, val_loader, model, optimizer, loss_function, device)