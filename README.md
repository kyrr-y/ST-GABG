# ST-GABG：轴承故障诊断 + 实时数据管道

基于 GCN+BiGRU+GAT 的故障诊断模型，提供 API 服务，并集成 MQTT、InfluxDB、Grafana 实现数据采集、存储与可视化。

## 目录结构
ST-GABG/
├── api.py # FastAPI 应用（模型推理接口）
├── ST_GABG.py # 模型定义
├── best_model.pth # 预训练权重（Git LFS）
├── requirements.txt
├── Dockerfile # API 镜像
├── Dockerfile.publisher # MQTT 发布者镜像
├── Dockerfile.subscriber # MQTT 订阅者镜像
├── docker-compose.yml # 完整服务编排
├── mqtt_publisher.py # 模拟振动数据发布
├── mqtt_subscriber_influx.py # 订阅 MQTT，调用 API，写入 InfluxDB
└── grafana/provisioning/ # Grafana 


## 一键启动

```bash
# 克隆仓库（含 LFS 模型文件）
git clone https://github.com/your-username/ST-GABG.git
cd ST-GABG
git lfs pull

# 首次需初始化 InfluxDB（创建 org/bucket/token）
docker-compose up -d influxdb
docker exec -it influxdb influx setup \
  --username admin --password admin123 \
  --org my-org --bucket vibration_bucket \
  --retention 0 --token your-token

# 将 token 填入 docker-compose.yml 的 INFLUXDB_TOKEN，然后启动所有服务
docker-compose up -d --build

API 文档	http://localhost:8000/docs	-
InfluxDB	http://localhost:8086	=
Grafana	http://localhost:3000	
