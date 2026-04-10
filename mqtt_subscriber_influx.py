import paho.mqtt.client as mqtt
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import requests
from collections import deque

# ========== 配置 ==========
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "sensor/vibration")

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "my-org")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "vibration_bucket")

API_URL = "http://api:8000/predict"          # Docker 服务名
BUFFER_SIZE = 1024                          # 正式用 1024，测试可改为 10
buffer = deque(maxlen=BUFFER_SIZE)

# ========== InfluxDB 客户端 ==========
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# ========== MQTT 回调 ==========
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        value = float(msg.payload.decode())
        buffer.append(value)

        # 1. 写入原始振动数据
        point = Point("vibration") \
            .tag("sensor", "simulator") \
            .field("value", value) \
            .time(datetime.utcnow())
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"Written to InfluxDB: {value}")

        # 2. 当缓冲区满时，调用模型 API
        if len(buffer) == BUFFER_SIZE:
            signal_list = list(buffer)
            try:
                resp = requests.post(API_URL, json={"signal": signal_list}, timeout=2)
                if resp.status_code == 200:
                    result = resp.json()
                    pred_class = result.get("predicted_class")
                    confidence = result.get("confidence")
                    if pred_class is not None and confidence is not None:
                        # 写入预测结果
                        pred_point = Point("predictions") \
                            .tag("sensor", "simulator") \
                            .field("predicted_class", int(pred_class)) \
                            .field("confidence", float(confidence)) \
                            .time(datetime.utcnow())
                        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=pred_point)
                        print(f"Prediction: class {pred_class}, confidence {confidence:.3f}")
                else:
                    print(f"API call failed: {resp.status_code} {resp.text}")
            except Exception as e:
                print(f"API request error: {e}")

    except Exception as e:
        print(f"Error in on_message: {e}")

# ========== 启动 MQTT 客户端 ==========
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_forever()