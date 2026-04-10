import paho.mqtt.client as mqtt
import time
import math
import random
import os

# 从环境变量读取配置
BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC = os.getenv("MQTT_TOPIC", "sensor/vibration")

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

print(f"Publishing to {BROKER}:{PORT} topic {TOPIC}")

t = 0
while True:
    # 模拟振动数据：10 Hz 正弦波 + 随机噪声
    value = 5 * math.sin(2 * math.pi * 10 * t) + random.gauss(0, 0.5)
    payload = f"{value:.3f}"
    client.publish(TOPIC, payload)
    print(f"Sent: {payload}")
    time.sleep(0.1)   # 10 Hz 采样率
    t += 0.1