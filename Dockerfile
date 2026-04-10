FROM python:3.8-slim

WORKDIR /app

# 安装编译工具（g++, make 等）
RUN apt-get update && apt-get install -y --no-install-recommends g++ build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --timeout 300 --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 现在可以成功编译安装 PyG 相关包
RUN pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]