# vllm总体架构

本系列文章主要参考的代码版本是**v0.1.0**。

### 两种调用方式



<img src="./img/vllm-0.png" style="zoom: 35%;" />

vllm有两种调用方式:

* **Offline Batched Inference**（**同步**，离线批处理）
* **API Server For Online Serving**（**异步**，在线推理服务）







