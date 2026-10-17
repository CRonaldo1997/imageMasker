"""pytest 共享配置：测试环境隔离项目 .env。

在任何 src 模块被 import 前把配置文件指向空设备，
避免测试结果受开发者本地 .env（如开启方向矫正、真实 API Key）影响。
"""
import os

os.environ.setdefault("IMAGE_MASK_ENV_FILE", os.devnull)
