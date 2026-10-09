#!/bin/bash
# AFAC2026 自动化目录初始化脚本

# 创建目录结构
mkdir -p auto_runner versions auto_logs

# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium

echo "✅ 初始化完成"

