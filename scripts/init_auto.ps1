# AFAC2026 自动化目录初始化脚本

# 创建目录结构
New-Item -Path \"auto_runner\" -ItemType Directory -Force
New-Item -Path \"versions\" -ItemType Directory -Force
New-Item -Path \"auto_logs\" -ItemType Directory -Force

# 安装依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium

Write-Host \"✅ 初始化完成\" -ForegroundColor Green

