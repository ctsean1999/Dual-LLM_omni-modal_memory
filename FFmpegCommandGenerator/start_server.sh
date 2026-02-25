#!/bin/bash

# 启动Web服务器的脚本

# 设置中文显示
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 打印启动信息
echo "====================================="
echo "  FFmpeg命令生成器 - Web服务启动脚本  "
echo "====================================="
echo "
请确保您已安装所有依赖：
  pip3 install -r requirements.txt
"
echo "服务将在 http://localhost:5000 启动..."
echo "
使用说明：
1. 服务启动后，在浏览器中打开web_app/index.html文件
2. 输入视频地址、起始时间、终止时间和输出文件名
3. 点击'生成命令'按钮查看完整的FFmpeg命令
4. 点击'保存命令'按钮将命令追加到command.txt文件
5. 使用'上一条'和'下一条'按钮浏览历史命令
"
echo "按Ctrl+C可以停止服务"
echo "====================================="
echo "
正在启动Flask服务器..."

# 启动Flask服务器
python3 app.py