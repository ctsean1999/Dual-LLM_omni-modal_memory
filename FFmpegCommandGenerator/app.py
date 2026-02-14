from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re

# 创建Flask应用实例，设置静态文件夹为当前目录
app = Flask(__name__, static_folder='.', static_url_path='/')
CORS(app)  # 启用CORS以允许前端请求

# 命令文件路径
COMMAND_FILE_PATH = './command.txt'

@app.route('/save_command', methods=['POST'])
def save_command():
    """保存命令到文件"""
    try:
        data = request.json
        command = data.get('command', '')
        
        # 验证命令格式
        if not command or not command.startswith('ffmpeg'):
            return jsonify({'status': 'error', 'message': '无效的FFmpeg命令'}), 400
        
        # 确保文件存在
        if not os.path.exists(COMMAND_FILE_PATH):
            with open(COMMAND_FILE_PATH, 'w') as f:
                pass  # 创建空文件
        
        # 读取现有内容，避免重复
        with open(COMMAND_FILE_PATH, 'r', encoding='utf-8') as f:
            existing_commands = f.readlines()
        
        # 检查是否已存在相同命令
        if any(command.strip() == c.strip() for c in existing_commands):
            return jsonify({'status': 'warning', 'message': '命令已存在'})
        
        # 追加命令到文件末尾
        with open(COMMAND_FILE_PATH, 'a', encoding='utf-8') as f:
            # 如果文件不为空，确保添加的命令和前一个命令之间没有空行
            if existing_commands and existing_commands[-1].strip():
                f.write('\n' + command)
            else:
                f.write(command)
        
        return jsonify({'status': 'success', 'message': '命令保存成功'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_commands', methods=['GET'])
def get_commands():
    """获取所有历史命令"""
    try:
        commands = []
        
        # 检查文件是否存在
        if os.path.exists(COMMAND_FILE_PATH):
            with open(COMMAND_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 只添加有效的ffmpeg命令
                    if line and line.startswith('ffmpeg'):
                        commands.append(line)
        
        return jsonify({'status': 'success', 'commands': commands})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_common_video_paths', methods=['GET'])
def get_common_video_paths():
    """获取常用视频路径"""
    try:
        # 从command.txt中提取常用的视频路径
        video_paths = set()
        
        if os.path.exists(COMMAND_FILE_PATH):
            with open(COMMAND_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('ffmpeg'):
                        # 匹配 -i 后面的视频路径
                        match = re.search(r'-i\s+([^\s]+)', line)
                        if match:
                            video_path = match.group(1)
                            video_paths.add(video_path)
        
        # 转换为列表并返回
        return jsonify({'status': 'success', 'paths': list(video_paths)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    """首页路由，提供index.html文件"""
    return send_from_directory('.', 'index.html')

# 修复静态文件服务路由，避免与API端点冲突
@app.route('/<path:path>')
def serve_static(path):
    """提供静态文件服务"""
    # 检查路径是否是静态文件（非API端点）
    if path.startswith('static/') or path.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg')):
        return send_from_directory('.', path)
    else:
        # 如果不是已知的静态文件类型，则返回index.html（支持SPA路由）
        return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)