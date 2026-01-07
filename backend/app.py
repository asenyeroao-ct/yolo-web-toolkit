"""
Flask 後端應用程式
提供 YOLO 模型轉換的 Web API
"""

import os
import json
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 導入轉換模塊
import sys

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from converters.onnx_to_tensorrt import convert_onnx_to_engine, ConversionConfig
from converters.pt_to_onnx import convert_pt_to_onnx
from training.train_yolo import train_yolo_model

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# 配置
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'outputs')
ALLOWED_EXTENSIONS = {'pt', 'pth', 'onnx', 'engine'}

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 轉換任務狀態
conversion_tasks = {}
task_lock = threading.Lock()

# 訓練任務狀態
training_tasks = {}
training_lock = threading.Lock()


def parse_training_metrics(log_content):
    """
    從訓練日誌中解析訓練指標
    
    解析格式：
    epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    1/100    2.5G      0.123      0.456      0.789    1234           640
    """
    import re
    metrics = []
    
    if not log_content:
        return metrics
    
    # 尋找指標標題行（更靈活的匹配，處理前導空格）
    lines = log_content.split('\n')
    header_line_idx = None
    
    for i, line in enumerate(lines):
        # 檢查是否包含指標標題（允許不同的空格和格式，忽略前導空格）
        stripped_line = line.strip()
        if re.search(r'epoch.*GPU_mem.*box_loss.*cls_loss.*dfl_loss.*Instances.*Size', stripped_line, re.IGNORECASE):
            header_line_idx = i
            break
    
    if header_line_idx is None:
        return metrics
    
    # 從標題行之後解析數據行
    for i in range(header_line_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # 跳過非數據行（包含字母但不符合格式的行）
        if re.search(r'[a-zA-Z]{3,}', line) and not re.search(r'^\d+/\d+', line):
            continue
        
        # 嘗試匹配指標行（更靈活的匹配，允許不同的空格）
        # 格式: epoch GPU_mem box_loss cls_loss dfl_loss Instances Size
        # 例如: 1/100    2.5G      0.123      0.456      0.789    1234           640
        # 或者: 1/100 2.5G 0.123 0.456 0.789 1234 640
        # 使用更寬鬆的匹配，允許可變空格
        pattern = r'^\s*(\d+/\d+)\s+([\d.]+[GMK]?)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+(\d+)\s+(\d+)\s*$'
        match = re.match(pattern, line)
        
        if match:
            epoch, gpu_mem, box_loss, cls_loss, dfl_loss, instances, size = match.groups()
            try:
                metrics.append({
                    'epoch': epoch,
                    'gpu_mem': gpu_mem,
                    'box_loss': float(box_loss),
                    'cls_loss': float(cls_loss),
                    'dfl_loss': float(dfl_loss),
                    'instances': int(instances),
                    'size': int(size)
                })
            except (ValueError, IndexError) as e:
                # 如果解析失敗，跳過這一行
                continue
        else:
            # 嘗試更寬鬆的匹配（允許前導空格和可變空格）
            # 使用 split 方法作為備選
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # 檢查第一個部分是否是 epoch 格式 (數字/數字)
                    if re.match(r'^\d+/\d+$', parts[0]):
                        epoch = parts[0]
                        gpu_mem = parts[1] if len(parts) > 1 else '0G'
                        box_loss = float(parts[2]) if len(parts) > 2 else 0.0
                        cls_loss = float(parts[3]) if len(parts) > 3 else 0.0
                        dfl_loss = float(parts[4]) if len(parts) > 4 else 0.0
                        instances = int(parts[5]) if len(parts) > 5 else 0
                        size = int(parts[6]) if len(parts) > 6 else 640
                        
                        metrics.append({
                            'epoch': epoch,
                            'gpu_mem': gpu_mem,
                            'box_loss': box_loss,
                            'cls_loss': cls_loss,
                            'dfl_loss': dfl_loss,
                            'instances': instances,
                            'size': size
                        })
                except (ValueError, IndexError):
                    continue
    
    return metrics


def allowed_file(filename):
    """檢查文件擴展名是否允許"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """提供主頁面"""
    return send_from_directory('../static', 'index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """獲取可用的模型列表"""
    models = []
    
    # 掃描上傳資料夾
    upload_path = Path(UPLOAD_FOLDER)
    for file in upload_path.glob('*.pt'):
        models.append({
            'name': file.name,
            'path': str(file),
            'type': 'pt',
            'size': file.stat().st_size
        })
    for file in upload_path.glob('*.pth'):
        models.append({
            'name': file.name,
            'path': str(file),
            'type': 'pt',
            'size': file.stat().st_size
        })
    for file in upload_path.glob('*.onnx'):
        models.append({
            'name': file.name,
            'path': str(file),
            'type': 'onnx',
            'size': file.stat().st_size
        })
    
    return jsonify({'models': models})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上傳模型文件"""
    if 'file' not in request.files:
        return jsonify({'error': '沒有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '沒有選擇文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
    
    return jsonify({'error': '不支持的文件類型'}), 400


@app.route('/api/upload-folder', methods=['POST'])
def upload_folder():
    """上傳文件夾（用於訓練數據）"""
    if 'files[]' not in request.files:
        return jsonify({'error': '沒有文件'}), 400
    
    files = request.files.getlist('files[]')
    folder_name = request.form.get('folder_name', '')
    folder_type = request.form.get('folder_type', 'images')  # 'images' or 'labels'
    
    if not folder_name:
        folder_name = f'{folder_type}_{len(files)}_files'
    
    # 創建文件夾
    folder_path = os.path.join(UPLOAD_FOLDER, folder_type, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(folder_path, filename)
            file.save(filepath)
            uploaded_files.append({
                'filename': filename,
                'path': filepath
            })
    
    return jsonify({
        'success': True,
        'folder_path': folder_path,
        'folder_name': folder_name,
        'files_count': len(uploaded_files),
        'files': uploaded_files
    })


@app.route('/api/convert', methods=['POST'])
def convert_model():
    """轉換模型"""
    data = request.json
    
    conversion_type = data.get('type')  # 'pt_to_onnx', 'pt_to_engine', 'onnx_to_engine'
    model_path = data.get('model_path')
    output_folder = data.get('output_folder', OUTPUT_FOLDER)
    imgsz = data.get('imgsz', 640)
    enable_fp16 = data.get('enable_fp16', False)
    enable_fp8 = data.get('enable_fp8', False)
    fixed_input_size = data.get('fixed_input_size', True)
    workspace_size_gb = data.get('workspace_size_gb', 1)
    tensorrt_version = data.get('tensorrt_version', '8.6')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 400
    
    # 生成任務 ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # 創建任務狀態
    with task_lock:
        conversion_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '開始轉換...',
            'output_path': None,
            'error': None
        }
    
    # 在後台線程中執行轉換
    def run_conversion():
        try:
            with task_lock:
                conversion_tasks[task_id]['message'] = '正在轉換...'
            
            if conversion_type == 'pt_to_onnx':
                # PT to ONNX
                model_name = Path(model_path).stem
                output_path = os.path.join(output_folder, f"{model_name}_{imgsz}.onnx")
                
                success, result_path = convert_pt_to_onnx(
                    pt_path=model_path,
                    onnx_path=output_path,
                    imgsz=imgsz,
                    verbose=True
                )
                
                if success:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'completed'
                        conversion_tasks[task_id]['progress'] = 100
                        conversion_tasks[task_id]['message'] = '轉換完成'
                        conversion_tasks[task_id]['output_path'] = result_path
                else:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'failed'
                        conversion_tasks[task_id]['error'] = '轉換失敗'
            
            elif conversion_type == 'pt_to_engine':
                # PT to Engine (先轉 ONNX，再轉 Engine)
                model_name = Path(model_path).stem
                onnx_path = os.path.join(output_folder, f"{model_name}_{imgsz}.onnx")
                engine_path = os.path.join(output_folder, f"{model_name}_{imgsz}.engine")
                
                # 步驟 1: PT to ONNX
                with task_lock:
                    conversion_tasks[task_id]['message'] = '步驟 1/2: 轉換 PT 到 ONNX...'
                    conversion_tasks[task_id]['progress'] = 30
                
                success, onnx_result = convert_pt_to_onnx(
                    pt_path=model_path,
                    onnx_path=onnx_path,
                    imgsz=imgsz,
                    verbose=True
                )
                
                if not success:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'failed'
                        conversion_tasks[task_id]['error'] = 'PT 到 ONNX 轉換失敗'
                    return
                
                # 步驟 2: ONNX to Engine
                with task_lock:
                    conversion_tasks[task_id]['message'] = '步驟 2/2: 轉換 ONNX 到 Engine...'
                    conversion_tasks[task_id]['progress'] = 60
                
                config = ConversionConfig(
                    enable_fp16=enable_fp16,
                    enable_fp8=enable_fp8,
                    fixed_input_size=fixed_input_size,
                    detection_resolution=imgsz,
                    workspace_size_gb=workspace_size_gb,
                    verbose=True
                )
                
                success, engine_result = convert_onnx_to_engine(
                    onnx_path=onnx_result,
                    engine_path=engine_path,
                    config=config
                )
                
                if success:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'completed'
                        conversion_tasks[task_id]['progress'] = 100
                        conversion_tasks[task_id]['message'] = '轉換完成'
                        conversion_tasks[task_id]['output_path'] = engine_result
                else:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'failed'
                        conversion_tasks[task_id]['error'] = 'ONNX 到 Engine 轉換失敗'
            
            elif conversion_type == 'onnx_to_engine':
                # ONNX to Engine
                model_name = Path(model_path).stem
                engine_path = os.path.join(output_folder, f"{model_name}_{imgsz}.engine")
                
                config = ConversionConfig(
                    enable_fp16=enable_fp16,
                    enable_fp8=enable_fp8,
                    fixed_input_size=fixed_input_size,
                    detection_resolution=imgsz,
                    workspace_size_gb=workspace_size_gb,
                    verbose=True
                )
                
                success, result_path = convert_onnx_to_engine(
                    onnx_path=model_path,
                    engine_path=engine_path,
                    config=config
                )
                
                if success:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'completed'
                        conversion_tasks[task_id]['progress'] = 100
                        conversion_tasks[task_id]['message'] = '轉換完成'
                        conversion_tasks[task_id]['output_path'] = result_path
                else:
                    with task_lock:
                        conversion_tasks[task_id]['status'] = 'failed'
                        conversion_tasks[task_id]['error'] = '轉換失敗'
            else:
                with task_lock:
                    conversion_tasks[task_id]['status'] = 'failed'
                    conversion_tasks[task_id]['error'] = f'不支持的轉換類型: {conversion_type}'
        
        except Exception as e:
            with task_lock:
                conversion_tasks[task_id]['status'] = 'failed'
                conversion_tasks[task_id]['error'] = str(e)
                import traceback
                conversion_tasks[task_id]['traceback'] = traceback.format_exc()
    
    thread = threading.Thread(target=run_conversion)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': task_id
    })


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """獲取轉換任務狀態"""
    with task_lock:
        if task_id not in conversion_tasks:
            return jsonify({'error': '任務不存在'}), 404
        
        return jsonify(conversion_tasks[task_id])


@app.route('/api/folders', methods=['GET'])
def get_folders():
    """獲取可用的輸出資料夾列表"""
    folders = []
    
    # 掃描專案根目錄下的資料夾
    current_path = project_root
    for item in current_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name not in ['backend', 'converters', 'training', '__pycache__', 'venv']:
            folders.append({
                'name': item.name,
                'path': str(item.absolute())
            })
    
    # 添加預設輸出資料夾
    folders.append({
        'name': 'outputs',
        'path': str(Path(OUTPUT_FOLDER).absolute())
    })
    
    return jsonify({'folders': folders})


@app.route('/api/train', methods=['POST'])
def train_model():
    """訓練模型"""
    data = request.json
    
    yolo_version = data.get('yolo_version', 'yolo12')
    model_size = data.get('model_size', 'n')
    images_folder = data.get('images_folder', '')
    labels_folder = data.get('labels_folder', '')
    output_destination = data.get('output_destination', '')
    epochs = data.get('epochs', 100)
    batch_size = data.get('batch_size', 16)
    imgsz = data.get('imgsz', 640)
    resume = data.get('resume', False)
    
    # 驗證路徑（允許相對路徑或絕對路徑）
    # 如果路徑不存在，嘗試在當前工作目錄下查找
    if images_folder:
        if not os.path.isabs(images_folder):
            # 相對路徑，嘗試在 uploads 或當前目錄查找
            possible_paths = [
                os.path.join(UPLOAD_FOLDER, images_folder),
                os.path.join('.', images_folder),
                images_folder
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    images_folder = os.path.abspath(path)
                    break
        else:
            images_folder = os.path.abspath(images_folder)
    
    if labels_folder:
        if not os.path.isabs(labels_folder):
            possible_paths = [
                os.path.join(UPLOAD_FOLDER, labels_folder),
                os.path.join('.', labels_folder),
                labels_folder
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    labels_folder = os.path.abspath(path)
                    break
        else:
            labels_folder = os.path.abspath(labels_folder)
    
    if output_destination:
        if not os.path.isabs(output_destination):
            output_destination = os.path.abspath(os.path.join(OUTPUT_FOLDER, output_destination))
        else:
            output_destination = os.path.abspath(output_destination)
    
    if not images_folder or not os.path.exists(images_folder):
        return jsonify({'error': f'Images folder does not exist: {images_folder}'}), 400
    
    if not labels_folder or not os.path.exists(labels_folder):
        return jsonify({'error': f'Labels folder does not exist: {labels_folder}'}), 400
    
    # 生成任務 ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # 創建任務狀態
    with training_lock:
        training_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting training...',
            'output_path': None,
            'error': None,
            'log': ''
        }
    
    # 在後台線程中執行訓練
    def run_training():
        import sys
        import time
        from io import StringIO
        
        # 捕獲標準輸出
        log_buffer = StringIO()
        old_stdout = sys.stdout
        
        # 創建一個標誌來控制日誌更新線程
        stop_logging = threading.Event()
        
        def update_log_periodically():
            """定期更新日誌和指標"""
            while not stop_logging.is_set():
                time.sleep(2)  # 每2秒更新一次
                try:
                    log_content = log_buffer.getvalue()
                    if log_content:
                        metrics = parse_training_metrics(log_content)
                        with training_lock:
                            if task_id in training_tasks:
                                training_tasks[task_id]['log'] = log_content
                                training_tasks[task_id]['metrics'] = metrics
                                # 根據指標數量估算進度
                                if metrics:
                                    try:
                                        last_epoch = metrics[-1]['epoch']
                                        current, total = map(int, last_epoch.split('/'))
                                        progress = int((current / total) * 90) + 10  # 10-100%
                                        training_tasks[task_id]['progress'] = min(progress, 95)
                                    except:
                                        pass
                except Exception as e:
                    # 忽略更新錯誤，繼續運行
                    pass
        
        # 啟動日誌更新線程
        log_thread = threading.Thread(target=update_log_periodically)
        log_thread.daemon = True
        log_thread.start()
        
        try:
            with training_lock:
                training_tasks[task_id]['message'] = 'Training in progress...'
                training_tasks[task_id]['progress'] = 10
                training_tasks[task_id]['log'] = '[訓練] 開始訓練 YOLO 模型...\n'
                training_tasks[task_id]['metrics'] = []
            
            # 重定向標準輸出到緩衝區
            sys.stdout = log_buffer
            
            success, model_path, results = train_yolo_model(
                yolo_version=yolo_version,
                model_size=model_size,
                images_folder=images_folder,
                labels_folder=labels_folder,
                output_destination=output_destination,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                resume=resume,
                verbose=True
            )
            
            # 停止日誌更新線程
            stop_logging.set()
            
            # 恢復標準輸出
            sys.stdout = old_stdout
            
            # 獲取日誌內容
            log_content = log_buffer.getvalue()
            
            # 解析訓練指標
            metrics = parse_training_metrics(log_content)
            
            if success:
                with training_lock:
                    training_tasks[task_id]['status'] = 'completed'
                    training_tasks[task_id]['progress'] = 100
                    training_tasks[task_id]['message'] = 'Training complete!'
                    training_tasks[task_id]['output_path'] = model_path
                    training_tasks[task_id]['log'] = log_content
                    training_tasks[task_id]['metrics'] = metrics
                    if results:
                        training_tasks[task_id]['results'] = results
            else:
                with training_lock:
                    training_tasks[task_id]['status'] = 'failed'
                    training_tasks[task_id]['error'] = 'Training failed'
                    training_tasks[task_id]['log'] = log_content
                    training_tasks[task_id]['metrics'] = metrics
        
        except Exception as e:
            # 停止日誌更新線程
            stop_logging.set()
            
            # 恢復標準輸出
            sys.stdout = old_stdout
            
            # 獲取日誌內容
            log_content = log_buffer.getvalue()
            
            # 解析最終指標
            metrics = parse_training_metrics(log_content)
            
            with training_lock:
                training_tasks[task_id]['status'] = 'failed'
                training_tasks[task_id]['error'] = str(e)
                import traceback
                training_tasks[task_id]['traceback'] = traceback.format_exc()
                training_tasks[task_id]['log'] = log_content + '\n' + traceback.format_exc()
                training_tasks[task_id]['metrics'] = metrics
    
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': task_id
    })


@app.route('/api/train/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """獲取訓練任務狀態"""
    with training_lock:
        if task_id not in training_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = training_tasks[task_id].copy()
        
        # 如果任務正在運行，嘗試從日誌中解析最新指標
        if task['status'] == 'running' and task.get('log'):
            metrics = parse_training_metrics(task['log'])
            # 始終設置 metrics 字段，即使為空數組
            task['metrics'] = metrics
            # 根據指標數量估算進度
            if metrics:
                try:
                    last_epoch = metrics[-1]['epoch']
                    current, total = map(int, last_epoch.split('/'))
                    progress = int((current / total) * 90) + 10  # 10-100%
                    task['progress'] = min(progress, 95)
                except:
                    pass
        elif 'metrics' not in task:
            # 確保即使沒有日誌也有 metrics 字段
            task['metrics'] = []
        
        return jsonify(task)


if __name__ == '__main__':
    print("啟動 YOLO Web Toolkit...")
    print("訪問 http://127.0.0.1:5000 使用界面")
    app.run(debug=True, host='0.0.0.0', port=5000)

