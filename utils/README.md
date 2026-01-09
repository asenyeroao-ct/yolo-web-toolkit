# 🛠️ Utils Module Documentation

**English** | [繁體中文](#繁體中文)

---

## English

This directory contains utility modules inspired by the **hank-ai/darknet** project, providing model validation, analysis, and evaluation capabilities.

### 📦 Module List

#### 1. Model Validator (`model_validator.py`)

Provides model format validation, structure checking, and inference testing functionality.

**Features:**
- ✅ Validate PyTorch (.pt/.pth) models
- ✅ Validate ONNX (.onnx) models
- ✅ Validate TensorRT Engine (.engine) models
- ✅ Check model structure integrity
- ✅ Get input/output shape information
- ✅ Detect model errors and warnings

**Usage Example:**
```python
from utils.model_validator import validate_model

result = validate_model("model.pt")
if result.is_valid:
    print(f"Model is valid, input shape: {result.input_shape}")
else:
    print(f"Model is invalid, errors: {result.errors}")
```

**Command Line Usage:**
```bash
python utils/model_validator.py model.pt --verbose
```

#### 2. Model Analyzer (`model_analyzer.py`)

Provides model structure analysis, parameter statistics, and performance evaluation functionality.

**Features:**
- ✅ Analyze model file size
- ✅ Count model parameters
- ✅ Get model input/output information
- ✅ Extract model metadata
- ✅ Analyze model architecture

**Usage Example:**
```python
from utils.model_analyzer import analyze_model

info = analyze_model("model.pt")
print(f"File size: {info.file_size_mb:.2f} MB")
print(f"Parameter count: {info.num_parameters:,}")
print(f"Number of classes: {info.num_classes}")
```

**Command Line Usage:**
```bash
python utils/model_analyzer.py model.pt --json
```

#### 3. mAP Calculator (`map_calculator.py`)

Provides model evaluation and mAP (Mean Average Precision) calculation functionality.

**Features:**
- ✅ Calculate mAP@0.5
- ✅ Calculate mAP@0.5:0.95
- ✅ Calculate precision and recall
- ✅ Per-class mAP statistics
- ✅ Other evaluation metrics (F1-score, etc.)

**Usage Example:**
```python
from utils.map_calculator import calculate_map

result = calculate_map(
    model_path="model.pt",
    data_yaml="data.yaml",
    conf_threshold=0.25,
    iou_threshold=0.45
)

print(f"mAP@0.5: {result.map_50:.4f}")
print(f"mAP@0.5:0.95: {result.map_50_95:.4f}")
```

**Command Line Usage:**
```bash
python utils/map_calculator.py model.pt data.yaml --conf 0.25 --iou 0.45
```

### 🔌 API Endpoints

These features are integrated into the backend API:

#### Validate Model
```http
POST /api/validate
Content-Type: application/json

{
  "model_path": "path/to/model.pt"
}
```

#### Analyze Model
```http
POST /api/analyze
Content-Type: application/json

{
  "model_path": "path/to/model.pt"
}
```

#### Calculate mAP
```http
POST /api/map
Content-Type: application/json

{
  "model_path": "path/to/model.pt",
  "data_yaml": "path/to/data.yaml",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45
}
```

### 📋 Dependencies

These utility modules require the following dependencies:

- `torch` - PyTorch model support
- `ultralytics` - YOLO model support
- `onnx` - ONNX model support
- `onnxruntime` - ONNX inference support
- `tensorrt` - TensorRT Engine support (optional)
- `numpy` - Numerical computation
- `pyyaml` - YAML file parsing

All dependencies are included in `requirements.txt`.

### 🎯 Reference

The design inspiration for these utility modules comes from:
- [hank-ai/darknet](https://github.com/hank-ai/darknet) - Darknet/YOLO object detection framework
- Particularly the model validation, analysis, and evaluation features

### 📝 Notes

1. **Model Validation**: Validation will attempt to load the model, which may take some time for large models
2. **mAP Calculation**: Requires a correctly formatted `data.yaml` file (YOLO format)
3. **TensorRT Engine**: Validating and analyzing TensorRT Engine requires CUDA and TensorRT environment
4. **Performance**: Some operations (such as model analysis) may take a long time, recommended to run in background

### 🔄 Future Improvements

Planned features:
- [ ] Model inference speed testing
- [ ] Model precision comparison (FP32 vs FP16 vs FP8)
- [ ] Batch model validation
- [ ] Model visualization (structure diagram)
- [ ] More detailed performance analysis

---

## 繁體中文

本目錄包含從 **hank-ai/darknet** 項目中汲取靈感而開發的實用工具模塊，提供模型驗證、分析和評估功能。

### 📦 模塊列表

#### 1. 模型驗證 (`model_validator.py`)

提供模型格式驗證、結構檢查和推理測試功能。

**功能：**
- ✅ 驗證 PyTorch (.pt/.pth) 模型
- ✅ 驗證 ONNX (.onnx) 模型
- ✅ 驗證 TensorRT Engine (.engine) 模型
- ✅ 檢查模型結構完整性
- ✅ 獲取輸入輸出形狀信息
- ✅ 檢測模型錯誤和警告

**使用範例：**
```python
from utils.model_validator import validate_model

result = validate_model("model.pt")
if result.is_valid:
    print(f"模型有效，輸入形狀: {result.input_shape}")
else:
    print(f"模型無效，錯誤: {result.errors}")
```

**命令行使用：**
```bash
python utils/model_validator.py model.pt --verbose
```

#### 2. 模型分析 (`model_analyzer.py`)

提供模型結構分析、參數統計和性能評估功能。

**功能：**
- ✅ 分析模型文件大小
- ✅ 統計模型參數數量
- ✅ 獲取模型輸入輸出信息
- ✅ 提取模型元數據
- ✅ 分析模型架構

**使用範例：**
```python
from utils.model_analyzer import analyze_model

info = analyze_model("model.pt")
print(f"文件大小: {info.file_size_mb:.2f} MB")
print(f"參數數量: {info.num_parameters:,}")
print(f"類別數量: {info.num_classes}")
```

**命令行使用：**
```bash
python utils/model_analyzer.py model.pt --json
```

#### 3. mAP 計算 (`map_calculator.py`)

提供模型評估和 mAP (Mean Average Precision) 計算功能。

**功能：**
- ✅ 計算 mAP@0.5
- ✅ 計算 mAP@0.5:0.95
- ✅ 計算精確度和召回率
- ✅ 每個類別的 mAP 統計
- ✅ 其他評估指標（F1-score 等）

**使用範例：**
```python
from utils.map_calculator import calculate_map

result = calculate_map(
    model_path="model.pt",
    data_yaml="data.yaml",
    conf_threshold=0.25,
    iou_threshold=0.45
)

print(f"mAP@0.5: {result.map_50:.4f}")
print(f"mAP@0.5:0.95: {result.map_50_95:.4f}")
```

**命令行使用：**
```bash
python utils/map_calculator.py model.pt data.yaml --conf 0.25 --iou 0.45
```

### 🔌 API 端點

這些功能已整合到後端 API 中：

#### 驗證模型
```http
POST /api/validate
Content-Type: application/json

{
  "model_path": "path/to/model.pt"
}
```

#### 分析模型
```http
POST /api/analyze
Content-Type: application/json

{
  "model_path": "path/to/model.pt"
}
```

#### 計算 mAP
```http
POST /api/map
Content-Type: application/json

{
  "model_path": "path/to/model.pt",
  "data_yaml": "path/to/data.yaml",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45
}
```

### 📋 依賴要求

這些工具模塊需要以下依賴：

- `torch` - PyTorch 模型支持
- `ultralytics` - YOLO 模型支持
- `onnx` - ONNX 模型支持
- `onnxruntime` - ONNX 推理支持
- `tensorrt` - TensorRT Engine 支持（可選）
- `numpy` - 數值計算
- `pyyaml` - YAML 文件解析

所有依賴已包含在 `requirements.txt` 中。

### 🎯 參考來源

這些工具模塊的設計靈感來自：
- [hank-ai/darknet](https://github.com/hank-ai/darknet) - Darknet/YOLO 物件檢測框架
- 特別是其中的模型驗證、分析和評估功能

### 📝 注意事項

1. **模型驗證**：驗證功能會嘗試載入模型，對於大型模型可能需要較長時間
2. **mAP 計算**：需要提供正確格式的 `data.yaml` 文件（YOLO 格式）
3. **TensorRT Engine**：驗證和分析 TensorRT Engine 需要 CUDA 和 TensorRT 環境
4. **性能**：某些操作（如模型分析）可能需要較長時間，建議在後台執行

### 🔄 未來改進

計劃添加的功能：
- [ ] 模型推理速度測試
- [ ] 模型精度對比（FP32 vs FP16 vs FP8）
- [ ] 批量模型驗證
- [ ] 模型可視化（結構圖）
- [ ] 更詳細的性能分析
