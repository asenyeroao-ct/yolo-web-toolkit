# 🛠️ YOLO Web Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

一個功能完整的 Web 工具集，用於 YOLO 模型轉換與訓練。支援 PyTorch、ONNX、TensorRT 格式互轉，以及自訂 YOLO 模型訓練（YOLO5/6/7/8/9/10/11/12），具備現代化界面、即時進度追蹤、訓練日誌顯示和多語言支援。

## ✨ 主要功能

### 🔄 模型轉換
- **PyTorch (.pt) → ONNX (.onnx)**: 將 PyTorch YOLO 模型轉換為 ONNX 格式
- **PyTorch (.pt) → TensorRT Engine (.engine)**: 將 PyTorch 模型轉換為 TensorRT Engine（自動進行兩步轉換）
- **ONNX (.onnx) → TensorRT Engine (.engine)**: 將 ONNX 模型轉換為 TensorRT Engine

### 🎓 模型訓練
- 支援多種 YOLO 版本：YOLO5、YOLO6、YOLO7、YOLO8、YOLO9、YOLO10、YOLO11、YOLO12
- 支援多種模型尺寸：Nano (n)、Small (s)、Medium (m)、Large (l)、XLarge (x)
- 可自訂訓練參數：Epochs、Batch Size、Image Size
- 支援繼續訓練（Resume Training）

## 🎨 特性

- 🎨 **現代化的 Web 界面** - 直觀易用的用戶界面
- 📁 **模型上傳和管理** - 輕鬆上傳和管理模型文件
- 📂 **自定義輸出資料夾** - 靈活選擇輸出位置
- 📏 **多種預設輸出尺寸** - 支援 128, 160, 256, 320, 480, 640
- ➕ **支援自定義輸出尺寸** - 可自訂任意尺寸（64-2048）
- ⚙️ **TensorRT 進階選項** - FP16、FP8、工作空間大小等配置
- 📊 **即時轉換進度顯示** - 實時追蹤轉換和訓練進度
- 🔄 **Tab 分頁導航** - 清晰的頁面組織
- 🌐 **多語言支援** - 支援英文、繁體中文、簡體中文
- 🚀 **一鍵啟動** - 簡單的安裝和啟動流程

## 📋 系統需求

- **作業系統**: Windows 10/11
- **Python**: 3.8 或更高版本
- **GPU** (可選): NVIDIA GPU（用於 TensorRT 轉換）
- **CUDA** (可選): CUDA Toolkit（用於 TensorRT 轉換）
- **TensorRT** (可選): NVIDIA TensorRT（用於 Engine 轉換）

## 🚀 快速開始

### 方法 1：使用自動設置腳本（推薦）

1. **運行設置腳本**（Windows）：
   ```bash
   setup.bat
   ```
   此腳本會自動：
   - 創建 Python 虛擬環境 (venv)
   - 升級 pip
   - 安裝所有必需的依賴

2. **啟動應用**：
   ```bash
   start.bat
   ```
   或手動激活虛擬環境後啟動：
   ```bash
   venv\Scripts\activate
   python backend\app.py
   ```

### 方法 2：手動安裝

1. **創建虛擬環境**（可選但推薦）：
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **升級 pip**：
   ```bash
   python -m pip install --upgrade pip
   ```

3. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

4. **啟動應用**：
   ```bash
   python backend\app.py
   ```

### 安裝 TensorRT（可選，僅用於 Engine 轉換）

如果您需要進行 TensorRT Engine 轉換，請確保已安裝：
- NVIDIA TensorRT
- PyCUDA
- CUDA Toolkit

詳細安裝指南請參考 [TensorRT 官方文檔](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

## 📖 使用方法

### 啟動服務器

**方式 1：使用啟動腳本**（Windows）
```bash
start.bat
```

**方式 2：手動啟動**
```bash
# 如果使用虛擬環境，先激活
venv\Scripts\activate  # Windows

# 啟動服務器
python backend\app.py
```

服務器將在 `http://127.0.0.1:5000` 啟動。

### 使用 Web 界面

#### 模型轉換

1. 打開瀏覽器訪問 `http://127.0.0.1:5000`
2. 在「模型轉換」頁面：
   - 選擇轉換類型（PT→ONNX、PT→Engine、ONNX→Engine）
   - 選擇或上傳模型文件
   - 選擇輸出資料夾
   - 選擇輸出尺寸（可多選）
   - 配置 TensorRT 選項（如需要）
   - 點擊「開始轉換」

#### 模型訓練

1. 切換到「訓練模型」頁面
2. 配置模型資訊：
   - 選擇 YOLO 版本和模型尺寸
   - 選擇圖像和標籤資料夾
   - 設定輸出目的地
3. 配置訓練參數：
   - 設定 Epochs、Batch Size、Image Size
   - 選擇是否繼續訓練
4. 點擊「開始訓練」

#### 支援頁面

在「支援」頁面可以查看：
- GitHub 連結
- 項目貢獻人員
- Discord 社群連結

## 📁 項目結構

```
yolo-web-toolkit/
├── backend/                 # 後端應用
│   └── app.py              # Flask 後端主程序
├── converters/             # 轉換器模塊
│   ├── pt_to_onnx.py      # PT 到 ONNX 轉換
│   └── onnx_to_tensorrt.py # ONNX 到 TensorRT 轉換
├── training/               # 訓練模塊
│   └── train_yolo.py      # YOLO 模型訓練
├── static/                 # 前端靜態文件
│   ├── index.html         # 主頁面
│   ├── style.css          # 樣式文件
│   ├── app.js             # JavaScript 邏輯
│   ├── i18n.js            # 國際化支持
│   └── locales/           # 語言文件
│       ├── en.json        # 英文
│       ├── zh-TW.json     # 繁體中文
│       └── zh-CN.json     # 簡體中文
├── uploads/               # 上傳的模型文件（自動創建）
├── outputs/               # 轉換輸出文件（自動創建）
├── requirements.txt       # Python 依賴
├── setup.bat             # 自動設置腳本
├── start.bat             # 啟動腳本
└── README.md             # 項目說明
```

## 🔌 API 端點

### 模型管理
- `GET /api/models` - 獲取可用的模型列表
- `POST /api/upload` - 上傳模型文件
- `POST /api/upload-folder` - 上傳文件夾（用於訓練數據）

### 模型轉換
- `POST /api/convert` - 開始模型轉換
- `GET /api/task/<task_id>` - 獲取轉換任務狀態

### 模型訓練
- `POST /api/train` - 開始模型訓練
- `GET /api/train/<task_id>` - 獲取訓練任務狀態

### 工具
- `GET /api/folders` - 獲取可用的輸出資料夾列表

## ⚠️ 注意事項

- 確保有足夠的磁碟空間用於轉換輸出
- TensorRT 轉換需要 NVIDIA GPU 和 CUDA 支援
- 大型模型轉換可能需要較長時間
- 建議在轉換前備份原始模型文件
- 訓練大型模型需要足夠的 GPU 記憶體

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

### 貢獻指南

1. Fork 本專案
2. 創建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

## 📝 授權

本專案採用 MIT License - 詳見 [LICENSE](LICENSE) 文件

## 🔗 相關連結

- [GitHub Repository](https://github.com/asenyeroao-ct/yolo-web-toolkit)
- [Issues](https://github.com/asenyeroao-ct/yolo-web-toolkit/issues)
- [Discord Community](https://discord.gg/7dwUjfbP)

## 🙏 致謝

感謝所有為本專案做出貢獻的開發者和使用者！

---

**⭐ 如果這個專案對您有幫助，請給我們一個 Star！**
