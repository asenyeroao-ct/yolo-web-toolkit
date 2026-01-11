# 🛠️ YOLO Web Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

**English** | [简体中文](#简体中文)

---

## English

A comprehensive web toolkit for YOLO model conversion and training. Supports conversion between PyTorch, ONNX, and TensorRT formats, as well as custom YOLO model training (YOLO5/6/7/8/9/10/11/12), featuring a modern interface, real-time progress tracking, training log display, and multi-language support.

### ✨ Main Features

#### 🔄 Model Conversion
- **PyTorch (.pt) → ONNX (.onnx)**: Convert PyTorch YOLO models to ONNX format
- **PyTorch (.pt) → TensorRT Engine (.engine)**: Convert PyTorch models to TensorRT Engine (automatic two-step conversion)
- **ONNX (.onnx) → TensorRT Engine (.engine)**: Convert ONNX models to TensorRT Engine

#### 🎓 Model Training
- Support for multiple YOLO versions: YOLO5, YOLO6, YOLO7, YOLO8, YOLO9, YOLO10, YOLO11, YOLO12
- Support for multiple model sizes: Nano (n), Small (s), Medium (m), Large (l), XLarge (x)
- Customizable training parameters: Epochs, Batch Size, Image Size
- Support for resume training

### 🎨 Features

- 🎨 **Modern Web Interface** - Intuitive and user-friendly interface
- 📁 **Model Upload and Management** - Easily upload and manage model files
- 📂 **Custom Output Folders** - Flexible output location selection
- 📏 **Multiple Preset Output Sizes** - Support for 128, 160, 256, 320, 480, 640
- ➕ **Custom Output Size Support** - Customize any size (64-2048)
- ⚙️ **TensorRT Advanced Options** - FP16, FP8, workspace size, and other configurations
- 📊 **Real-time Conversion Progress** - Real-time tracking of conversion and training progress
- 🔄 **Tab Navigation** - Clear page organization
- 🌐 **Multi-language Support** - Support for English, Traditional Chinese, Simplified Chinese
- 🚀 **One-click Launch** - Simple installation and startup process

### 📋 System Requirements

- **Operating System**: Windows 10/11
- **Python**: 3.8 or higher
- **GPU** (Optional): NVIDIA GPU (for TensorRT conversion)
- **CUDA** (Optional): CUDA Toolkit (for TensorRT conversion)
- **TensorRT** (Optional): NVIDIA TensorRT (for Engine conversion)

### 🚀 Quick Start

#### Method 1: Using Auto Setup Script (Recommended)

1. **Run the setup script** (Windows):
   ```bash
   setup.bat
   ```
   This script will automatically:
   - Create a Python virtual environment (venv)
   - Upgrade pip
   - Install all required dependencies

2. **Start the application**:
   ```bash
   start.bat
   ```
   Or manually activate the virtual environment and start:
   ```bash
   venv\Scripts\activate
   python backend\app.py
   ```

#### Method 2: Manual Installation

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install dependencies**:
   
   Choose one of the following setup scripts based on your system:
   
   **For NVIDIA GPU (CUDA support)**:
   ```bash
   python setup/cuda_setup.py
   ```
   
   **For systems without NVIDIA GPU (DirectML support)**:
   ```bash
   python setup/directml_setup.py
   ```
   
   These scripts will automatically:
   - Create a Python virtual environment (venv)
   - Install all required dependencies including PyTorch with appropriate GPU support

4. **Start the application**:
   ```bash
   python backend\app.py
   ```

#### Installing TensorRT (Optional, for Engine conversion only)

If you need to perform TensorRT Engine conversion, make sure you have installed:
- NVIDIA TensorRT
- PyCUDA
- CUDA Toolkit

For detailed installation instructions, please refer to the [TensorRT Official Documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

### 📖 Usage

#### Starting the Server

**Method 1: Using the startup script** (Windows)
```bash
start.bat
```

**Method 2: Manual startup**
```bash
# If using a virtual environment, activate it first
venv\Scripts\activate  # Windows

# Start the server
python backend\app.py
```

The server will start at `http://127.0.0.1:5000`.

#### Using the Web Interface

##### Model Conversion

1. Open your browser and visit `http://127.0.0.1:5000`
2. On the "Model Conversion" page:
   - Select conversion type (PT→ONNX, PT→Engine, ONNX→Engine)
   - Select or upload model file
   - Select output folder
   - Select output sizes (multiple selection supported)
   - Configure TensorRT options (if needed)
   - Click "Start Conversion"

##### Model Training

1. Switch to the "Train Model" page
2. Configure model information:
   - Select YOLO version and model size
   - Select image and label folders
   - Set output destination
3. Configure training parameters:
   - Set Epochs, Batch Size, Image Size
   - Choose whether to resume training
4. Click "Start Training"

##### Support Page

On the "Support" page, you can view:
- GitHub link
- Project contributors
- Discord community link

### 📁 Project Structure

```
yolo-web-toolkit/
├── backend/                 # Backend application
│   └── app.py              # Flask backend main program
├── converters/             # Converter modules
│   ├── pt_to_onnx.py      # PT to ONNX conversion
│   └── onnx_to_tensorrt.py # ONNX to TensorRT conversion
├── training/               # Training module
│   └── train_yolo.py      # YOLO model training
├── setup/                  # Setup scripts
│   ├── cuda_setup.py      # CUDA environment setup script
│   └── directml_setup.py  # DirectML environment setup script
├── static/                 # Frontend static files
│   ├── index.html         # Main page
│   ├── style.css          # Style file
│   ├── app.js             # JavaScript logic
│   ├── i18n.js            # Internationalization support
│   └── locales/           # Language files
│       ├── en.json        # English
│       ├── zh-TW.json     # Traditional Chinese
│       └── zh-CN.json     # Simplified Chinese
├── uploads/               # Uploaded model files (auto-created)
├── outputs/               # Conversion output files (auto-created)
├── requirements.txt       # Python dependencies
├── setup.bat             # Auto setup script
├── start.bat             # Startup script
└── README.md             # Project documentation
```

### 🔌 API Endpoints

#### Model Management
- `GET /api/models` - Get available model list
- `POST /api/upload` - Upload model file
- `POST /api/upload-folder` - Upload folder (for training data)

#### Model Conversion
- `POST /api/convert` - Start model conversion
- `GET /api/task/<task_id>` - Get conversion task status

#### Model Training
- `POST /api/train` - Start model training
- `GET /api/train/<task_id>` - Get training task status

#### Tools
- `GET /api/folders` - Get available output folder list

### ⚠️ Notes

- Ensure sufficient disk space for conversion output
- TensorRT conversion requires NVIDIA GPU and CUDA support
- Large model conversion may take a long time
- It is recommended to backup original model files before conversion
- Training large models requires sufficient GPU memory

### 🤝 Contributing

Issues and Pull Requests are welcome!

#### Contribution Guidelines

1. Fork this project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🔗 Related Links

- [GitHub Repository](https://github.com/asenyeroao-ct/yolo-web-toolkit)
- [Issues](https://github.com/asenyeroao-ct/yolo-web-toolkit/issues)
- [Discord Community](https://discord.gg/7dwUjfbP)

### 🙏 Acknowledgments

Thanks to all developers and users who have contributed to this project!

---

**⭐ If this project is helpful to you, please give us a Star!**

---

## 简体中文

一个功能完整的 Web 工具集，用于 YOLO 模型转换与训练。支持 PyTorch、ONNX、TensorRT 格式互转，以及自定义 YOLO 模型训练（YOLO5/6/7/8/9/10/11/12），具备现代化界面、实时进度追踪、训练日志显示和多语言支持。

### ✨ 主要功能

#### 🔄 模型转换
- **PyTorch (.pt) → ONNX (.onnx)**: 将 PyTorch YOLO 模型转换为 ONNX 格式
- **PyTorch (.pt) → TensorRT Engine (.engine)**: 将 PyTorch 模型转换为 TensorRT Engine（自动进行两步转换）
- **ONNX (.onnx) → TensorRT Engine (.engine)**: 将 ONNX 模型转换为 TensorRT Engine

#### 🎓 模型训练
- 支持多种 YOLO 版本：YOLO5、YOLO6、YOLO7、YOLO8、YOLO9、YOLO10、YOLO11、YOLO12
- 支持多种模型尺寸：Nano (n)、Small (s)、Medium (m)、Large (l)、XLarge (x)
- 可自定义训练参数：Epochs、Batch Size、Image Size
- 支持继续训练（Resume Training）

### 🎨 特性

- 🎨 **现代化的 Web 界面** - 直观易用的用户界面
- 📁 **模型上传和管理** - 轻松上传和管理模型文件
- 📂 **自定义输出文件夹** - 灵活选择输出位置
- 📏 **多种预设输出尺寸** - 支持 128, 160, 256, 320, 480, 640
- ➕ **支持自定义输出尺寸** - 可自定义任意尺寸（64-2048）
- ⚙️ **TensorRT 进阶选项** - FP16、FP8、工作空间大小等配置
- 📊 **实时转换进度显示** - 实时追踪转换和训练进度
- 🔄 **Tab 分页导航** - 清晰的页面组织
- 🌐 **多语言支持** - 支持英文、繁体中文、简体中文
- 🚀 **一键启动** - 简单的安装和启动流程

### 📋 系统需求

- **操作系统**: Windows 10/11
- **Python**: 3.8 或更高版本
- **GPU** (可选): NVIDIA GPU（用于 TensorRT 转换）
- **CUDA** (可选): CUDA Toolkit（用于 TensorRT 转换）
- **TensorRT** (可选): NVIDIA TensorRT（用于 Engine 转换）

### 🚀 快速开始

#### 方法 1：使用自动设置脚本（推荐）

1. **运行设置脚本**（Windows）：
   ```bash
   setup.bat
   ```
   此脚本会自动：
   - 创建 Python 虚拟环境 (venv)
   - 升级 pip
   - 安装所有必需的依赖

2. **启动应用**：
   ```bash
   start.bat
   ```
   或手动激活虚拟环境后启动：
   ```bash
   venv\Scripts\activate
   python backend\app.py
   ```

#### 方法 2：手动安装

1. **创建虚拟环境**（可选但推荐）：
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **升级 pip**：
   ```bash
   python -m pip install --upgrade pip
   ```

3. **安装依赖**：
   
   根据您的系统选择以下安装脚本之一：
   
   **适用于 NVIDIA GPU（CUDA 支持）**：
   ```bash
   python setup/cuda_setup.py
   ```
   
   **适用于没有 NVIDIA GPU 的系统（DirectML 支持）**：
   ```bash
   python setup/directml_setup.py
   ```
   
   这些脚本将自动：
   - 创建 Python 虚拟环境 (venv)
   - 安装所有必需的依赖，包括带有相应 GPU 支持的 PyTorch

4. **启动应用**：
   ```bash
   python backend\app.py
   ```

#### 安装 TensorRT（可选，仅用于 Engine 转换）

如果您需要进行 TensorRT Engine 转换，请确保已安装：
- NVIDIA TensorRT
- PyCUDA
- CUDA Toolkit

详细安装指南请参考 [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

### 📖 使用方法

#### 启动服务器

**方式 1：使用启动脚本**（Windows）
```bash
start.bat
```

**方式 2：手动启动**
```bash
# 如果使用虚拟环境，先激活
venv\Scripts\activate  # Windows

# 启动服务器
python backend\app.py
```

服务器将在 `http://127.0.0.1:5000` 启动。

#### 使用 Web 界面

##### 模型转换

1. 打开浏览器访问 `http://127.0.0.1:5000`
2. 在「模型转换」页面：
   - 选择转换类型（PT→ONNX、PT→Engine、ONNX→Engine）
   - 选择或上传模型文件
   - 选择输出文件夹
   - 选择输出尺寸（可多选）
   - 配置 TensorRT 选项（如需要）
   - 点击「开始转换」

##### 模型训练

1. 切换到「训练模型」页面
2. 配置模型信息：
   - 选择 YOLO 版本和模型尺寸
   - 选择图像和标签文件夹
   - 设定输出目的地
3. 配置训练参数：
   - 设定 Epochs、Batch Size、Image Size
   - 选择是否继续训练
4. 点击「开始训练」

##### 支持页面

在「支持」页面可以查看：
- GitHub 链接
- 项目贡献人员
- Discord 社群链接

### 📁 项目结构

```
yolo-web-toolkit/
├── backend/                 # 后端应用
│   └── app.py              # Flask 后端主程序
├── converters/             # 转换器模块
│   ├── pt_to_onnx.py      # PT 到 ONNX 转换
│   └── onnx_to_tensorrt.py # ONNX 到 TensorRT 转换
├── training/               # 训练模块
│   └── train_yolo.py      # YOLO 模型训练
├── setup/                  # 安装脚本
│   ├── cuda_setup.py      # CUDA 环境安装脚本
│   └── directml_setup.py  # DirectML 环境安装脚本
├── static/                 # 前端静态文件
│   ├── index.html         # 主页面
│   ├── style.css          # 样式文件
│   ├── app.js             # JavaScript 逻辑
│   ├── i18n.js            # 国际化支持
│   └── locales/           # 语言文件
│       ├── en.json        # 英文
│       ├── zh-TW.json     # 繁体中文
│       └── zh-CN.json     # 简体中文
├── uploads/               # 上传的模型文件（自动创建）
├── outputs/               # 转换输出文件（自动创建）
├── requirements.txt       # Python 依赖
├── setup.bat             # 自动设置脚本
├── start.bat             # 启动脚本
└── README.md             # 项目说明
```

### 🔌 API 端点

#### 模型管理
- `GET /api/models` - 获取可用的模型列表
- `POST /api/upload` - 上传模型文件
- `POST /api/upload-folder` - 上传文件夹（用于训练数据）

#### 模型转换
- `POST /api/convert` - 开始模型转换
- `GET /api/task/<task_id>` - 获取转换任务状态

#### 模型训练
- `POST /api/train` - 开始模型训练
- `GET /api/train/<task_id>` - 获取训练任务状态

#### 工具
- `GET /api/folders` - 获取可用的输出文件夹列表

### ⚠️ 注意事项

- 确保有足够的磁盘空间用于转换输出
- TensorRT 转换需要 NVIDIA GPU 和 CUDA 支持
- 大型模型转换可能需要较长时间
- 建议在转换前备份原始模型文件
- 训练大型模型需要足够的 GPU 内存

### 🤝 贡献

欢迎提交 Issue 和 Pull Request！

#### 贡献指南

1. Fork 本项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 📝 授权

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

### 🔗 相关链接

- [GitHub Repository](https://github.com/asenyeroao-ct/yolo-web-toolkit)
- [Issues](https://github.com/asenyeroao-ct/yolo-web-toolkit/issues)
- [Discord Community](https://discord.gg/7dwUjfbP)

### 🙏 致谢

感谢所有为本项目做出贡献的开发者和使用者！

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**
