"""
模型分析模塊

提供模型結構分析、參數統計和性能評估功能
參考 hank-ai/darknet 的模型分析方法
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """模型信息"""
    model_path: str
    """模型文件路徑"""
    
    model_type: str
    """模型類型 (pt/onnx/engine)"""
    
    file_size_mb: float
    """文件大小 (MB)"""
    
    input_shape: Optional[tuple] = None
    """輸入形狀"""
    
    output_shape: Optional[tuple] = None
    """輸出形狀"""
    
    num_parameters: Optional[int] = None
    """參數數量"""
    
    num_classes: Optional[int] = None
    """類別數量"""
    
    opset_version: Optional[int] = None
    """ONNX opset 版本"""
    
    precision: Optional[str] = None
    """精度 (fp32/fp16/fp8)"""
    
    metadata: Dict = None
    """額外元數據"""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def analyze_pt_model(model_path: str) -> ModelInfo:
    """
    分析 PyTorch 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        模型信息
    """
    info = ModelInfo(
        model_path=model_path,
        model_type='pt',
        file_size_mb=os.path.getsize(model_path) / (1024 * 1024)
    )
    
    try:
        import torch
        from ultralytics import YOLO
        
        # 載入模型
        model = YOLO(model_path)
        
        # 獲取模型信息
        if hasattr(model, 'model'):
            # 計算參數數量
            try:
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                info.num_parameters = total_params
                info.metadata['trainable_parameters'] = trainable_params
                info.metadata['non_trainable_parameters'] = total_params - trainable_params
            except:
                pass
        
        # 獲取類別信息
        if hasattr(model, 'names'):
            info.num_classes = len(model.names)
            info.metadata['class_names'] = list(model.names.values())
        
        # 獲取模型架構信息
        if hasattr(model, 'info'):
            try:
                model_info = model.info()
                info.metadata['model_info'] = model_info
            except:
                pass
                
    except Exception as e:
        info.metadata['error'] = str(e)
    
    return info


def analyze_onnx_model(model_path: str) -> ModelInfo:
    """
    分析 ONNX 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        模型信息
    """
    info = ModelInfo(
        model_path=model_path,
        model_type='onnx',
        file_size_mb=os.path.getsize(model_path) / (1024 * 1024)
    )
    
    try:
        import onnx
        
        # 載入模型
        model = onnx.load(model_path)
        
        # 獲取 opset 版本
        if model.opset_import:
            info.opset_version = model.opset_import[0].version
        
        # 獲取輸入輸出信息
        if len(model.graph.input) > 0:
            input_tensor = model.graph.input[0]
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(int(dim.dim_value))
                else:
                    shape.append(-1)
            info.input_shape = tuple(shape)
        
        if len(model.graph.output) > 0:
            output_tensor = model.graph.output[0]
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(int(dim.dim_value))
                else:
                    shape.append(-1)
            info.output_shape = tuple(shape)
        
        # 統計節點數量
        info.metadata['num_nodes'] = len(model.graph.node)
        info.metadata['num_inputs'] = len(model.graph.input)
        info.metadata['num_outputs'] = len(model.graph.output)
        
        # 獲取初始值數量
        info.metadata['num_initializers'] = len(model.graph.initializer)
        
        # 讀取元數據（包括 classes）
        if model.metadata_props:
            metadata_dict = {}
            for prop in model.metadata_props:
                metadata_dict[prop.key] = prop.value
            
            # 提取 classes 資訊
            if 'classes' in metadata_dict:
                # 如果有 classes 鍵，解析為列表
                classes_str = metadata_dict['classes']
                info.metadata['classes'] = classes_str.split(',') if classes_str else []
                info.metadata['num_classes'] = len(info.metadata['classes'])
            elif 'num_classes' in metadata_dict:
                # 如果有 num_classes，嘗試讀取 class_0, class_1, ...
                try:
                    num_classes = int(metadata_dict['num_classes'])
                    classes = []
                    for i in range(num_classes):
                        class_key = f'class_{i}'
                        if class_key in metadata_dict:
                            classes.append(metadata_dict[class_key])
                    if classes:
                        info.metadata['classes'] = classes
                        info.metadata['num_classes'] = len(classes)
                except ValueError:
                    pass
            
            # 保存所有其他元數據
            for key, value in metadata_dict.items():
                if not key.startswith('class'):  # 避免重複保存 class_0, class_1 等
                    if key not in ['classes', 'num_classes']:  # 這些已經處理過了
                        info.metadata[f'onnx_{key}'] = value
        
    except Exception as e:
        info.metadata['error'] = str(e)
    
    return info


def analyze_engine_model(model_path: str) -> ModelInfo:
    """
    分析 TensorRT Engine 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        模型信息
    """
    info = ModelInfo(
        model_path=model_path,
        model_type='engine',
        file_size_mb=os.path.getsize(model_path) / (1024 * 1024)
    )
    
    try:
        import tensorrt as trt
        
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine:
            # 獲取綁定信息
            num_bindings = engine.num_io_tensors
            info.metadata['num_bindings'] = num_bindings
            
            # 獲取輸入輸出形狀
            for i in range(num_bindings):
                name = engine.get_tensor_name(i)
                shape = tuple(engine.get_tensor_shape(name))
                dtype = engine.get_tensor_dtype(name)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                
                if is_input and info.input_shape is None:
                    info.input_shape = shape
                elif not is_input and info.output_shape is None:
                    info.output_shape = shape
                
                info.metadata[f'tensor_{i}'] = {
                    'name': name,
                    'shape': shape,
                    'dtype': str(dtype),
                    'is_input': is_input
                }
            
            # 獲取精度信息
            if engine.has_implicit_batch_dimension:
                info.metadata['has_implicit_batch'] = True
            
    except Exception as e:
        info.metadata['error'] = str(e)
    
    return info


def analyze_model(model_path: str) -> ModelInfo:
    """
    分析模型（自動檢測類型）
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        模型信息
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 根據擴展名判斷模型類型
    ext = Path(model_path).suffix.lower()
    
    if ext in ['.pt', '.pth']:
        return analyze_pt_model(model_path)
    elif ext == '.onnx':
        return analyze_onnx_model(model_path)
    elif ext == '.engine':
        return analyze_engine_model(model_path)
    else:
        raise ValueError(f"不支持的模型格式: {ext}")


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="分析 YOLO 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("model_path", help="模型文件路徑")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    
    args = parser.parse_args()
    
    try:
        info = analyze_model(args.model_path)
        
        if args.json:
            import json
            print(json.dumps({
                'model_path': info.model_path,
                'model_type': info.model_type,
                'file_size_mb': info.file_size_mb,
                'input_shape': info.input_shape,
                'output_shape': info.output_shape,
                'num_parameters': info.num_parameters,
                'num_classes': info.num_classes,
                'opset_version': info.opset_version,
                'precision': info.precision,
                'metadata': info.metadata
            }, indent=2, default=str))
        else:
            print(f"模型路徑: {info.model_path}")
            print(f"模型類型: {info.model_type}")
            print(f"文件大小: {info.file_size_mb:.2f} MB")
            if info.input_shape:
                print(f"輸入形狀: {info.input_shape}")
            if info.output_shape:
                print(f"輸出形狀: {info.output_shape}")
            if info.num_parameters:
                print(f"參數數量: {info.num_parameters:,}")
            if info.num_classes:
                print(f"類別數量: {info.num_classes}")
            if info.opset_version:
                print(f"ONNX Opset 版本: {info.opset_version}")
            if info.precision:
                print(f"精度: {info.precision}")
            
            if info.metadata:
                print(f"\n額外信息:")
                for key, value in info.metadata.items():
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"[錯誤] {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

