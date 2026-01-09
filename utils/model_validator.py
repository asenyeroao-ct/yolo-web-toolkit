"""
模型驗證模塊

提供模型格式驗證、結構檢查和推理測試功能
參考 hank-ai/darknet 的模型驗證方法
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ModelValidationResult:
    """模型驗證結果"""
    is_valid: bool
    """模型是否有效"""
    
    model_type: str
    """模型類型 (pt/onnx/engine)"""
    
    input_shape: Optional[Tuple[int, ...]] = None
    """輸入形狀"""
    
    output_shape: Optional[Tuple[int, ...]] = None
    """輸出形狀"""
    
    num_classes: Optional[int] = None
    """類別數量"""
    
    errors: List[str] = None
    """錯誤列表"""
    
    warnings: List[str] = None
    """警告列表"""
    
    info: Dict = None
    """額外信息"""
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.info is None:
            self.info = {}


def validate_pt_model(model_path: str) -> ModelValidationResult:
    """
    驗證 PyTorch 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        驗證結果
    """
    result = ModelValidationResult(
        is_valid=False,
        model_type='pt'
    )
    
    try:
        import torch
        from ultralytics import YOLO
        
        # 檢查文件是否存在
        if not os.path.exists(model_path):
            result.errors.append(f"模型文件不存在: {model_path}")
            return result
        
        # 嘗試載入模型
        try:
            model = YOLO(model_path)
            result.is_valid = True
            result.info['model_name'] = model_path
            
            # 獲取模型信息
            if hasattr(model, 'model'):
                # 嘗試獲取輸入輸出形狀
                try:
                    # 創建一個測試輸入
                    test_input = torch.zeros(1, 3, 640, 640)
                    with torch.no_grad():
                        output = model.model(test_input)
                        if isinstance(output, (list, tuple)):
                            result.output_shape = tuple(output[0].shape) if len(output) > 0 else None
                        else:
                            result.output_shape = tuple(output.shape)
                    result.input_shape = (1, 3, 640, 640)
                except Exception as e:
                    result.warnings.append(f"無法獲取模型形狀: {e}")
            
            # 獲取類別數量
            if hasattr(model, 'names'):
                result.num_classes = len(model.names)
                result.info['class_names'] = list(model.names.values())
            
        except Exception as e:
            result.errors.append(f"無法載入模型: {e}")
            
    except ImportError:
        result.errors.append("需要安裝 torch 和 ultralytics")
    
    return result


def validate_onnx_model(model_path: str) -> ModelValidationResult:
    """
    驗證 ONNX 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        驗證結果
    """
    result = ModelValidationResult(
        is_valid=False,
        model_type='onnx'
    )
    
    try:
        import onnx
        import onnxruntime as ort
        
        # 檢查文件是否存在
        if not os.path.exists(model_path):
            result.errors.append(f"模型文件不存在: {model_path}")
            return result
        
        # 驗證 ONNX 模型結構
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            result.is_valid = True
            result.info['onnx_version'] = model.opset_import[0].version if model.opset_import else None
            
            # 獲取輸入輸出信息
            if len(model.graph.input) > 0:
                input_tensor = model.graph.input[0]
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(int(dim.dim_value))
                    else:
                        shape.append(-1)
                result.input_shape = tuple(shape)
            
            if len(model.graph.output) > 0:
                output_tensor = model.graph.output[0]
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(int(dim.dim_value))
                    else:
                        shape.append(-1)
                result.output_shape = tuple(shape)
            
            # 嘗試創建推理會話
            try:
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                result.info['providers'] = session.get_providers()
                
                # 獲取輸入輸出名稱
                input_names = [inp.name for inp in session.get_inputs()]
                output_names = [out.name for out in session.get_outputs()]
                result.info['input_names'] = input_names
                result.info['output_names'] = output_names
                
            except Exception as e:
                result.warnings.append(f"無法創建推理會話: {e}")
                
        except onnx.checker.ValidationError as e:
            result.errors.append(f"ONNX 模型驗證失敗: {e}")
        except Exception as e:
            result.errors.append(f"無法載入 ONNX 模型: {e}")
            
    except ImportError:
        result.errors.append("需要安裝 onnx 和 onnxruntime")
    
    return result


def validate_engine_model(model_path: str) -> ModelValidationResult:
    """
    驗證 TensorRT Engine 模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        驗證結果
    """
    result = ModelValidationResult(
        is_valid=False,
        model_type='engine'
    )
    
    try:
        import tensorrt as trt
        
        # 檢查文件是否存在
        if not os.path.exists(model_path):
            result.errors.append(f"模型文件不存在: {model_path}")
            return result
        
        # 嘗試載入 engine
        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            with open(model_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                result.errors.append("無法反序列化 TensorRT engine")
                return result
            
            result.is_valid = True
            
            # 獲取綁定信息
            num_bindings = engine.num_io_tensors
            result.info['num_bindings'] = num_bindings
            
            # 獲取輸入輸出信息
            input_shapes = []
            output_shapes = []
            
            for i in range(num_bindings):
                name = engine.get_tensor_name(i)
                shape = tuple(engine.get_tensor_shape(name))
                dtype = engine.get_tensor_dtype(name)
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                
                if is_input:
                    input_shapes.append(shape)
                    result.info[f'input_{i}'] = {'name': name, 'shape': shape, 'dtype': str(dtype)}
                else:
                    output_shapes.append(shape)
                    result.info[f'output_{i}'] = {'name': name, 'shape': shape, 'dtype': str(dtype)}
            
            if input_shapes:
                result.input_shape = input_shapes[0]
            if output_shapes:
                result.output_shape = output_shapes[0]
                
        except Exception as e:
            result.errors.append(f"無法載入 TensorRT engine: {e}")
            
    except ImportError:
        result.errors.append("需要安裝 tensorrt")
    except Exception as e:
        result.errors.append(f"TensorRT 相關錯誤: {e}")
    
    return result


def validate_model(model_path: str) -> ModelValidationResult:
    """
    驗證模型（自動檢測類型）
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        驗證結果
    """
    if not os.path.exists(model_path):
        result = ModelValidationResult(
            is_valid=False,
            model_type='unknown'
        )
        result.errors.append(f"文件不存在: {model_path}")
        return result
    
    # 根據擴展名判斷模型類型
    ext = Path(model_path).suffix.lower()
    
    if ext in ['.pt', '.pth']:
        return validate_pt_model(model_path)
    elif ext == '.onnx':
        return validate_onnx_model(model_path)
    elif ext == '.engine':
        return validate_engine_model(model_path)
    else:
        result = ModelValidationResult(
            is_valid=False,
            model_type='unknown'
        )
        result.errors.append(f"不支持的模型格式: {ext}")
        return result


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="驗證 YOLO 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("model_path", help="模型文件路徑")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細信息")
    
    args = parser.parse_args()
    
    print(f"[驗證] 開始驗證模型: {args.model_path}")
    result = validate_model(args.model_path)
    
    if result.is_valid:
        print(f"[成功] 模型驗證通過")
        print(f"  類型: {result.model_type}")
        if result.input_shape:
            print(f"  輸入形狀: {result.input_shape}")
        if result.output_shape:
            print(f"  輸出形狀: {result.output_shape}")
        if result.num_classes:
            print(f"  類別數量: {result.num_classes}")
    else:
        print(f"[失敗] 模型驗證失敗")
        for error in result.errors:
            print(f"  錯誤: {error}")
    
    if result.warnings:
        print(f"[警告]")
        for warning in result.warnings:
            print(f"  {warning}")
    
    if args.verbose and result.info:
        print(f"[詳細信息]")
        for key, value in result.info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

