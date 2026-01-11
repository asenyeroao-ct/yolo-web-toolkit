"""
ONNX 到 TensorRT Engine 轉換模塊

此模塊提供將 ONNX 模型轉換為 TensorRT engine 的功能，可用於其他軟件調用。

使用範例:
    from onnx_to_tensorrt import convert_onnx_to_engine, ConversionConfig
    
    # 基本使用
    result = convert_onnx_to_engine("model.onnx", "model.engine")
    
    # 使用配置
    config = ConversionConfig(
        enable_fp16=True,
        enable_fp8=False,
        fixed_input_size=True,
        detection_resolution=640,
        workspace_size_gb=1
    )
    result = convert_onnx_to_engine("model.onnx", "model.engine", config=config)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # 不使用 pycuda.autoinit，改為按需初始化以加快導入速度
    # import pycuda.autoinit
except ImportError as e:
    raise ImportError(
        "需要安裝 TensorRT 和 PyCUDA。請確保已安裝:\n"
        "  - TensorRT Python 綁定\n"
        "  - PyCUDA\n"
        f"原始錯誤: {e}"
    )


# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# CUDA 初始化標誌
_cuda_initialized = False
_cuda_init_lock = None


def _ensure_cuda_initialized():
    """確保 CUDA 已初始化（按需初始化）"""
    global _cuda_initialized, _cuda_init_lock
    
    if _cuda_initialized:
        return
    
    # 延遲導入 threading 以避免循環導入
    if _cuda_init_lock is None:
        import threading
        _cuda_init_lock = threading.Lock()
    
    with _cuda_init_lock:
        if not _cuda_initialized:
            try:
                cuda.init()
                # 創建 CUDA 上下文
                device = cuda.Device(0)
                ctx = device.make_context()
                import atexit
                atexit.register(ctx.pop)
                _cuda_initialized = True
            except Exception as e:
                print(f"[警告] CUDA 初始化失敗: {e}")
                raise


@dataclass
class ConversionConfig:
    """轉換配置類"""
    enable_fp16: bool = False
    """啟用 FP16 精度"""
    
    enable_fp8: bool = False
    """啟用 FP8 精度"""
    
    fixed_input_size: bool = False
    """是否使用固定輸入尺寸"""
    
    detection_resolution: int = 640
    """檢測分辨率（當 fixed_input_size=True 時使用）"""
    
    workspace_size_gb: int = 1
    """工作空間大小（GB）"""
    
    min_input_size: Tuple[int, int] = (160, 160)
    """最小輸入尺寸 (H, W) - 用於動態輸入"""
    
    opt_input_size: Tuple[int, int] = (320, 320)
    """最優輸入尺寸 (H, W) - 用於動態輸入"""
    
    max_input_size: Tuple[int, int] = (640, 640)
    """最大輸入尺寸 (H, W) - 用於動態輸入"""
    
    verbose: bool = False
    """是否輸出詳細信息"""
    
    save_classes_file: bool = True
    """是否保存類別信息文件"""


class ProgressMonitor:
    """進度監控器（用於 TensorRT 構建進度回調）"""
    
    def __init__(self, callback: Optional[Callable[[int, int], None]] = None):
        """
        初始化進度監控器
        
        Args:
            callback: 進度回調函數，簽名為 (current: int, total: int) -> None
        """
        self.callback = callback
        self.current = 0
        self.total = 0
    
    def __call__(self, phase: int, completed: int, total: int):
        """TensorRT 進度回調接口"""
        self.current = completed
        self.total = total
        if self.callback:
            self.callback(completed, total)


def load_onnx_model(onnx_path: str) -> Optional[bytes]:
    """
    載入 ONNX 模型文件
    
    Args:
        onnx_path: ONNX 模型文件路徑
        
    Returns:
        模型字節數據，如果載入失敗則返回 None
    """
    if not os.path.exists(onnx_path):
        print(f"[錯誤] ONNX 文件不存在: {onnx_path}")
        return None
    
    try:
        with open(onnx_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[錯誤] 無法讀取 ONNX 文件: {e}")
        return None


def get_model_input_shape(onnx_path: str, verbose: bool = False) -> Optional[Tuple[int, ...]]:
    """
    從 ONNX 模型獲取輸入形狀
    
    Args:
        onnx_path: ONNX 模型文件路徑
        verbose: 是否輸出詳細信息
        
    Returns:
        輸入形狀元組，如果獲取失敗則返回 None
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        if len(model.graph.input) == 0:
            return None
        
        input_tensor = model.graph.input[0]
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(int(dim.dim_value))
            else:
                shape.append(-1)  # 動態維度
        
        return tuple(shape)
    except Exception as e:
        if verbose:
            print(f"[警告] 無法獲取模型輸入形狀: {e}")
        return None


def load_model_class_names(model_path: str) -> List[str]:
    """
    載入模型類別名稱
    
    嘗試從以下位置載入:
    1. ONNX 模型內部的元數據 (metadata_props)
    2. 模型同目錄的 .names 或 .txt 文件
    3. 模型同目錄的 _classes.txt 文件
    4. 模型同目錄的 classes.txt 文件
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        類別名稱列表
    """
    # 首先嘗試從 ONNX 模型元數據中讀取
    if model_path.lower().endswith('.onnx'):
        try:
            import onnx
            onnx_model = onnx.load(model_path)
            
            # 方法1: 從 "classes" 鍵讀取逗號分隔的類別列表
            for meta in onnx_model.metadata_props:
                if meta.key == "classes":
                    class_names = [name.strip() for name in meta.value.split(',') if name.strip()]
                    if class_names:
                        return class_names
            
            # 方法2: 從 "class_0", "class_1", ... 鍵讀取
            class_dict = {}
            for meta in onnx_model.metadata_props:
                if meta.key.startswith("class_"):
                    try:
                        idx = int(meta.key.split("_")[1])
                        class_dict[idx] = meta.value
                    except (ValueError, IndexError):
                        continue
            
            if class_dict:
                # 按索引排序並返回
                max_idx = max(class_dict.keys())
                class_names = [class_dict.get(i, f"class_{i}") for i in range(max_idx + 1)]
                return class_names
                
        except ImportError:
            pass  # onnx 未安裝，繼續嘗試其他方法
        except Exception as e:
            if ConversionConfig().verbose:
                print(f"[警告] 無法從 ONNX 模型讀取類別信息: {e}")
    
    # 如果從模型中讀取失敗，嘗試從外部文件讀取
    model_path_obj = Path(model_path)
    candidates = [
        model_path_obj.parent / (model_path_obj.stem + ".names"),
        model_path_obj.parent / (model_path_obj.stem + ".txt"),
        model_path_obj.parent / (model_path_obj.stem + "_classes.txt"),
        model_path_obj.parent / (model_path_obj.stem + "_classes.names"),
        model_path_obj.parent / "classes.txt",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    names = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            names.append(line)
                    if names:
                        return names
            except Exception as e:
                if ConversionConfig().verbose:
                    print(f"[警告] 無法讀取類別文件 {candidate}: {e}")
    
    return []


def build_engine_from_onnx(
    onnx_path: str,
    config: ConversionConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Optional[trt.ICudaEngine]:
    """
    從 ONNX 文件構建 TensorRT engine
    
    Args:
        onnx_path: ONNX 模型文件路徑
        config: 轉換配置
        progress_callback: 進度回調函數，簽名為 (current: int, total: int) -> None
        
    Returns:
        構建好的 TensorRT engine，如果構建失敗則返回 None
    """
    if not os.path.exists(onnx_path):
        print(f"[錯誤] ONNX 文件不存在: {onnx_path}")
        return None
    
    if config.verbose:
        print(f"[TensorRT] 開始構建 engine，來源: {onnx_path}")
    
    # 檢查 TensorRT 版本
    try:
        trt_version = trt.__version__
        if config.verbose:
            print(f"[TensorRT] TensorRT 版本: {trt_version}")
    except:
        pass
    
    # 創建 builder 和 network
    try:
        builder = trt.Builder(TRT_LOGGER)
        if builder is None:
            print("[錯誤] 無法創建 TensorRT Builder。請檢查 TensorRT 和 CUDA 安裝。")
            return None
    except Exception as e:
        print(f"[錯誤] 創建 TensorRT Builder 失敗: {e}")
        print("[提示] 請確保已正確安裝 TensorRT 和 CUDA，並且 GPU 可用。")
        return None
    
    try:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        if network is None:
            print("[錯誤] 無法創建 TensorRT Network")
            return None
    except Exception as e:
        print(f"[錯誤] 創建 TensorRT Network 失敗: {e}")
        return None
    
    try:
        parser = trt.OnnxParser(network, TRT_LOGGER)
        if parser is None:
            print("[錯誤] 無法創建 ONNX Parser")
            return None
    except Exception as e:
        print(f"[錯誤] 創建 ONNX Parser 失敗: {e}")
        return None
    
    try:
        builder_config = builder.create_builder_config()
        if builder_config is None:
            print("[錯誤] 無法創建 TensorRT Builder Config。這可能是由於 GPU/CUDA 問題或 TensorRT 版本不兼容。")
            print("[提示] 請檢查：")
            print("  1. GPU 是否可用且驅動程序已正確安裝")
            print("  2. CUDA 是否正確安裝並與 TensorRT 版本兼容")
            print("  3. TensorRT 版本是否與您的系統兼容")
            return None
    except Exception as e:
        print(f"[錯誤] 創建 TensorRT Builder Config 失敗: {e}")
        print("[提示] 這通常是因為 TensorRT 初始化失敗。請檢查 GPU/CUDA 環境。")
        import traceback
        traceback.print_exc()
        return None
    
    # 設置進度監控器
    if progress_callback:
        progress_monitor = ProgressMonitor(progress_callback)
        builder_config.progress_monitor = progress_monitor
    
    # 解析 ONNX 模型
    if config.verbose:
        print("[TensorRT] 正在解析 ONNX 文件...")
    
    onnx_data = load_onnx_model(onnx_path)
    if not onnx_data:
        return None
    
    if not parser.parse(onnx_data):
        print("[錯誤] 無法解析 ONNX 文件")
        for i in range(parser.num_errors):
            print(f"  錯誤 {i}: {parser.get_error(i)}")
        return None
    
    if config.verbose:
        print("[TensorRT] ONNX 文件解析完成")
    
    # 獲取輸入張量
    if network.num_inputs == 0:
        print("[錯誤] 網絡沒有輸入張量")
        return None
    
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = list(input_tensor.shape)
    
    if config.verbose:
        print(f"[TensorRT] 輸入張量: {input_name}, 形狀: {input_shape}")
    
    # 獲取模型的實際輸入形狀
    model_shape = get_model_input_shape(onnx_path, config.verbose)
    
    # 確定是否為靜態輸入
    # 檢查網絡輸入形狀是否為靜態（所有維度都 > 0）
    is_static = all(d > 0 for d in input_shape) if len(input_shape) >= 4 else False
    
    # 如果模型有靜態輸入尺寸，優先使用模型的實際尺寸
    if model_shape and len(model_shape) >= 4:
        model_h = int(model_shape[2]) if model_shape[2] > 0 else None
        model_w = int(model_shape[3]) if model_shape[3] > 0 else None
        
        if model_h and model_w:
            is_static = True
            if config.verbose:
                print(f"[TensorRT] 檢測到模型實際輸入尺寸: {model_h}x{model_w}")
    
    # 設置優化配置文件
    try:
        profile = builder.create_optimization_profile()
        if profile is None:
            print("[錯誤] 無法創建優化配置文件")
            return None
    except Exception as e:
        print(f"[錯誤] 創建優化配置文件失敗: {e}")
        return None
    
    if is_static:
        # 靜態輸入 - 優先使用模型的實際尺寸
        if model_shape and len(model_shape) >= 4:
            model_h = int(model_shape[2]) if model_shape[2] > 0 else None
            model_w = int(model_shape[3]) if model_shape[3] > 0 else None
            
            if model_h and model_w:
                # 使用模型的實際尺寸
                h = model_h
                w = model_w
                if config.verbose and (h != config.detection_resolution or w != config.detection_resolution):
                    print(f"[TensorRT] 警告: 模型實際輸入尺寸為 {h}x{w}，將使用模型尺寸而非配置的 {config.detection_resolution}x{config.detection_resolution}")
            else:
                # 從網絡輸入形狀獲取
                if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
                    h = int(input_shape[2])
                    w = int(input_shape[3])
                else:
                    h = w = config.detection_resolution
        else:
            # 從網絡輸入形狀獲取
            if len(input_shape) >= 4 and input_shape[2] > 0 and input_shape[3] > 0:
                h = int(input_shape[2])
                w = int(input_shape[3])
            else:
                h = w = config.detection_resolution
        
        static_shape = [1, 3, h, w]
        profile.set_shape(input_name, static_shape, static_shape, static_shape)
        if config.verbose:
            print(f"[TensorRT] 使用靜態輸入尺寸: {h}x{w}")
    else:
        # 動態輸入
        min_h, min_w = config.min_input_size
        opt_h, opt_w = config.opt_input_size
        max_h, max_w = config.max_input_size
        
        min_shape = [1, 3, min_h, min_w]
        opt_shape = [1, 3, opt_h, opt_w]
        max_shape = [1, 3, max_h, max_w]
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        if config.verbose:
            print(f"[TensorRT] 使用動態輸入尺寸: MIN={min_h}x{min_w}, OPT={opt_h}x{opt_w}, MAX={max_h}x{max_w}")
    
    builder_config.add_optimization_profile(profile)
    
    # 設置精度標誌
    if config.enable_fp16:
        if builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            if config.verbose:
                print("[TensorRT] 啟用 FP16 精度")
        else:
            print("[警告] 當前平台不支持 FP16，將使用 FP32")
    
    if config.enable_fp8:
        if builder.platform_has_fast_fp8:
            builder_config.set_flag(trt.BuilderFlag.FP8)
            if config.verbose:
                print("[TensorRT] 啟用 FP8 精度")
        else:
            print("[警告] 當前平台不支持 FP8，將使用 FP32/FP16")
    
    # 設置工作空間大小
    workspace_bytes = config.workspace_size_gb * (1 << 30)  # GB 轉換為字節
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    
    if config.verbose:
        print(f"[TensorRT] 工作空間大小: {config.workspace_size_gb} GB")
        print("[TensorRT] 開始構建 engine（這可能需要幾分鐘）...")
    
    # 構建 engine
    try:
        # 嘗試使用 build_serialized_network (TensorRT 8.0+)
        plan = None
        try:
            plan = builder.build_serialized_network(network, builder_config)
        except AttributeError:
            # 如果不支持 build_serialized_network，使用舊方法
            if config.verbose:
                print("[TensorRT] 使用舊版 API 構建 engine...")
            engine = builder.build_engine(network, builder_config)
            if engine is None:
                print("[錯誤] 無法構建 engine")
                return None
            if config.verbose:
                print("[TensorRT] Engine 構建完成")
            return engine
        
        if plan is None:
            print("[錯誤] 無法構建 engine (build_serialized_network 返回 None)")
            print("[提示] 這可能是由於：")
            print("  1. 模型不兼容或損壞")
            print("  2. 輸入尺寸配置錯誤")
            print("  3. GPU 記憶體不足")
            print("  4. TensorRT 版本問題")
            return None
        
        runtime = trt.Runtime(TRT_LOGGER)
        if runtime is None:
            print("[錯誤] 無法創建 TensorRT Runtime")
            return None
        
        engine = runtime.deserialize_cuda_engine(plan)
        
        if engine is None:
            print("[錯誤] 無法反序列化 engine")
            return None
        
        if config.verbose:
            print("[TensorRT] Engine 構建完成")
        
        return engine
    
    except RuntimeError as e:
        error_msg = str(e)
        if "pybind11::init(): factory function returned nullptr" in error_msg:
            print("[錯誤] TensorRT 初始化失敗：factory function returned nullptr")
            print("[可能的原因]：")
            print("  1. TensorRT 與 CUDA 版本不兼容")
            print("  2. GPU 驅動程序過舊或未正確安裝")
            print("  3. TensorRT 庫文件損壞或缺失")
            print("  4. GPU 不可用或記憶體不足")
            print("[建議]：")
            print("  - 檢查 TensorRT 和 CUDA 版本兼容性")
            print("  - 確認 GPU 驅動程序已正確安裝")
            print("  - 嘗試重啟應用或檢查 GPU 狀態")
        else:
            print(f"[錯誤] 構建 engine 時發生 RuntimeError: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[錯誤] 構建 engine 時發生異常: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_engine(engine: trt.ICudaEngine, engine_path: str) -> bool:
    """
    保存 TensorRT engine 到文件
    
    Args:
        engine: TensorRT engine 對象
        engine_path: 保存路徑
        
    Returns:
        是否保存成功
    """
    try:
        serialized_engine = engine.serialize()
        if serialized_engine is None:
            print("[錯誤] 無法序列化 engine")
            return False
        
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        return True
    except Exception as e:
        print(f"[錯誤] 保存 engine 時發生異常: {e}")
        return False


def save_classes_file(model_path: str, class_names: List[str], output_path: Optional[str] = None) -> bool:
    """
    保存類別信息到文件
    
    Args:
        model_path: 模型文件路徑（用於確定輸出路徑）
        class_names: 類別名稱列表
        output_path: 可選的輸出路徑，如果為 None 則自動生成
        
    Returns:
        是否保存成功
    """
    if not class_names:
        return False
    
    if output_path is None:
        model_path_obj = Path(model_path)
        output_path = str(model_path_obj.parent / (model_path_obj.stem + "_classes.txt"))
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Classes Information\n")
            f.write("# Generated during TensorRT engine export\n")
            f.write(f"# Total classes: {len(class_names)}\n\n")
            
            for name in class_names:
                f.write(f"{name}\n")
        
        return True
    except Exception as e:
        print(f"[警告] 無法保存類別文件: {e}")
        return False


def convert_onnx_to_engine(
    onnx_path: str,
    engine_path: Optional[str] = None,
    config: Optional[ConversionConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[bool, Optional[str]]:
    """
    將 ONNX 模型轉換為 TensorRT engine
    
    這是主要的 API 函數，用於將 ONNX 模型轉換為 TensorRT engine。
    
    Args:
        onnx_path: ONNX 模型文件路徑
        engine_path: 輸出的 engine 文件路徑。如果為 None，則自動生成（與 ONNX 文件同目錄，擴展名改為 .engine）
        config: 轉換配置。如果為 None，則使用默認配置
        progress_callback: 可選的進度回調函數，簽名為 (current: int, total: int) -> None
        
    Returns:
        (成功標誌, engine 文件路徑) 元組。如果轉換失敗，成功標誌為 False，路徑為 None
        
    範例:
        >>> # 基本使用
        >>> success, engine_path = convert_onnx_to_engine("model.onnx")
        >>> if success:
        ...     print(f"轉換成功: {engine_path}")
        
        >>> # 使用自定義配置
        >>> config = ConversionConfig(enable_fp16=True, detection_resolution=640)
        >>> success, engine_path = convert_onnx_to_engine("model.onnx", config=config)
        
        >>> # 帶進度回調
        >>> def progress(current, total):
        ...     print(f"進度: {current}/{total}")
        >>> success, engine_path = convert_onnx_to_engine("model.onnx", progress_callback=progress)
    """
    # 確保 CUDA 已初始化（按需初始化）
    _ensure_cuda_initialized()
    
    if config is None:
        config = ConversionConfig()
    
    # 驗證 ONNX 文件
    if not os.path.exists(onnx_path):
        print(f"[錯誤] ONNX 文件不存在: {onnx_path}")
        return False, None
    
    if not onnx_path.lower().endswith(".onnx"):
        print(f"[錯誤] 文件不是 ONNX 格式: {onnx_path}")
        return False, None
    
    # 確定輸出路徑
    if engine_path is None:
        onnx_path_obj = Path(onnx_path)
        engine_path = str(onnx_path_obj.parent / (onnx_path_obj.stem + ".engine"))
    
    # 構建 engine
    engine = build_engine_from_onnx(onnx_path, config, progress_callback)
    if engine is None:
        return False, None
    
    # 保存 engine
    if not save_engine(engine, engine_path):
        return False, None
    
    print(f"[TensorRT] Engine 已保存到: {engine_path}")
    
    # 保存類別信息（如果啟用）
    if config.save_classes_file:
        class_names = load_model_class_names(onnx_path)
        if class_names:
            save_classes_file(onnx_path, class_names)
            if config.verbose:
                print(f"[TensorRT] 類別信息已保存（共 {len(class_names)} 個類別）")
    
    return True, engine_path


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="將 ONNX 模型轉換為 TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本轉換
  python onnx_to_tensorrt.py model.onnx
  
  # 指定輸出路徑
  python onnx_to_tensorrt.py model.onnx -o model.engine
  
  # 啟用 FP16
  python onnx_to_tensorrt.py model.onnx --fp16
  
  # 自定義配置
  python onnx_to_tensorrt.py model.onnx --fp16 --resolution 640 --workspace 2
        """
    )
    
    parser.add_argument("onnx_path", help="ONNX 模型文件路徑")
    parser.add_argument("-o", "--output", help="輸出的 engine 文件路徑（默認：與 ONNX 文件同目錄，擴展名改為 .engine）")
    parser.add_argument("--fp16", action="store_true", help="啟用 FP16 精度")
    parser.add_argument("--fp8", action="store_true", help="啟用 FP8 精度")
    parser.add_argument("--fixed-size", action="store_true", help="使用固定輸入尺寸")
    parser.add_argument("--resolution", type=int, default=640, help="檢測分辨率（默認：640）")
    parser.add_argument("--workspace", type=int, default=1, help="工作空間大小（GB，默認：1）")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細信息")
    parser.add_argument("--no-classes", action="store_true", help="不保存類別信息文件")
    
    args = parser.parse_args()
    
    config = ConversionConfig(
        enable_fp16=args.fp16,
        enable_fp8=args.fp8,
        fixed_input_size=args.fixed_size,
        detection_resolution=args.resolution,
        workspace_size_gb=args.workspace,
        verbose=args.verbose,
        save_classes_file=not args.no_classes
    )
    
    def progress_callback(current, total):
        if total > 0:
            percent = (current / total) * 100
            print(f"\r[進度] {current}/{total} ({percent:.1f}%)", end="", flush=True)
    
    print(f"[TensorRT] 開始轉換: {args.onnx_path}")
    success, engine_path = convert_onnx_to_engine(
        args.onnx_path,
        args.output,
        config,
        progress_callback if args.verbose else None
    )
    
    if success:
        print(f"\n[成功] Engine 已保存到: {engine_path}")
        sys.exit(0)
    else:
        print("\n[失敗] 轉換失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

