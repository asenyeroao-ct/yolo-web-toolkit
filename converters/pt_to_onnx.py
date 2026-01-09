"""
PyTorch YOLO 模型到 ONNX 轉換模塊

此模塊提供將 PyTorch YOLO 模型轉換為 ONNX 格式的功能。
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "需要安裝 PyTorch 和 Ultralytics YOLO。請確保已安裝:\n"
        "  - torch\n"
        "  - ultralytics\n"
        f"原始錯誤: {e}"
    )

# 全局鎖，確保 ONNX 導出不會併發執行
_onnx_export_lock = threading.Lock()


def convert_pt_to_onnx(
    pt_path: str,
    onnx_path: Optional[str] = None,
    imgsz: int = 640,
    simplify: bool = True,
    opset: int = 12,
    dynamic: bool = False,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    將 PyTorch YOLO 模型轉換為 ONNX 格式
    
    Args:
        pt_path: PyTorch 模型文件路徑 (.pt)
        onnx_path: 輸出的 ONNX 文件路徑。如果為 None，則自動生成
        imgsz: 輸入圖像大小（默認：640）
        simplify: 是否簡化 ONNX 模型（默認：True）
        opset: ONNX opset 版本（默認：12）
        dynamic: 是否使用動態輸入尺寸（默認：False）
        verbose: 是否輸出詳細信息（默認：False）
        
    Returns:
        (成功標誌, ONNX 文件路徑) 元組。如果轉換失敗，成功標誌為 False，路徑為 None
    """
    # 驗證 PyTorch 文件
    if not os.path.exists(pt_path):
        print(f"[錯誤] PyTorch 文件不存在: {pt_path}")
        return False, None
    
    if not pt_path.lower().endswith(('.pt', '.pth')):
        print(f"[錯誤] 文件不是 PyTorch 格式: {pt_path}")
        return False, None
    
    # 確定輸出路徑
    if onnx_path is None:
        pt_path_obj = Path(pt_path)
        onnx_path = str(pt_path_obj.parent / (pt_path_obj.stem + ".onnx"))
    
    try:
        if verbose:
            print(f"[轉換] 載入 PyTorch 模型: {pt_path}")
        
        # 載入 YOLO 模型
        model = YOLO(pt_path)
        
        if verbose:
            print(f"[轉換] 開始導出為 ONNX 格式...")
            print(f"[轉換] 輸入尺寸: {imgsz}x{imgsz}")
            print(f"[轉換] 輸出路徑: {onnx_path}")
        
        # 使用鎖確保 ONNX 導出序列化執行（避免 GLOBALS.in_onnx_export 衝突）
        with _onnx_export_lock:
            if verbose:
                print(f"[轉換] 獲取 ONNX 導出鎖，開始轉換...")
            
            # 導出為 ONNX
            success = model.export(
                format='onnx',
                imgsz=imgsz,
                simplify=simplify,
                opset=opset,
                dynamic=dynamic,
                verbose=verbose
            )
        
        if not success:
            print("[錯誤] ONNX 導出失敗")
            return False, None
        
        # Ultralytics 總是將檔案導出到模型所在的目錄
        # 我們需要找到實際生成的檔案並移動到目標位置
        pt_path_obj = Path(pt_path)
        auto_generated_path = pt_path_obj.parent / (pt_path_obj.stem + ".onnx")
        
        if verbose:
            print(f"[轉換] 檢查自動生成的檔案: {auto_generated_path}")
        
        if not os.path.exists(auto_generated_path):
            print(f"[錯誤] 無法找到導出的 ONNX 文件: {auto_generated_path}")
            return False, None
        
        # 如果目標路徑與自動生成的路徑不同，則移動檔案
        if str(auto_generated_path) != onnx_path:
            import shutil
            if verbose:
                print(f"[轉換] 移動檔案從 {auto_generated_path} 到 {onnx_path}")
            
            # 確保目標目錄存在
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            
            # 如果目標檔案已存在，先刪除
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            shutil.move(str(auto_generated_path), onnx_path)
        
        if verbose:
            print(f"[成功] ONNX 文件已保存到: {onnx_path}")
        
        # 將類別名稱嵌入到 ONNX 模型元數據中
        try:
            import onnx
            
            if hasattr(model, 'names') and model.names:
                class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
                if class_names:
                    # 載入 ONNX 模型
                    onnx_model = onnx.load(onnx_path)
                    
                    # 添加類別信息到模型元數據
                    # 方法1: 使用 metadata_props (推薦)
                    meta = onnx_model.metadata_props.add()
                    meta.key = "classes"
                    meta.value = ",".join(class_names)
                    
                    # 方法2: 添加單獨的類別數量
                    meta_count = onnx_model.metadata_props.add()
                    meta_count.key = "num_classes"
                    meta_count.value = str(len(class_names))
                    
                    # 方法3: 為每個類別添加索引映射
                    for idx, name in enumerate(class_names):
                        meta_class = onnx_model.metadata_props.add()
                        meta_class.key = f"class_{idx}"
                        meta_class.value = name
                    
                    # 保存更新後的 ONNX 模型
                    onnx.save(onnx_model, onnx_path)
                    
                    if verbose:
                        print(f"[成功] 類別信息已嵌入 ONNX 模型")
                        print(f"[信息] 共 {len(class_names)} 個類別")
                        print(f"[信息] 類別列表: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        except ImportError:
            if verbose:
                print(f"[警告] 未安裝 onnx 包，無法嵌入類別信息")
        except Exception as e:
            if verbose:
                print(f"[警告] 無法嵌入類別信息到 ONNX 模型: {e}")
                import traceback
                traceback.print_exc()
        
        return True, onnx_path
        
    except Exception as e:
        print(f"[錯誤] 轉換過程中發生異常: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, None


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="將 PyTorch YOLO 模型轉換為 ONNX 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本轉換
  python pt_to_onnx.py model.pt
  
  # 指定輸出路徑和尺寸
  python pt_to_onnx.py model.pt -o model.onnx --imgsz 640
  
  # 使用動態輸入
  python pt_to_onnx.py model.pt --dynamic
        """
    )
    
    parser.add_argument("pt_path", help="PyTorch 模型文件路徑 (.pt)")
    parser.add_argument("-o", "--output", help="輸出的 ONNX 文件路徑（默認：與 PT 文件同目錄，擴展名改為 .onnx）")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入圖像大小（默認：640）")
    parser.add_argument("--no-simplify", action="store_true", help="不簡化 ONNX 模型")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本（默認：12）")
    parser.add_argument("--dynamic", action="store_true", help="使用動態輸入尺寸")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細信息")
    
    args = parser.parse_args()
    
    print(f"[轉換] 開始轉換: {args.pt_path}")
    success, onnx_path = convert_pt_to_onnx(
        args.pt_path,
        args.output,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
        opset=args.opset,
        dynamic=args.dynamic,
        verbose=args.verbose
    )
    
    if success:
        print(f"[成功] ONNX 文件已保存到: {onnx_path}")
        sys.exit(0)
    else:
        print("[失敗] 轉換失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

