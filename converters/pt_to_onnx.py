"""
PyTorch YOLO 模型到 ONNX 轉換模塊

此模塊提供將 PyTorch YOLO 模型轉換為 ONNX 格式的功能。
"""

import os
import sys
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
        
        # 檢查輸出文件是否存在
        if not os.path.exists(onnx_path):
            # 嘗試查找自動生成的文件名
            pt_path_obj = Path(pt_path)
            possible_paths = [
                str(pt_path_obj.parent / (pt_path_obj.stem + ".onnx")),
                str(pt_path_obj.parent / (pt_path_obj.stem + "_" + str(imgsz) + ".onnx")),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    onnx_path = path
                    break
            else:
                print("[錯誤] 無法找到導出的 ONNX 文件")
                return False, None
        
        if verbose:
            print(f"[成功] ONNX 文件已保存到: {onnx_path}")
        
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

