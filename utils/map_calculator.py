"""
mAP (Mean Average Precision) 計算模塊

提供模型評估和 mAP 計算功能
參考 hank-ai/darknet 的 mAP 計算方法
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MapResult:
    """mAP 計算結果"""
    map_50: float
    """mAP@0.5"""
    
    map_50_95: float
    """mAP@0.5:0.95"""
    
    precision: float
    """精確度"""
    
    recall: float
    """召回率"""
    
    num_classes: int
    """類別數量"""
    
    per_class_map: Dict[str, float] = None
    """每個類別的 mAP"""
    
    metrics: Dict = None
    """其他評估指標"""
    
    def __post_init__(self):
        if self.per_class_map is None:
            self.per_class_map = {}
        if self.metrics is None:
            self.metrics = {}


def calculate_map_with_ultralytics(
    model_path: str,
    data_yaml: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    verbose: bool = True
) -> Optional[MapResult]:
    """
    使用 Ultralytics YOLO 計算 mAP
    
    Args:
        model_path: 模型文件路徑
        data_yaml: 數據配置文件路徑 (YOLO 格式)
        conf_threshold: 置信度閾值
        iou_threshold: IoU 閾值
        verbose: 是否輸出詳細信息
        
    Returns:
        mAP 結果，如果計算失敗則返回 None
    """
    try:
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            print(f"[錯誤] 模型文件不存在: {model_path}")
            return None
        
        if not os.path.exists(data_yaml):
            print(f"[錯誤] 數據配置文件不存在: {data_yaml}")
            return None
        
        # 載入模型
        model = YOLO(model_path)
        
        # 執行驗證
        if verbose:
            print(f"[mAP] 開始計算 mAP...")
            print(f"[mAP] 模型: {model_path}")
            print(f"[mAP] 數據配置: {data_yaml}")
        
        results = model.val(
            data=data_yaml,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=verbose
        )
        
        # 提取結果
        if hasattr(results, 'box'):
            box_metrics = results.box
            
            # 獲取類別數量
            num_classes = len(box_metrics.maps) if hasattr(box_metrics, 'maps') else 0
            
            # 獲取 mAP 值
            map_50 = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
            map_50_95 = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
            
            # 獲取精確度和召回率
            precision = float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0
            recall = float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0
            
            # 獲取每個類別的 mAP
            per_class_map = {}
            if hasattr(box_metrics, 'maps') and hasattr(box_metrics, 'names'):
                for i, (class_id, map_value) in enumerate(box_metrics.maps.items()):
                    class_name = box_metrics.names.get(class_id, f'class_{class_id}')
                    per_class_map[class_name] = float(map_value)
            
            # 其他指標
            metrics = {
                'f1_score': float(box_metrics.f1) if hasattr(box_metrics, 'f1') else None,
                'precision_per_class': {},
                'recall_per_class': {}
            }
            
            if hasattr(box_metrics, 'p') and hasattr(box_metrics, 'names'):
                for i, (class_id, p_value) in enumerate(box_metrics.p.items()):
                    class_name = box_metrics.names.get(class_id, f'class_{class_id}')
                    metrics['precision_per_class'][class_name] = float(p_value)
            
            if hasattr(box_metrics, 'r') and hasattr(box_metrics, 'names'):
                for i, (class_id, r_value) in enumerate(box_metrics.r.items()):
                    class_name = box_metrics.names.get(class_id, f'class_{class_id}')
                    metrics['recall_per_class'][class_name] = float(r_value)
            
            result = MapResult(
                map_50=map_50,
                map_50_95=map_50_95,
                precision=precision,
                recall=recall,
                num_classes=num_classes,
                per_class_map=per_class_map,
                metrics=metrics
            )
            
            if verbose:
                print(f"[mAP] 計算完成:")
                print(f"  mAP@0.5: {map_50:.4f}")
                print(f"  mAP@0.5:0.95: {map_50_95:.4f}")
                print(f"  精確度: {precision:.4f}")
                print(f"  召回率: {recall:.4f}")
                print(f"  類別數量: {num_classes}")
            
            return result
        else:
            print("[錯誤] 無法從驗證結果中提取指標")
            return None
            
    except ImportError:
        print("[錯誤] 需要安裝 ultralytics")
        return None
    except Exception as e:
        print(f"[錯誤] 計算 mAP 時發生異常: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def calculate_map(
    model_path: str,
    data_yaml: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    verbose: bool = True
) -> Optional[MapResult]:
    """
    計算 mAP（自動選擇方法）
    
    Args:
        model_path: 模型文件路徑
        data_yaml: 數據配置文件路徑 (YOLO 格式)
        conf_threshold: 置信度閾值
        iou_threshold: IoU 閾值
        verbose: 是否輸出詳細信息
        
    Returns:
        mAP 結果，如果計算失敗則返回 None
    """
    # 根據模型類型選擇計算方法
    ext = Path(model_path).suffix.lower()
    
    if ext in ['.pt', '.pth', '.onnx']:
        return calculate_map_with_ultralytics(
            model_path=model_path,
            data_yaml=data_yaml,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            verbose=verbose
        )
    else:
        print(f"[錯誤] 不支持的模型格式: {ext}")
        return None


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="計算 YOLO 模型的 mAP",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("model_path", help="模型文件路徑")
    parser.add_argument("data_yaml", help="數據配置文件路徑 (YOLO 格式)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度閾值 (默認: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU 閾值 (默認: 0.45)")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細信息")
    
    args = parser.parse_args()
    
    result = calculate_map(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        verbose=args.verbose
    )
    
    if result:
        if args.json:
            print(json.dumps({
                'map_50': result.map_50,
                'map_50_95': result.map_50_95,
                'precision': result.precision,
                'recall': result.recall,
                'num_classes': result.num_classes,
                'per_class_map': result.per_class_map,
                'metrics': result.metrics
            }, indent=2, default=str))
        else:
            print(f"\n[mAP 結果]")
            print(f"  mAP@0.5: {result.map_50:.4f}")
            print(f"  mAP@0.5:0.95: {result.map_50_95:.4f}")
            print(f"  精確度: {result.precision:.4f}")
            print(f"  召回率: {result.recall:.4f}")
            print(f"  類別數量: {result.num_classes}")
            
            if result.per_class_map:
                print(f"\n[每個類別的 mAP]")
                for class_name, map_value in result.per_class_map.items():
                    print(f"  {class_name}: {map_value:.4f}")
    else:
        print("[失敗] 無法計算 mAP")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

