"""
YOLO 模型訓練模塊

此模塊提供使用 Ultralytics YOLO 訓練模型的功能。
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "需要安裝 Ultralytics YOLO。請確保已安裝:\n"
        "  - ultralytics\n"
        f"原始錯誤: {e}"
    )


def train_yolo_model(
    yolo_version: str = 'yolo12',
    model_size: str = 'n',
    images_folder: str = '',
    labels_folder: str = '',
    output_destination: str = '',
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    resume: bool = False,
    verbose: bool = True
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    訓練 YOLO 模型
    
    Args:
        yolo_version: YOLO 版本 ('yolo5', 'yolo6', 'yolo7', 'yolo8', 'yolo9', 'yolo10', 'yolo11', 'yolo12')
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        images_folder: 圖像資料夾路徑
        labels_folder: 標籤資料夾路徑
        output_destination: 輸出目的資料夾
        epochs: 訓練輪數
        batch_size: 批次大小
        imgsz: 圖像尺寸
        resume: 是否繼續訓練
        verbose: 是否輸出詳細信息
        
    Returns:
        (成功標誌, 模型路徑, 訓練結果字典) 元組
    """
    # 驗證輸入
    if not images_folder or not os.path.exists(images_folder):
        print(f"[錯誤] 圖像資料夾不存在: {images_folder}")
        return False, None, None
    
    if not labels_folder or not os.path.exists(labels_folder):
        print(f"[錯誤] 標籤資料夾不存在: {labels_folder}")
        return False, None, None
    
    if not output_destination:
        output_destination = os.path.join(os.getcwd(), 'runs', 'train')
    
    os.makedirs(output_destination, exist_ok=True)
    
    # 構建模型名稱（確保格式正確）
    # Ultralytics YOLO 支持的格式：
    # - YOLOv5: yolov5n.pt, yolov5s.pt 等
    # - YOLOv8+: yolo8n.pt, yolo8s.pt, yolo10n.pt, yolo10s.pt 等
    # 注意：YOLOv5 使用 'yolov5' 前綴，其他版本使用 'yolo' 前綴
    
    # 處理 YOLOv5 的特殊格式
    if yolo_version.lower() == 'yolo5':
        model_name = f"yolov5{model_size}.pt"
    else:
        model_name = f"{yolo_version}{model_size}.pt"
    
    try:
        if verbose:
            print(f"[訓練] 開始訓練 YOLO 模型...")
            print(f"[訓練] 模型: {model_name}")
            print(f"[訓練] 圖像資料夾: {images_folder}")
            print(f"[訓練] 標籤資料夾: {labels_folder}")
            print(f"[訓練] 輸出目錄: {output_destination}")
            print(f"[訓練] 輪數: {epochs}, 批次大小: {batch_size}, 圖像尺寸: {imgsz}")
        
        # 檢查模型文件是否存在（檢查當前目錄和 Ultralytics 默認位置）
        ultralytics_weights_dir = os.path.join(os.path.expanduser('~'), '.ultralytics', 'weights')
        possible_paths = [
            model_name,  # 當前目錄
            os.path.join(ultralytics_weights_dir, model_name),  # Ultralytics 默認位置
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = os.path.abspath(path)
                if verbose:
                    print(f"[訓練] 找到本地模型: {model_path}")
                break
        
        # 載入模型（如果不存在，Ultralytics 會自動下載）
        # YOLO() 類會自動處理下載，模型會保存到 ~/.ultralytics/weights/ 目錄
        if not model_path:
            if verbose:
                print(f"[訓練] 模型文件不存在，將自動從 Ultralytics 下載: {model_name}")
                print(f"[訓練] 下載可能需要一些時間，請稍候...")
                print(f"[訓練] 模型將下載到: {ultralytics_weights_dir}")
        
        try:
            # 直接使用模型名稱，Ultralytics 會自動下載（如果不存在）
            # 這是最簡單和可靠的方法
            model = YOLO(model_name)
            
            if verbose:
                # 檢查模型是否已下載
                downloaded_path = os.path.join(ultralytics_weights_dir, model_name)
                if os.path.exists(downloaded_path):
                    print(f"[訓練] 模型已成功下載到: {downloaded_path}")
                print(f"[訓練] 模型載入成功")
        except FileNotFoundError as e:
            # 如果自動下載失敗，提供更詳細的錯誤信息
            if verbose:
                print(f"[錯誤] 模型載入失敗: {e}")
                print(f"[錯誤] 無法找到或下載模型: {model_name}")
                print(f"[提示] 可能的原因:")
                print(f"  1. 網絡連接問題 - 請檢查網絡連接")
                print(f"  2. 模型名稱不正確 - 請確認 YOLO 版本和模型大小")
                print(f"  3. Ultralytics 版本問題 - 請更新 ultralytics: pip install --upgrade ultralytics")
                print(f"[提示] 手動下載:")
                print(f"  可以從 https://github.com/ultralytics/assets/releases 下載模型文件")
                print(f"  並放在以下位置之一:")
                print(f"    - 當前目錄: {os.getcwd()}")
                print(f"    - Ultralytics 目錄: {ultralytics_weights_dir}")
            raise
        except Exception as e:
            if verbose:
                print(f"[錯誤] 模型載入失敗: {e}")
                import traceback
                traceback.print_exc()
            raise
        
        # 創建 data.yaml 文件
        # YOLO 需要 data.yaml 來指定數據路徑
        import yaml
        
        # 確保路徑使用正斜杠（YOLO 要求）
        images_folder_normalized = images_folder.replace('\\', '/')
        labels_folder_normalized = labels_folder.replace('\\', '/')
        
        # 檢查是否為標準 YOLO 格式（train/val/test 子資料夾）
        train_images = os.path.join(images_folder, 'train')
        train_labels = os.path.join(labels_folder, 'train')
        
        # 確定數據集根目錄
        if os.path.exists(train_images) and os.path.exists(train_labels):
            # 標準格式：images/train, labels/train 等
            dataset_root = os.path.dirname(images_folder)
            data_yaml_path = os.path.join(dataset_root, 'data.yaml')
            
            # 檢查是否有驗證集
            val_images = os.path.join(images_folder, 'val')
            val_path = 'images/val' if os.path.exists(val_images) else 'images/train'
            
            data_config = {
                'path': dataset_root.replace('\\', '/'),
                'train': 'images/train',
                'val': val_path,  # 必須有 val，如果沒有則使用 train
            }
            
            if os.path.exists(os.path.join(images_folder, 'test')):
                data_config['test'] = 'images/test'
            
            # 計算類別數量（從 labels 資料夾中的文件推斷）
            import glob
            label_files = glob.glob(os.path.join(train_labels, '*.txt'))
            if label_files:
                # 讀取第一個標籤文件來確定類別數量
                try:
                    with open(label_files[0], 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            # YOLO 格式：class_id x y w h
                            num_classes = max([int(line.split()[0]) for line in [first_line] + f.readlines() if line.strip()]) + 1
                            data_config['nc'] = num_classes
                            # 創建類別名稱列表
                            data_config['names'] = [f'class{i}' for i in range(num_classes)]
                except:
                    pass
            
            if 'nc' not in data_config:
                data_config['nc'] = 1  # 默認1個類別
                data_config['names'] = ['class0']
            
        else:
            # 非標準格式：直接使用 images 和 labels 資料夾
            # 創建臨時 data.yaml 在輸出目錄
            dataset_root = os.path.dirname(images_folder)
            data_yaml_path = os.path.join(output_destination, 'data.yaml')
            os.makedirs(output_destination, exist_ok=True)
            
            # 計算相對路徑
            rel_images = os.path.relpath(images_folder, dataset_root).replace('\\', '/')
            rel_labels = os.path.relpath(labels_folder, dataset_root).replace('\\', '/')
            
            # YOLO 要求必須有 train 和 val，如果沒有 val，使用 train 作為 val
            data_config = {
                'path': dataset_root.replace('\\', '/'),
                'train': rel_images,
                'val': rel_images,  # 如果沒有驗證集，使用訓練集
            }
            
            # 嘗試確定類別數量
            import glob
            label_files = glob.glob(os.path.join(labels_folder, '*.txt'))
            if label_files:
                try:
                    max_class = 0
                    for label_file in label_files[:10]:  # 檢查前10個文件
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    max_class = max(max_class, class_id)
                    num_classes = max_class + 1
                    data_config['nc'] = num_classes
                    data_config['names'] = [f'class{i}' for i in range(num_classes)]
                except:
                    pass
            
            if 'nc' not in data_config:
                data_config['nc'] = 1
                data_config['names'] = ['class0']
        
        # 寫入 data.yaml 文件
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        if verbose:
            print(f"[訓練] 創建 data.yaml: {data_yaml_path}")
            print(f"[訓練] 數據配置: {data_config}")
        
        # 訓練模型 - 使用 data.yaml 的絕對路徑
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=output_destination,
            resume=resume,
            verbose=verbose
        )
        
        # 獲取最佳模型路徑
        best_model_path = None
        if hasattr(results, 'save_dir'):
            best_model_path = os.path.join(str(results.save_dir), 'weights', 'best.pt')
            if not os.path.exists(best_model_path):
                # 嘗試其他可能的路徑
                possible_paths = [
                    os.path.join(str(results.save_dir), 'best.pt'),
                    os.path.join(output_destination, 'weights', 'best.pt'),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        best_model_path = path
                        break
        
        if verbose:
            print(f"[成功] 訓練完成！")
            if best_model_path:
                print(f"[成功] 最佳模型保存在: {best_model_path}")
        
        return True, best_model_path, {
            'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') else output_destination,
            'best_model': best_model_path
        }
        
    except Exception as e:
        print(f"[錯誤] 訓練過程中發生異常: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False, None, None


# 命令行接口
def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="訓練 YOLO 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--yolo-version", type=str, default='yolo12', help="YOLO 版本 (yolo5/yolo6/yolo7/yolo8/yolo9/yolo10/yolo11/yolo12)")
    parser.add_argument("--model-size", type=str, default='n', help="模型大小 (n/s/m/l/x)")
    parser.add_argument("--images", type=str, required=True, help="圖像資料夾路徑")
    parser.add_argument("--labels", type=str, required=True, help="標籤資料夾路徑")
    parser.add_argument("--output", type=str, default='', help="輸出目的資料夾")
    parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="圖像尺寸")
    parser.add_argument("--resume", action="store_true", help="繼續訓練")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細信息")
    
    args = parser.parse_args()
    
    print(f"[訓練] 開始訓練: {args.yolo_version}{args.model_size}")
    success, model_path, results = train_yolo_model(
        yolo_version=args.yolo_version,
        model_size=args.model_size,
        images_folder=args.images,
        labels_folder=args.labels,
        output_destination=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        resume=args.resume,
        verbose=args.verbose
    )
    
    if success:
        print(f"[成功] 訓練完成！模型保存在: {model_path}")
        sys.exit(0)
    else:
        print("[失敗] 訓練失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

