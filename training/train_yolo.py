"""
YOLO 模型訓練模塊

此模塊提供使用 Ultralytics YOLO 訓練模型的功能。
整合了智能數據集清理、探針訓練、自動超參數調整等優化策略。
"""

import os
import sys
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from ultralytics import YOLO
    import torch
    import pandas as pd
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError as e:
    raise ImportError(
        "需要安裝必要的依賴。請確保已安裝:\n"
        "  - ultralytics\n"
        "  - torch\n"
        "  - pandas\n"
        "  - Pillow\n"
        f"原始錯誤: {e}"
    )


# ================= 數據集優化功能 =================

def remove_bad_images(train_images_folder: str, train_labels_folder: str, 
                      min_w: int = 14, min_h: int = 24, verbose: bool = True) -> int:
    """
    移除包含過小框的圖像（保留背景圖像）
    
    Args:
        train_images_folder: 訓練圖像資料夾
        train_labels_folder: 訓練標籤資料夾
        min_w: 最小寬度（像素）
        min_h: 最小高度（像素）
        verbose: 是否輸出詳細信息
        
    Returns:
        移除的圖像數量
    """
    removed = 0
    if not os.path.exists(train_labels_folder):
        return removed
    
    for lbl_file in os.listdir(train_labels_folder):
        if not lbl_file.endswith('.txt'):
            continue
            
        lbl_path = os.path.join(train_labels_folder, lbl_file)
        img_path = os.path.join(train_images_folder, lbl_file.replace('.txt', '.jpg'))
        
        # 如果圖像不存在，嘗試其他格式
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                alt_path = os.path.join(train_images_folder, lbl_file.replace('.txt', ext))
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        if not os.path.exists(img_path):
            continue
        
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 保留背景圖像（無標籤）
            if len(lines) == 0:
                continue
            
            # 讀取圖像尺寸
            with Image.open(img_path) as im:
                W, H = im.size
            
            # 檢查是否有過小的框
            should_remove = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    _, x, y, w, h = map(float, parts[:5])
                    if w * W < min_w or h * H < min_h:
                        should_remove = True
                        break
                except (ValueError, IndexError):
                    continue
            
            if should_remove:
                try:
                    os.remove(lbl_path)
                    os.remove(img_path)
                    removed += 1
                except Exception:
                    pass
        except Exception:
            continue
    
    if verbose and removed > 0:
        print(f"🗑️ 移除了 {removed} 張包含過小框的圖像（背景圖像已保留）")
    
    return removed


def cap_instances(train_labels_folder: str, max_inst: int = 6, verbose: bool = True) -> int:
    """
    限制每張圖像的最大實例數量
    
    Args:
        train_labels_folder: 訓練標籤資料夾
        max_inst: 最大實例數量
        verbose: 是否輸出詳細信息
        
    Returns:
        被限制的標籤文件數量
    """
    capped = 0
    if not os.path.exists(train_labels_folder):
        return capped
    
    for lbl_file in os.listdir(train_labels_folder):
        if not lbl_file.endswith('.txt'):
            continue
        
        lbl_path = os.path.join(train_labels_folder, lbl_file)
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > max_inst:
                random.shuffle(lines)
                with open(lbl_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[:max_inst])
                capped += 1
        except Exception:
            continue
    
    if verbose and capped > 0:
        print(f"📊 限制了 {capped} 個標籤文件的實例數量（最多 {max_inst} 個）")
    
    return capped


def create_validation_set(train_images_folder: str, train_labels_folder: str,
                          val_images_folder: str, val_labels_folder: str,
                          val_split: float = 0.15, bg_val_ratio: float = 0.08,
                          verbose: bool = True) -> Tuple[int, int]:
    """
    智能創建驗證集（區分多實例、單實例和背景圖像）
    
    Args:
        train_images_folder: 訓練圖像資料夾
        train_labels_folder: 訓練標籤資料夾
        val_images_folder: 驗證圖像資料夾
        val_labels_folder: 驗證標籤資料夾
        val_split: 驗證集比例
        bg_val_ratio: 背景圖像在驗證集中的比例
        verbose: 是否輸出詳細信息
        
    Returns:
        (驗證集圖像數量, 背景圖像數量) 元組
    """
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    
    # 如果驗證集已存在且不為空，跳過創建
    if os.path.exists(val_labels_folder) and len(os.listdir(val_labels_folder)) > 0:
        if verbose:
            print(f"📂 驗證集已存在，跳過創建")
        return len(os.listdir(val_labels_folder)), 0
    
    multi, single, bg = [], [], []
    
    if not os.path.exists(train_labels_folder):
        return 0, 0
    
    for lbl_file in os.listdir(train_labels_folder):
        if not lbl_file.endswith('.txt'):
            continue
        
        lbl_path = os.path.join(train_labels_folder, lbl_file)
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                n = len([line for line in f if line.strip()])
            
            img_file = lbl_file.replace('.txt', '.jpg')
            img_path = os.path.join(train_images_folder, img_file)
            if not os.path.exists(img_path):
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = os.path.join(train_images_folder, lbl_file.replace('.txt', ext))
                    if os.path.exists(alt_path):
                        img_file = lbl_file.replace('.txt', ext)
                        img_path = alt_path
                        break
                if not os.path.exists(img_path):
                    continue
            
            if n >= 2:
                multi.append((lbl_file, img_file))
            elif n == 1:
                single.append((lbl_file, img_file))
            else:
                bg.append((lbl_file, img_file))
        except Exception:
            continue
    
    total_train = len(multi) + len(single) + len(bg)
    if total_train == 0:
        return 0, 0
    
    total_val = max(1, int(total_train * val_split))
    bg_target = max(1, int(total_val * bg_val_ratio))
    non_bg_target = total_val - bg_target
    
    chosen = []
    chosen.extend(bg[:bg_target])
    chosen.extend((multi + single)[:non_bg_target])
    chosen = chosen[:total_val]
    
    moved = 0
    for lbl_file, img_file in chosen:
        try:
            src_lbl = os.path.join(train_labels_folder, lbl_file)
            src_img = os.path.join(train_images_folder, img_file)
            dst_lbl = os.path.join(val_labels_folder, lbl_file)
            dst_img = os.path.join(val_images_folder, img_file)
            
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            moved += 1
        except Exception:
            continue
    
    if verbose:
        print(f"📂 創建驗證集: {moved} 張圖像 | {bg_target} 張背景圖像")
    
    return moved, bg_target


def calculate_dataset_stats(train_images_folder: str, train_labels_folder: str) -> Dict:
    """
    計算數據集統計信息
    
    Returns:
        包含圖像數量、實例數量、密度的字典
    """
    if not os.path.exists(train_images_folder) or not os.path.exists(train_labels_folder):
        return {'num_imgs': 0, 'instances': 0, 'density': 0.0}
    
    num_imgs = len([f for f in os.listdir(train_images_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    instances = 0
    for lbl_file in os.listdir(train_labels_folder):
        if not lbl_file.endswith('.txt'):
            continue
        try:
            with open(os.path.join(train_labels_folder, lbl_file), 'r', encoding='utf-8') as f:
                instances += len([line for line in f if line.strip()])
        except Exception:
            continue
    
    density = instances / max(1, num_imgs)
    
    return {
        'num_imgs': num_imgs,
        'instances': instances,
        'density': density
    }


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
    verbose: bool = True,
    # 優化選項
    enable_optimization: bool = False,
    min_w: int = 14,
    min_h: int = 24,
    max_instances: int = 6,
    val_split: float = 0.15,
    bg_val_ratio: float = 0.08,
    probe_epochs: int = 80
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    訓練 YOLO 模型
    
    Args:
        yolo_version: YOLO 版本 ('yolo5', 'yolo6', 'yolo7', 'yolo8', 'yolo9', 'yolo10', 'yolo11', 'yolo12')
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        images_folder: 圖像資料夾路徑
        labels_folder: 標籤資料夾路徑
        output_destination: 輸出目的資料夾
        epochs: 訓練輪數（優化模式下會被自動計算）
        batch_size: 批次大小
        imgsz: 圖像尺寸
        resume: 是否繼續訓練
        verbose: 是否輸出詳細信息
        enable_optimization: 是否啟用優化模式（數據集清理、探針訓練、動態超參數）
        min_w: 最小框寬度（像素）
        min_h: 最小框高度（像素）
        max_instances: 每張圖像最大實例數
        val_split: 驗證集比例
        bg_val_ratio: 背景圖像在驗證集中的比例
        probe_epochs: 探針訓練輪數
        
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
    
    # ================= 優化模式：數據集清理 =================
    if enable_optimization:
        if verbose:
            print("🧼 [優化] 開始數據集清理...")
        
        # 確定訓練和驗證資料夾路徑
        train_images = os.path.join(images_folder, 'train') if os.path.exists(os.path.join(images_folder, 'train')) else images_folder
        train_labels = os.path.join(labels_folder, 'train') if os.path.exists(os.path.join(labels_folder, 'train')) else labels_folder
        val_images = os.path.join(images_folder, 'val') if os.path.exists(os.path.join(images_folder, 'val')) else None
        val_labels = os.path.join(labels_folder, 'val') if os.path.exists(os.path.join(labels_folder, 'val')) else None
        
        # 數據集清理
        remove_bad_images(train_images, train_labels, min_w=min_w, min_h=min_h, verbose=verbose)
        cap_instances(train_labels, max_inst=max_instances, verbose=verbose)
        
        # 創建驗證集（如果不存在）
        if val_images is None or val_labels is None:
            # 嘗試標準 YOLO 格式結構
            dataset_root = os.path.dirname(images_folder) if os.path.dirname(images_folder) else os.path.dirname(labels_folder)
            if not dataset_root or dataset_root == images_folder or dataset_root == labels_folder:
                dataset_root = os.path.dirname(os.path.abspath(images_folder))
                if not dataset_root:
                    dataset_root = os.path.dirname(os.path.abspath(labels_folder))
            
            # 確保目錄結構存在
            val_images_dir = os.path.join(dataset_root, 'images', 'val')
            val_labels_dir = os.path.join(dataset_root, 'labels', 'val')
            os.makedirs(val_images_dir, exist_ok=True)
            os.makedirs(val_labels_dir, exist_ok=True)
            
            val_images = val_images_dir
            val_labels = val_labels_dir
        
        create_validation_set(
            train_images, train_labels,
            val_images, val_labels,
            val_split=val_split, bg_val_ratio=bg_val_ratio,
            verbose=verbose
        )
        
        # 計算數據集統計
        stats = calculate_dataset_stats(train_images, train_labels)
        if verbose:
            print(f"📊 [優化] 數據集統計: {stats['num_imgs']} 張圖像 | {stats['instances']} 個實例 | 密度={stats['density']:.2f}")
        
        # 更新 images_folder 和 labels_folder 以指向標準結構
        # 這樣後續的 data.yaml 創建邏輯能正確識別
        if os.path.exists(os.path.join(images_folder, 'train')) or os.path.exists(os.path.join(labels_folder, 'train')):
            # 已經是標準格式，不需要更新
            pass
        else:
            # 如果不是標準格式，但我們創建了驗證集，需要確保後續邏輯能正確處理
            # 這裡我們暫時保持原樣，讓後續邏輯處理
            pass
    
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
        
        # ================= 優化模式：探針訓練和動態超參數調整 =================
        final_epochs = epochs
        final_lr0 = 0.003
        final_mosaic = 0.6
        final_mixup = 0.05
        final_copy_paste = 0.05
        final_label_smoothing = 0.012
        stats = None
        
        if enable_optimization:
            # 獲取數據集統計（如果還沒有獲取）
            train_images = os.path.join(images_folder, 'train') if os.path.exists(os.path.join(images_folder, 'train')) else images_folder
            train_labels = os.path.join(labels_folder, 'train') if os.path.exists(os.path.join(labels_folder, 'train')) else labels_folder
            stats = calculate_dataset_stats(train_images, train_labels)
            
            # 根據數據集大小選擇模型
            if stats and stats['num_imgs'] < 1500 and model_size == 'n':
                # 小數據集使用 nano 模型
                pass
            elif stats and stats['num_imgs'] >= 1500 and model_size == 'n':
                # 中等數據集建議使用 small 模型
                if verbose:
                    print(f"💡 [優化] 數據集較大，建議使用 's' 模型")
            
            # 根據 GPU 可用性調整批次大小
            if batch_size == 16:  # 使用默認值時才自動調整
                batch_size = 35 if torch.cuda.is_available() else 8
                if verbose:
                    print(f"⚙️ [優化] 自動調整批次大小: {batch_size}")
            
            # 探針訓練
            if verbose:
                print(f"🔍 [優化] 開始探針訓練 ({probe_epochs} epochs)...")
            
            probe_project = os.path.join(output_destination, 'probe_auto')
            probe_results = model.train(
                data=data_yaml_path,
                epochs=probe_epochs,
                batch=batch_size,
                imgsz=imgsz,
                optimizer='AdamW',
                lr0=0.003,
                cos_lr=True,
                mosaic=0.6,
                mixup=0.05,
                copy_paste=0.05,
                project=output_destination,
                name='probe_auto',
                verbose=verbose
            )
            
            # 分析探針訓練結果
            probe_results_csv = os.path.join(probe_project, 'results.csv')
            if os.path.exists(probe_results_csv):
                try:
                    df = pd.read_csv(probe_results_csv)
                    
                    if len(df) >= 80 and 'metrics/mAP50(B)' in df.columns:
                        # 計算不同階段的增長速度
                        early = df["metrics/mAP50(B)"].iloc[20] - df["metrics/mAP50(B)"].iloc[0] if len(df) > 20 else 0
                        mid = df["metrics/mAP50(B)"].iloc[50] - df["metrics/mAP50(B)"].iloc[20] if len(df) > 50 else 0
                        late = df["metrics/mAP50(B)"].iloc[-1] - df["metrics/mAP50(B)"].iloc[50] if len(df) > 50 else 0
                        
                        growth_speed = early + mid * 0.7
                        saturation = max(0.0, late)
                        
                        # 自動選擇最終訓練輪數
                        if stats:
                            base_epochs = 150 if stats['num_imgs'] < 3000 else 180
                            density_bonus = int(40 * min(1.5, stats['density']))
                        else:
                            base_epochs = 150
                            density_bonus = 0
                        growth_bonus = int(60 * min(1.0, growth_speed))
                        
                        final_epochs = base_epochs + density_bonus + growth_bonus
                        final_epochs = max(160, min(240, final_epochs))
                        
                        # 動態調整超參數
                        final_mosaic = 0.65 if early < 0.15 else 0.55
                        final_mixup = 0.07 if mid > 0.12 else 0.05
                        final_copy_paste = 0.05
                        final_label_smoothing = 0.02 if saturation < 0.03 else 0.012
                        final_lr0 = 0.0035 if early < 0.15 else 0.0030
                        
                        if verbose:
                            print(f"⚙️ [優化] 自動選擇最終訓練輪數: {final_epochs}")
                            print(f"⚙️ [優化] 動態超參數: lr0={final_lr0} mosaic={final_mosaic} mixup={final_mixup} copy_paste={final_copy_paste} label_smoothing={final_label_smoothing}")
                    else:
                        if verbose:
                            print(f"⚠️ [優化] 探針訓練結果不完整，使用默認參數")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ [優化] 分析探針訓練結果時出錯: {e}，使用默認參數")
            
            # 從探針訓練的最佳模型繼續
            probe_best_model = os.path.join(probe_project, 'weights', 'best.pt')
            if os.path.exists(probe_best_model):
                model = YOLO(probe_best_model)
                if verbose:
                    print(f"📦 [優化] 從探針訓練最佳模型繼續: {probe_best_model}")
        
        # 訓練模型 - 使用 data.yaml 的絕對路徑
        train_kwargs = {
            'data': data_yaml_path,
            'epochs': final_epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'project': output_destination,
            'resume': resume,
            'verbose': verbose
        }
        
        # 如果啟用優化模式，添加動態超參數
        if enable_optimization:
            train_kwargs.update({
                'optimizer': 'AdamW',
                'lr0': final_lr0,
                'cos_lr': True,
                'mosaic': final_mosaic,
                'mixup': final_mixup,
                'copy_paste': final_copy_paste,
                'label_smoothing': final_label_smoothing,
                'patience': 45,
                'name': 'final_auto'
            })
        
        results = model.train(**train_kwargs)
        
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
    parser.add_argument("--enable-optimization", action="store_true", help="啟用優化模式（數據集清理、探針訓練、動態超參數）")
    parser.add_argument("--min-w", type=int, default=14, help="最小框寬度（像素）")
    parser.add_argument("--min-h", type=int, default=24, help="最小框高度（像素）")
    parser.add_argument("--max-instances", type=int, default=6, help="每張圖像最大實例數")
    parser.add_argument("--val-split", type=float, default=0.15, help="驗證集比例")
    parser.add_argument("--bg-val-ratio", type=float, default=0.08, help="背景圖像在驗證集中的比例")
    parser.add_argument("--probe-epochs", type=int, default=80, help="探針訓練輪數")
    
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
        verbose=args.verbose,
        enable_optimization=args.enable_optimization,
        min_w=args.min_w,
        min_h=args.min_h,
        max_instances=args.max_instances,
        val_split=args.val_split,
        bg_val_ratio=args.bg_val_ratio,
        probe_epochs=args.probe_epochs
    )
    
    if success:
        print(f"[成功] 訓練完成！模型保存在: {model_path}")
        sys.exit(0)
    else:
        print("[失敗] 訓練失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()

