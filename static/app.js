// API 基礎 URL
const API_BASE = 'http://127.0.0.1:5000/api';

// 選中的尺寸列表
let selectedSizes = [640];

// 翻譯錯誤代碼
function translateErrorCode(errorCode, conversionStep) {
    const errorMessages = {
        'pt_to_onnx_failed': 'errors.ptToOnnxFailed',
        'onnx_to_engine_failed': 'errors.onnxToEngineFailed',
        'unsupported_conversion_type': 'errors.unsupportedType'
    };
    
    const key = errorMessages[errorCode];
    if (key) {
        return window.i18n.t(key);
    }
    
    // 回退到轉換步驟訊息
    if (conversionStep === 'pt_to_onnx') {
        return window.i18n.t('errors.ptToOnnxFailed');
    } else if (conversionStep === 'onnx_to_engine') {
        return window.i18n.t('errors.onnxToEngineFailed');
    }
    
    return window.i18n.t('convert.conversionFailed');
}

// 初始化
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize i18n first
    await window.i18n.initI18n();
    
    initTabs();
    initForm();
    // 初始化工具頁面
    initTools();
    // 初始化系統資訊頁面
    initSystemInfo();
    
    loadModels();
    loadFolders();
    
    // 監聽模型選擇變化
    document.getElementById('model-select').addEventListener('change', (e) => {
        updateConversionTypes(e.target.value);
    });
    
    // 監聽轉換類型變化
    document.getElementById('conversion-type').addEventListener('change', (e) => {
        const type = e.target.value;
        const engineOptions = document.getElementById('engine-options');
        if (type === 'pt_to_engine' || type === 'onnx_to_engine') {
            engineOptions.style.display = 'block';
        } else {
            engineOptions.style.display = 'none';
        }
    });
    
    // 初始更新轉換類型
    updateConversionTypes('');
    
    // 優化選項顯示/隱藏
    const enableOptimizationCheckbox = document.getElementById('enable-optimization');
    const optimizationOptions = document.getElementById('optimization-options');
    if (enableOptimizationCheckbox && optimizationOptions) {
        enableOptimizationCheckbox.addEventListener('change', (e) => {
            optimizationOptions.style.display = e.target.checked ? 'block' : 'none';
        });
    }
});

// Tab 切換
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const activeTitle = document.getElementById('active-tab-title');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // 更新按鈕狀態
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 更新內容顯示
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-tab`) {
                    content.classList.add('active');
                }
            });

            // 更新頁面標題 (i18n)
            if (activeTitle) {
                activeTitle.setAttribute('data-i18n', `${targetTab}.title`);
                if (window.i18n) window.i18n.updateTranslations();
            }
        });
    });
}

// 初始化表單
function initForm() {
    // 上傳按鈕
    document.getElementById('upload-btn').addEventListener('click', uploadFile);
    
    // 刷新模型列表
    document.getElementById('refresh-models').addEventListener('click', loadModels);
    
    // 刷新資料夾列表
    document.getElementById('refresh-folders').addEventListener('click', loadFolders);
    
    // 添加自定義尺寸
    document.getElementById('add-custom-size').addEventListener('click', addCustomSize);
    
    // 尺寸選擇框
    document.querySelectorAll('.size-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const value = parseInt(e.target.value);
            if (e.target.checked) {
                if (!selectedSizes.includes(value)) {
                    selectedSizes.push(value);
                }
            } else {
                selectedSizes = selectedSizes.filter(s => s !== value);
            }
        });
    });
    
    // 轉換按鈕
    document.getElementById('convert-btn').addEventListener('click', startConversion);
    
    // 訓練相關按鈕
    if (document.getElementById('train-btn')) {
        document.getElementById('train-btn').addEventListener('click', startTraining);
    }
}

// 更新轉換類型選項（根據選擇的模型類型）
function updateConversionTypes(selectedModelPath) {
    const conversionTypeSelect = document.getElementById('conversion-type');
    const allOptions = conversionTypeSelect.querySelectorAll('option');
    
    // 如果沒有選擇模型，顯示所有選項
    if (!selectedModelPath) {
        allOptions.forEach(option => {
            option.style.display = '';
        });
        return;
    }
    
    // 獲取模型文件擴展名
    const modelPath = selectedModelPath.toLowerCase();
    let modelType = '';
    
    if (modelPath.endsWith('.pt') || modelPath.endsWith('.pth')) {
        modelType = 'pt';
    } else if (modelPath.endsWith('.onnx')) {
        modelType = 'onnx';
    } else {
        // 未知類型，顯示所有選項
        allOptions.forEach(option => {
            option.style.display = '';
        });
        return;
    }
    
    // 根據模型類型顯示/隱藏選項
    allOptions.forEach(option => {
        const value = option.value;
        
        if (modelType === 'pt') {
            // PT 模型：只顯示 pt_to_onnx 和 pt_to_engine
            if (value === 'pt_to_onnx' || value === 'pt_to_engine') {
                option.style.display = '';
            } else {
                option.style.display = 'none';
            }
        } else if (modelType === 'onnx') {
            // ONNX 模型：只顯示 onnx_to_engine
            if (value === 'onnx_to_engine') {
                option.style.display = '';
            } else {
                option.style.display = 'none';
            }
        }
    });
    
    // 如果當前選擇的選項被隱藏了，自動選擇第一個可見的選項
    const selectedOption = conversionTypeSelect.options[conversionTypeSelect.selectedIndex];
    if (selectedOption.style.display === 'none') {
        for (let i = 0; i < allOptions.length; i++) {
            if (allOptions[i].style.display !== 'none') {
                conversionTypeSelect.selectedIndex = i;
                // 觸發 change 事件以更新 engine 選項
                conversionTypeSelect.dispatchEvent(new Event('change'));
                break;
            }
        }
    }
}

// 初始化工具頁面
function initTools() {
    // 子標籤切換
    const subTabBtns = document.querySelectorAll('.tab-sub-btn');
    const subTabContents = document.querySelectorAll('.subtab-content');
    
    subTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // 更新按鈕狀態
            subTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 更新內容顯示
            const target = btn.getAttribute('data-subtab');
            subTabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${target}-subtab`) {
                    content.classList.add('active');
                }
            });
        });
    });

    // 刷新工具頁面的模型列表
    const toolsRefreshBtn = document.getElementById('tools-refresh-models');
    if (toolsRefreshBtn) {
        toolsRefreshBtn.addEventListener('click', loadModels);
    }
    
    // 驗證按鈕
    const validateBtn = document.getElementById('validate-btn');
    if (validateBtn) {
        validateBtn.addEventListener('click', startValidation);
    }
    
    // 瀏覽 YAML 文件
    const browseYamlBtn = document.getElementById('browse-yaml-btn');
    const uploadYamlInput = document.getElementById('upload-yaml');
    if (browseYamlBtn && uploadYamlInput) {
        browseYamlBtn.addEventListener('click', () => uploadYamlInput.click());
        uploadYamlInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                // 這裡我們只是獲取文件名，實際上後端需要完整路徑或上傳文件
                // 為了簡化，我們假設用戶選擇的是服務器上的文件，或者我們需要實現文件上傳
                // 這裡簡單地將文件名填入輸入框，並提示用戶這只是模擬
                // 在實際應用中，可能需要實現文件上傳 API
                // 由於瀏覽器安全限制，我們無法獲取完整路徑
                // 這裡我們使用一個變通方法：假設文件在 uploads 資料夾中，或者讓用戶手動輸入
                // 這裡暫時只顯示文件名
                document.getElementById('data-yaml-path').value = e.target.files[0].name;
                alert(window.i18n.t('tools.yamlUploadHint'));
            }
        });
    }
    
    // mAP 計算按鈕
    const mapBtn = document.getElementById('map-btn');
    if (mapBtn) {
        mapBtn.addEventListener('click', startMapCalculation);
    }
}

// 開始驗證和分析
async function startValidation() {
    const modelSelect = document.getElementById('tools-model-select');
    const modelPath = modelSelect.value;
    
    if (!modelPath) {
        alert(window.i18n.t('convert.selectModelError'));
        return;
    }
    
    const validateBtn = document.getElementById('validate-btn');
    validateBtn.disabled = true;
    validateBtn.textContent = window.i18n.t('tools.processing');
    
    document.getElementById('validate-result-container').style.display = 'none';
    document.getElementById('validation-output').innerHTML = '<div class="loader"></div>';
    document.getElementById('analysis-output').innerHTML = '<div class="loader"></div>';
    document.getElementById('validate-result-container').style.display = 'block';
    
    try {
        // 並行執行驗證和分析
        const [validateRes, analyzeRes] = await Promise.all([
            fetch(`${API_BASE}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelPath })
            }),
            fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelPath })
            })
        ]);
        
        const validateData = await validateRes.json();
        const analyzeData = await analyzeRes.json();
        
        // 渲染驗證結果
        const valOutput = document.getElementById('validation-output');
        if (validateData.success) {
            let html = `<div class="status-item ${validateData.is_valid ? 'success' : 'error'}">
                <strong>Status:</strong> ${validateData.is_valid ? 'Valid' : 'Invalid'}
            </div>`;
            html += `<div class="info-list">
                <div><strong>Type:</strong> ${validateData.model_type}</div>
                <div><strong>Input Shape:</strong> ${JSON.stringify(validateData.input_shape)}</div>
                <div><strong>Output Shape:</strong> ${JSON.stringify(validateData.output_shape)}</div>
                <div><strong>Classes:</strong> ${validateData.num_classes || 'N/A'}</div>
            </div>`;
            
            if (validateData.errors && validateData.errors.length > 0) {
                html += `<div class="error-list"><strong>Errors:</strong><ul>${validateData.errors.map(e => `<li>${e}</li>`).join('')}</ul></div>`;
            }
            if (validateData.warnings && validateData.warnings.length > 0) {
                html += `<div class="warning-list"><strong>Warnings:</strong><ul>${validateData.warnings.map(w => `<li>${w}</li>`).join('')}</ul></div>`;
            }
            valOutput.innerHTML = html;
        } else {
            valOutput.innerHTML = `<div class="alert alert-error">${validateData.error}</div>`;
        }
        
        // 渲染分析結果
        const anaOutput = document.getElementById('analysis-output');
        if (analyzeData.success) {
            let html = `<div class="info-list">
                <div><strong>File Size:</strong> ${analyzeData.file_size_mb.toFixed(2)} MB</div>
                <div><strong>Params:</strong> ${analyzeData.num_parameters ? analyzeData.num_parameters.toLocaleString() : 'N/A'}</div>
                <div><strong>Precision:</strong> ${analyzeData.precision || 'N/A'}</div>
                <div><strong>Opset:</strong> ${analyzeData.opset_version || 'N/A'}</div>
            </div>`;
            
            if (analyzeData.metadata) {
                // 特別處理 classes 資訊
                if (analyzeData.metadata.classes && Array.isArray(analyzeData.metadata.classes)) {
                    html += `<div class="metadata-section">
                        <strong>Classes (${analyzeData.metadata.num_classes || analyzeData.metadata.classes.length}):</strong>
                        <div class="classes-list">
                            ${analyzeData.metadata.classes.map((cls, idx) => 
                                `<span class="class-badge" title="Class ${idx}">${idx}: ${cls}</span>`
                            ).join('')}
                        </div>
                    </div>`;
                }
                
                // 顯示其他元數據（排除已顯示的 classes 和 num_classes）
                const otherMetadata = {};
                for (const [key, value] of Object.entries(analyzeData.metadata)) {
                    if (key !== 'classes' && key !== 'num_classes') {
                        otherMetadata[key] = value;
                    }
                }
                
                if (Object.keys(otherMetadata).length > 0) {
                    html += `<div class="metadata-list"><strong>Other Metadata:</strong><pre>${JSON.stringify(otherMetadata, null, 2)}</pre></div>`;
                }
            }
            anaOutput.innerHTML = html;
        } else {
            anaOutput.innerHTML = `<div class="alert alert-error">${analyzeData.error}</div>`;
        }
        
    } catch (error) {
        console.error('Validation failed:', error);
        alert('Validation failed: ' + error.message);
    } finally {
        validateBtn.disabled = false;
        validateBtn.textContent = window.i18n.t('tools.validateAnalyzeBtn');
    }
}

// 開始 mAP 計算
async function startMapCalculation() {
    const modelSelect = document.getElementById('tools-model-select');
    const modelPath = modelSelect.value;
    const dataYaml = document.getElementById('data-yaml-path').value;
    const confThreshold = parseFloat(document.getElementById('conf-threshold').value);
    const iouThreshold = parseFloat(document.getElementById('iou-threshold').value);
    
    if (!modelPath) {
        alert(window.i18n.t('convert.selectModelError'));
        return;
    }
    
    if (!dataYaml) {
        alert(window.i18n.t('tools.selectDataYaml'));
        return;
    }
    
    const mapBtn = document.getElementById('map-btn');
    mapBtn.disabled = true;
    
    document.getElementById('map-result-container').style.display = 'none';
    document.getElementById('map-progress-container').style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/map`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                data_yaml: dataYaml,
                conf_threshold: confThreshold,
                iou_threshold: iouThreshold
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('map-50-value').textContent = data.map_50.toFixed(4);
            document.getElementById('map-50-95-value').textContent = data.map_50_95.toFixed(4);
            document.getElementById('precision-value').textContent = data.precision.toFixed(4);
            document.getElementById('recall-value').textContent = data.recall.toFixed(4);
            
            const tbody = document.getElementById('per-class-map-body');
            tbody.innerHTML = '';
            
            if (data.per_class_map) {
                Object.entries(data.per_class_map).forEach(([cls, map]) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${cls}</td><td>${map.toFixed(4)}</td>`;
                    tbody.appendChild(row);
                });
            }
            
            document.getElementById('map-result-container').style.display = 'block';
        } else {
            alert(window.i18n.t('tools.mapFailed') + ': ' + data.error);
        }
    } catch (error) {
        console.error('mAP calculation failed:', error);
        alert(window.i18n.t('tools.mapFailed') + ': ' + error.message);
    } finally {
        document.getElementById('map-progress-container').style.display = 'none';
        mapBtn.disabled = false;
    }
}

// 載入模型列表
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        const toolsSelect = document.getElementById('tools-model-select'); // 工具頁面的選擇框
        const placeholder = window.i18n.t('convert.selectModelPlaceholder');
        select.innerHTML = `<option value="">${placeholder}</option>`;
        if (toolsSelect) toolsSelect.innerHTML = `<option value="">${placeholder}</option>`;
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            // 顯示資料夾名稱（如果有）
            const folderPrefix = model.folder ? `[${model.folder}] ` : '';
            option.textContent = `${folderPrefix}${model.name} (${formatFileSize(model.size)})`;
            option.setAttribute('data-type', model.type); // 保存模型類型
            select.appendChild(option);
            
            // 添加到工具頁面的選擇框
            if (toolsSelect) {
                const toolsOption = option.cloneNode(true);
                toolsSelect.appendChild(toolsOption);
            }
        });
        
        // 如果之前有選擇，觸發更新
        if (select.value) {
            updateConversionTypes(select.value);
        }
    } catch (error) {
        console.error('Load models failed:', error);
        showError(window.i18n.t('convert.selectModelError'));
    }
}

// 載入資料夾列表
async function loadFolders() {
    try {
        const response = await fetch(`${API_BASE}/folders`);
        const data = await response.json();
        
        const select = document.getElementById('output-folder');
        select.innerHTML = '';
        
        data.folders.forEach(folder => {
            const option = document.createElement('option');
            option.value = folder.path;
            option.textContent = folder.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('載入資料夾失敗:', error);
    }
}

// 上傳文件
async function uploadFile() {
    const fileInput = document.getElementById('upload-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert(window.i18n.t('convert.selectModelError'));
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(window.i18n.t('convert.uploadSuccess'));
            await loadModels();
            // 自動選擇剛上傳的模型
            const select = document.getElementById('model-select');
            const toolsSelect = document.getElementById('tools-model-select');
            
            const updateSelect = (sel) => {
                if (!sel) return;
                for (let i = 0; i < sel.options.length; i++) {
                    if (sel.options[i].value === data.path) {
                        sel.selectedIndex = i;
                        if (sel.id === 'model-select') {
                            updateConversionTypes(data.path);
                        }
                        sel.dispatchEvent(new Event('change'));
                        break;
                    }
                }
            };
            
            updateSelect(select);
            updateSelect(toolsSelect);
            
            fileInput.value = '';
        } else {
            alert(window.i18n.t('convert.uploadFailed') + ': ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Upload failed:', error);
        alert(window.i18n.t('convert.uploadFailed') + ': ' + error.message);
    }
}

// 添加自定義尺寸
function addCustomSize() {
    const input = document.getElementById('custom-size');
    const value = parseInt(input.value);
    
    if (isNaN(value) || value < 64 || value > 2048) {
        alert(window.i18n.t('convert.invalidSize'));
        return;
    }
    
    if (selectedSizes.includes(value)) {
        alert(window.i18n.t('convert.sizeExists'));
        return;
    }
    
    // 添加到選中列表
    selectedSizes.push(value);
    selectedSizes.sort((a, b) => a - b);
    
    // 創建新的選項
    const presetSizes = document.querySelector('.preset-sizes');
    const label = document.createElement('label');
    label.className = 'size-option';
    label.innerHTML = `
        <input type="checkbox" value="${value}" class="size-checkbox" checked> ${value}
    `;
    
    const checkbox = label.querySelector('.size-checkbox');
    checkbox.addEventListener('change', (e) => {
        const val = parseInt(e.target.value);
        if (e.target.checked) {
            if (!selectedSizes.includes(val)) {
                selectedSizes.push(val);
            }
        } else {
            selectedSizes = selectedSizes.filter(s => s !== val);
        }
    });
    
    presetSizes.appendChild(label);
    input.value = '';
}

// 開始轉換
async function startConversion() {
    const modelSelect = document.getElementById('model-select');
    const modelPath = modelSelect.value;
    const conversionType = document.getElementById('conversion-type').value;
    const outputFolder = document.getElementById('output-folder').value;
    
    if (!modelPath) {
        alert(window.i18n.t('convert.selectModelError'));
        return;
    }
    
    if (selectedSizes.length === 0) {
        alert(window.i18n.t('convert.selectSizeError'));
        return;
    }
    
    // 隱藏之前的結果
    document.getElementById('result-container').style.display = 'none';
    document.getElementById('error-container').style.display = 'none';
    document.getElementById('progress-container').style.display = 'block';
    
    // 禁用轉換按鈕
    const convertBtn = document.getElementById('convert-btn');
    convertBtn.disabled = true;
    convertBtn.textContent = window.i18n.t('convert.converting');
    
    // 獲取 TensorRT 選項
    const tensorrtVersion = document.getElementById('tensorrt-version') ? document.getElementById('tensorrt-version').value : '8.6';
    const enableFp16 = document.getElementById('enable-fp16').checked;
    const enableFp8 = document.getElementById('enable-fp8').checked;
    const fixedInputSize = document.getElementById('fixed-input-size').checked;
    const workspaceSize = parseInt(document.getElementById('workspace-size').value) || 1;
    
    const tasks = [];
    const totalSizes = selectedSizes.length;
    
    // 為每個選中的尺寸執行轉換（序列化執行以避免 ONNX 導出衝突）
    // 只發起第一個轉換請求，其他的等待上一個完成後才發起
    try {
        const response = await fetch(`${API_BASE}/convert`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: conversionType,
                model_path: modelPath,
                output_folder: outputFolder,
                imgsz: selectedSizes[0],
                enable_fp16: enableFp16,
                enable_fp8: enableFp8,
                fixed_input_size: fixedInputSize,
                workspace_size_gb: workspaceSize,
                tensorrt_version: tensorrtVersion
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            tasks.push({ taskId: data.task_id, size: selectedSizes[0] });
        } else {
            showError(window.i18n.t('convert.conversionFailed') + ': ' + (data.error || 'Unknown error'));
            convertBtn.disabled = false;
            convertBtn.textContent = window.i18n.t('convert.startConvert');
            return;
        }
    } catch (error) {
        console.error('Conversion failed:', error);
        showError(window.i18n.t('convert.conversionFailed') + ': ' + error.message);
        convertBtn.disabled = false;
        convertBtn.textContent = window.i18n.t('convert.startConvert');
        return;
    }
    
    // 監控所有任務（包括序列化發起剩餘的轉換）
    if (tasks.length > 0) {
        monitorAllTasksSequential(tasks, selectedSizes, totalSizes, {
            conversionType,
            modelPath,
            outputFolder,
            enableFp16,
            enableFp8,
            fixedInputSize,
            workspaceSize,
            tensorrtVersion
        });
    }
}

// 序列化監控並發起後續轉換任務
async function monitorAllTasksSequential(tasks, allSizes, totalSizes, conversionParams) {
    const checkInterval = 1000;
    const completedTasks = new Set();
    const results = [];
    let currentSizeIndex = 1; // 從第二個尺寸開始（第一個已經發起）
    
    const startNextConversion = async () => {
        if (currentSizeIndex >= allSizes.length) {
            return null; // 所有轉換都已發起
        }
        
        const size = allSizes[currentSizeIndex];
        currentSizeIndex++;
        
        try {
            const response = await fetch(`${API_BASE}/convert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: conversionParams.conversionType,
                    model_path: conversionParams.modelPath,
                    output_folder: conversionParams.outputFolder,
                    imgsz: size,
                    enable_fp16: conversionParams.enableFp16,
                    enable_fp8: conversionParams.enableFp8,
                    fixed_input_size: conversionParams.fixedInputSize,
                    workspace_size_gb: conversionParams.workspaceSize,
                    tensorrt_version: conversionParams.tensorrtVersion
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                return { taskId: data.task_id, size: size };
            }
        } catch (error) {
            console.error(`Failed to start conversion for size ${size}:`, error);
        }
        return null;
    };
    
    const checkAllTasks = async () => {
        let allCompleted = true;
        let totalProgress = 0;
        let currentMessage = '';
        
        for (const { taskId, size } of tasks) {
            if (completedTasks.has(taskId)) {
                totalProgress += 100;
                continue;
            }
            
            try {
                const response = await fetch(`${API_BASE}/task/${taskId}`);
                const task = await response.json();
                
                if (task.status === 'completed') {
                    completedTasks.add(taskId);
                    results.push({ size, path: task.output_path });
                    totalProgress += 100;
                    
                    // 當一個任務完成後，發起下一個轉換（如果還有）
                    const nextTask = await startNextConversion();
                    if (nextTask) {
                        tasks.push(nextTask);
                    }
                } else if (task.status === 'failed') {
                    completedTasks.add(taskId);
                    
                    let errorMessage;
                    if (task.error_code) {
                        errorMessage = translateErrorCode(task.error_code, task.conversion_step);
                    } else {
                        errorMessage = task.error || window.i18n.t('convert.conversionFailed');
                    }
                    
                    showError(window.i18n.t('messages.sizeComplete', { size: size }) + ' ' + window.i18n.t('convert.conversionFailed') + ': ' + errorMessage);
                    const convertBtn = document.getElementById('convert-btn');
                    convertBtn.disabled = false;
                    convertBtn.textContent = window.i18n.t('convert.startConvert');
                    return;
                } else if (task.status === 'running') {
                    allCompleted = false;
                    totalProgress += task.progress || 0;
                    if (task.message) {
                        currentMessage = window.i18n.t('messages.sizeComplete', { size: size }) + ': ' + task.message;
                    } else {
                        currentMessage = window.i18n.t('messages.converting');
                    }
                }
            } catch (error) {
                console.error('檢查任務狀態失敗:', error);
                allCompleted = false;
            }
        }
        
        const avgProgress = totalProgress / totalSizes;
        updateProgress(avgProgress, currentMessage || window.i18n.t('messages.progress', { completed: completedTasks.size, total: totalSizes }));
        
        if (allCompleted && completedTasks.size === tasks.length) {
            updateProgress(100, window.i18n.t('convert.allComplete'));
            const resultMessage = results.map(r => window.i18n.t('messages.outputFile', { size: r.size, path: r.path })).join('\n');
            showResult(window.i18n.t('convert.allComplete') + '\n\n' + resultMessage);
            const convertBtn = document.getElementById('convert-btn');
            convertBtn.disabled = false;
            convertBtn.textContent = window.i18n.t('convert.startConvert');
        } else {
            setTimeout(checkAllTasks, checkInterval);
        }
    };
    
    checkAllTasks();
}

// 監控所有轉換任務（舊版本，保留以防需要）
async function monitorAllTasks(tasks, totalSizes) {
    const checkInterval = 1000; // 每秒檢查一次
    const completedTasks = new Set();
    const results = [];
    
    const checkAllTasks = async () => {
        let allCompleted = true;
        let totalProgress = 0;
        let currentMessage = '';
        
        for (const { taskId, size } of tasks) {
            if (completedTasks.has(taskId)) {
                totalProgress += 100;
                continue;
            }
            
            try {
                const response = await fetch(`${API_BASE}/task/${taskId}`);
                const task = await response.json();
                
                if (task.status === 'completed') {
                    completedTasks.add(taskId);
                    results.push({ size, path: task.output_path });
                    totalProgress += 100;
                } else if (task.status === 'failed') {
                    completedTasks.add(taskId);
                    
                    // 使用錯誤代碼翻譯或回退到錯誤訊息
                    let errorMessage;
                    if (task.error_code) {
                        errorMessage = translateErrorCode(task.error_code, task.conversion_step);
                    } else {
                        errorMessage = task.error || window.i18n.t('convert.conversionFailed');
                    }
                    
                    showError(window.i18n.t('messages.sizeComplete', { size: size }) + ' ' + window.i18n.t('convert.conversionFailed') + ': ' + errorMessage);
                    const convertBtn = document.getElementById('convert-btn');
                    convertBtn.disabled = false;
                    convertBtn.textContent = window.i18n.t('convert.startConvert');
                    return;
                } else if (task.status === 'running') {
                    allCompleted = false;
                    totalProgress += task.progress || 0;
                    if (task.message) {
                        currentMessage = window.i18n.t('messages.sizeComplete', { size: size }) + ': ' + task.message;
                    } else {
                        currentMessage = window.i18n.t('messages.converting');
                    }
                }
            } catch (error) {
                console.error('檢查任務狀態失敗:', error);
                allCompleted = false;
            }
        }
        
        const avgProgress = totalProgress / totalSizes;
        updateProgress(avgProgress, currentMessage || window.i18n.t('messages.progress', { completed: completedTasks.size, total: totalSizes }));
        
        if (allCompleted && completedTasks.size === tasks.length) {
            // 所有任務完成
            updateProgress(100, window.i18n.t('convert.allComplete'));
            const resultMessage = results.map(r => window.i18n.t('messages.outputFile', { size: r.size, path: r.path })).join('\n');
            showResult(window.i18n.t('convert.allComplete') + '\n\n' + resultMessage);
            const convertBtn = document.getElementById('convert-btn');
            convertBtn.disabled = false;
            convertBtn.textContent = window.i18n.t('convert.startConvert');
        } else {
            setTimeout(checkAllTasks, checkInterval);
        }
    };
    
    checkAllTasks();
}

// 更新進度條
function updateProgress(percent, message) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressFill.style.width = `${percent}%`;
    progressText.textContent = message;
}

// 顯示結果
function showResult(message) {
    document.getElementById('progress-container').style.display = 'none';
    document.getElementById('result-container').style.display = 'block';
    document.getElementById('result-message').textContent = message;
}

// 顯示錯誤
function showError(message) {
    document.getElementById('progress-container').style.display = 'none';
    document.getElementById('error-container').style.display = 'block';
    document.getElementById('error-message').textContent = message;
    
    // Re-enable convert button
    const convertBtn = document.getElementById('convert-btn');
    if (convertBtn) {
        convertBtn.disabled = false;
        convertBtn.textContent = window.i18n.t('convert.startConvert');
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 處理文件夾選擇
function handleFolderSelection(files, inputId, folderType) {
    const input = document.getElementById(inputId);
    
    // 處理空資料夾的情況
    if (!files || files.length === 0) {
        // 空資料夾：設置提示文字，讓用戶知道已選擇但需要輸入完整路徑
        const folderPath = `[已選擇空資料夾，請輸入完整路徑]`;
        input.value = folderPath;
        console.log(`Selected empty ${folderType} folder`);
        
        // 聚焦輸入框，讓用戶可以輸入路徑
        setTimeout(() => {
            input.focus();
            input.select();
        }, 100);
        return;
    }
    
    // 從第一個文件的 webkitRelativePath 獲取資料夾路徑
    let folderPath = '';
    if (files[0].webkitRelativePath) {
        // 獲取資料夾路徑（去掉文件名）
        const pathParts = files[0].webkitRelativePath.split('/');
        pathParts.pop(); // 移除文件名
        folderPath = pathParts.join('/');
        
        // 如果路徑為空，說明文件在根目錄，使用資料夾名稱
        if (!folderPath && pathParts.length > 0) {
            folderPath = pathParts[0];
        }
    }
    
    // 如果還是沒有路徑，使用資料夾名稱作為提示
    if (!folderPath) {
        // 嘗試從文件名推斷（如果文件在根目錄）
        const fileName = files[0].name;
        folderPath = `[已選擇資料夾，包含 ${files.length} 個文件]`;
    }
    
    // 顯示路徑（用戶可以手動編輯為完整路徑）
    input.value = folderPath;
    console.log(`Selected ${folderType} folder: ${folderPath} (${files.length} files)`);
    
    // 如果顯示的是提示文字，讓用戶知道可以編輯
    if (folderPath.includes('[')) {
        // 延遲一下再聚焦，讓用戶看到提示
        setTimeout(() => {
            input.focus();
            input.select();
        }, 100);
    }
}

// 開始訓練
async function startTraining() {
    const yoloVersion = document.getElementById('yolo-version').value;
    const modelSize = document.getElementById('model-size').value;
    const imagesFolder = document.getElementById('images-folder').value;
    const labelsFolder = document.getElementById('labels-folder').value;
    const outputDestination = document.getElementById('output-destination').value;
    const epochs = parseInt(document.getElementById('epochs').value) || 100;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 16;
    const imgsz = parseInt(document.getElementById('image-size').value) || 640;
    const resume = document.getElementById('resume-training').checked;
    
    // 優化選項
    const enableOptimization = document.getElementById('enable-optimization').checked;
    const minW = parseInt(document.getElementById('min-w').value) || 14;
    const minH = parseInt(document.getElementById('min-h').value) || 24;
    const maxInstances = parseInt(document.getElementById('max-instances').value) || 6;
    const valSplit = parseFloat(document.getElementById('val-split').value) || 0.15;
    const bgValRatio = parseFloat(document.getElementById('bg-val-ratio').value) || 0.08;
    const probeEpochs = parseInt(document.getElementById('probe-epochs').value) || 80;
    
    if (!imagesFolder) {
        alert(window.i18n.t('train.selectImagesFolder'));
        return;
    }
    
    if (!labelsFolder) {
        alert(window.i18n.t('train.selectLabelsFolder'));
        return;
    }
    
    if (!outputDestination) {
        alert(window.i18n.t('train.selectOutputDestination'));
        return;
    }
    
    // 隱藏之前的結果
    document.getElementById('train-result-container').style.display = 'none';
    document.getElementById('train-error-container').style.display = 'none';
    document.getElementById('train-progress-container').style.display = 'block';
    
    // 清空指標表格
    const metricsBody = document.getElementById('training-metrics-body');
    if (metricsBody) {
        metricsBody.innerHTML = '';
    }
    
    // 清空日誌
    const logOutput = document.getElementById('train-log-output');
    if (logOutput) {
        logOutput.textContent = '';
    }
    
    // 禁用訓練按鈕
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.textContent = window.i18n.t('train.startTraining') + '...';
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                yolo_version: yoloVersion,
                model_size: modelSize,
                images_folder: imagesFolder,
                labels_folder: labelsFolder,
                output_destination: outputDestination,
                epochs: epochs,
                batch_size: batchSize,
                imgsz: imgsz,
                resume: resume,
                enable_optimization: enableOptimization,
                min_w: minW,
                min_h: minH,
                max_instances: maxInstances,
                val_split: valSplit,
                bg_val_ratio: bgValRatio,
                probe_epochs: probeEpochs
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 監控訓練進度
            monitorTrainingTask(data.task_id);
        } else {
            showTrainingError('Training failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Training failed:', error);
        showTrainingError('Training failed: ' + error.message);
    }
}

// 更新訓練指標表格
function updateTrainingMetrics(metrics) {
    const tbody = document.getElementById('training-metrics-body');
    if (!tbody) {
        console.warn('Training metrics table body not found');
        return;
    }
    
    if (!metrics || metrics.length === 0) {
        // 即使沒有指標，也清空表格（顯示空狀態）
        tbody.innerHTML = '';
        return;
    }
    
    // 清空現有內容
    tbody.innerHTML = '';
    
    // 添加指標行
    metrics.forEach(metric => {
        try {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${metric.epoch || 'N/A'}</td>
                <td>${metric.gpu_mem || 'N/A'}</td>
                <td>${metric.box_loss !== undefined ? metric.box_loss.toFixed(4) : 'N/A'}</td>
                <td>${metric.cls_loss !== undefined ? metric.cls_loss.toFixed(4) : 'N/A'}</td>
                <td>${metric.dfl_loss !== undefined ? metric.dfl_loss.toFixed(4) : 'N/A'}</td>
                <td>${metric.instances !== undefined ? metric.instances : 'N/A'}</td>
                <td>${metric.size !== undefined ? metric.size : 'N/A'}</td>
            `;
            tbody.appendChild(row);
        } catch (error) {
            console.error('Error adding metric row:', error, metric);
        }
    });
    
    // 自動滾動到底部（顯示最新指標）
    const tableWrapper = document.querySelector('.metrics-table-wrapper');
    if (tableWrapper) {
        tableWrapper.scrollTop = tableWrapper.scrollHeight;
    }
}

// 監控訓練任務
async function monitorTrainingTask(taskId) {
    const checkInterval = 2000; // 每2秒檢查一次
    const logOutput = document.getElementById('train-log-output');
    
    const checkStatus = async () => {
        try {
            const response = await fetch(`${API_BASE}/train/${taskId}`);
            const task = await response.json();
            
            // 更新訓練指標表格
            if (task.metrics) {
                if (task.metrics.length > 0) {
                    console.log('Updating training metrics:', task.metrics.length, 'rows');
                }
                updateTrainingMetrics(task.metrics);
            } else {
                // 如果沒有指標數據，也調用一次以清空表格
                updateTrainingMetrics([]);
            }
            
            // 更新訓練日誌
            if (logOutput) {
                let logText = '';
                if (task.message) {
                    logText += task.message + '\n';
                }
                if (task.log) {
                    logText += task.log;
                }
                if (task.traceback) {
                    logText += '\n' + task.traceback;
                }
                if (logText) {
                    logOutput.textContent = logText;
                    // 自動滾動到底部
                    logOutput.scrollTop = logOutput.scrollHeight;
                }
            }
            
            if (task.status === 'completed') {
                updateTrainingProgress(100, window.i18n.t('train.trainingComplete'));
                // 最終更新指標
                if (task.metrics && task.metrics.length > 0) {
                    updateTrainingMetrics(task.metrics);
                }
                showTrainingResult(window.i18n.t('train.trainingComplete') + '\n' + window.i18n.t('messages.outputFile', { path: task.output_path || 'N/A' }));
            } else if (task.status === 'failed') {
                let errorMsg = window.i18n.t('train.trainingFailed') + ': ' + (task.error || 'Unknown error');
                if (task.traceback) {
                    errorMsg += '\n\n' + task.traceback;
                }
                showTrainingError(errorMsg);
            } else if (task.status === 'running') {
                updateTrainingProgress(task.progress || 0, task.message || window.i18n.t('messages.converting'));
                setTimeout(checkStatus, checkInterval);
            }
        } catch (error) {
            console.error('Check training status failed:', error);
            if (logOutput) {
                logOutput.textContent += '\n[錯誤] 無法檢查訓練狀態: ' + error.message;
            }
            showTrainingError('Unable to check training status');
        }
    };
    
    checkStatus();
}

// 更新訓練進度條
function updateTrainingProgress(percent, message) {
    const progressFill = document.getElementById('train-progress-fill');
    const progressText = document.getElementById('train-progress-text');
    
    if (progressFill) progressFill.style.width = `${percent}%`;
    if (progressText) progressText.textContent = message;
}

// 顯示訓練結果
function showTrainingResult(message) {
    document.getElementById('train-progress-container').style.display = 'none';
    document.getElementById('train-result-container').style.display = 'block';
    document.getElementById('train-result-message').textContent = message;
    
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.textContent = window.i18n.t('train.startTraining');
    }
}

// 顯示訓練錯誤
function showTrainingError(message) {
    document.getElementById('train-progress-container').style.display = 'none';
    document.getElementById('train-error-container').style.display = 'block';
    document.getElementById('train-error-message').textContent = message;
    
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.textContent = window.i18n.t('train.startTraining');
    }
}

// 初始化系統資訊頁面
function initSystemInfo() {
    const setGpuBtn = document.getElementById('set-gpu-btn');
    if (setGpuBtn) {
        setGpuBtn.addEventListener('click', setGpu);
    }
    
    // 當切換到系統分頁時自動刷新資訊
    document.querySelectorAll('.tab-btn[data-tab="system"]').forEach(btn => {
        btn.addEventListener('click', loadSystemInfo);
    });
    
    // 初始載入
    loadSystemInfo();
}

// 載入系統資訊
async function loadSystemInfo() {
    const gpuListContainer = document.getElementById('gpu-list-container');
    const gpuSelect = document.getElementById('gpu-select');
    if (!gpuListContainer || !gpuSelect) return;
    
    try {
        const response = await fetch(`${API_BASE}/system-info`);
        const data = await response.json();
        
        // 更新 CUDA 狀態
        const cudaStatus = document.getElementById('cuda-status');
        if (cudaStatus) {
            cudaStatus.textContent = data.cuda_available ? 'Available' : 'Unavailable';
            cudaStatus.className = `status-value badge ${data.cuda_available ? 'badge-success' : 'badge-error'}`;
        }
        const cudaVersion = document.getElementById('cuda-version');
        if (cudaVersion) cudaVersion.textContent = data.cuda_version || 'N/A';
        
        // 更新 TensorRT 狀態
        const trtStatus = document.getElementById('trt-status');
        if (trtStatus) {
            trtStatus.textContent = data.tensorrt_available ? 'Available' : 'Unavailable';
            trtStatus.className = `status-value badge ${data.tensorrt_available ? 'badge-success' : 'badge-error'}`;
        }
        const trtVersion = document.getElementById('trt-version');
        if (trtVersion) trtVersion.textContent = data.tensorrt_version || 'N/A';
        
        // 更新 GPU 列表
        if (data.cuda_available && data.gpus.length > 0) {
            let html = '<div class="gpu-grid">';
            gpuSelect.innerHTML = '';
            
            data.gpus.forEach(gpu => {
                const isActive = gpu.id === data.current_gpu;
                html += `
                    <div class="gpu-card ${isActive ? 'active' : ''}">
                        <div class="gpu-id">#${gpu.id}</div>
                        <div class="gpu-name">${gpu.name}</div>
                        <div class="gpu-mem">${gpu.memory.toFixed(1)} GB</div>
                        ${isActive ? '<div class="active-badge">Active</div>' : ''}
                    </div>
                `;
                
                const option = document.createElement('option');
                option.value = gpu.id;
                option.textContent = `GPU ${gpu.id}: ${gpu.name}`;
                if (isActive) option.selected = true;
                gpuSelect.appendChild(option);
            });
            html += '</div>';
            gpuListContainer.innerHTML = html;
        } else {
            gpuListContainer.innerHTML = `<div class="alert alert-warning">${window.i18n.t('system.noGpu')}</div>`;
            gpuSelect.innerHTML = `<option value="">${window.i18n.t('system.noGpu')}</option>`;
        }
        
    } catch (error) {
        console.error('Failed to load system info:', error);
        if (gpuListContainer) gpuListContainer.innerHTML = `<div class="alert alert-error">Failed to load system info</div>`;
    }
}

// 設置 GPU
async function setGpu() {
    const gpuId = parseInt(document.getElementById('gpu-select').value);
    const btn = document.getElementById('set-gpu-btn');
    
    if (isNaN(gpuId)) return;
    
    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = 'Applying...';
    
    try {
        const response = await fetch(`${API_BASE}/set-gpu`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gpu_id: gpuId })
        });
        
        const data = await response.json();
        if (data.success) {
            alert('GPU 設置成功！伺服器將使用新選定的 GPU。');
            loadSystemInfo();
        } else {
            alert('GPU 設置失敗: ' + data.error);
        }
    } catch (error) {
        console.error('Failed to set GPU:', error);
        alert('GPU 設置失敗: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}
