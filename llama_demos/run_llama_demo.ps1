# Llama 水印Demo快速启动脚本
# 支持多种Llama模型，默认使用 Llama 2 7B
# 用于PowerShell

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "Llama 水印Demo - 快速启动" -ForegroundColor Cyan
Write-Host "默认模型: Llama 2 7B" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python
Write-Host "检查Python环境..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ 未找到Python，请先安装Python 3.8+" -ForegroundColor Red
    exit 1
}

# 模型选择
Write-Host ""
Write-Host "可用的模型:" -ForegroundColor Yellow
Write-Host "1. Llama-2-7b-hf (默认，推荐)" -ForegroundColor White
Write-Host "2. Llama-2-13b-hf (需要更多显存)" -ForegroundColor White
Write-Host "3. Llama-2-7b-chat-hf (对话优化版本)" -ForegroundColor White
Write-Host "4. Llama-3.2-1B (需要权限)" -ForegroundColor White
Write-Host "5. Llama-3.2-3B (需要权限)" -ForegroundColor White
Write-Host "6. 自定义模型路径" -ForegroundColor White
Write-Host ""

$modelChoice = Read-Host "选择模型 (1-6，直接回车使用默认)"

$modelName = "meta-llama/Llama-2-7b-hf"
switch ($modelChoice) {
    "2" { $modelName = "meta-llama/Llama-2-13b-hf" }
    "3" { $modelName = "meta-llama/Llama-2-7b-chat-hf" }
    "4" { $modelName = "meta-llama/Llama-3.2-1B" }
    "5" { $modelName = "meta-llama/Llama-3.2-3B" }
    "6" { 
        $modelName = Read-Host "请输入模型路径或名称"
    }
    default { 
        Write-Host "使用默认模型: Llama-2-7b-hf" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "已选择模型: $modelName" -ForegroundColor Green
Write-Host ""
# 显示菜单
Write-Host "请选择要运行的Demo:" -ForegroundColor Yellow
Write-Host "1. 简单示例 (llama_simple_example.py) - 推荐新手" -ForegroundColor White
Write-Host "2. 完整演示 (llama_watermark_demo.py) - 多个测试案例" -ForegroundColor White
Write-Host "3. 交互式界面 (llama_interactive_demo.py) - 交互式使用" -ForegroundColor White
Write-Host "4. 批量测试 (llama_batch_test.py) - 参数对比测试" -ForegroundColor White
Write-Host "5. 安装依赖" -ForegroundColor White
Write-Host "6. 查看使用文档" -ForegroundColor White
Write-Host "Q. 退出" -ForegroundColor White
Write-Host ""

$choice = Read-Host "请输入选择 (1-6/Q)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "启动简单示例..." -ForegroundColor Green
        Write-Host ""
        python llama_simple_example.py $modelName
    }
    "2" {
        Write-Host ""
        Write-Host "启动完整演示..." -ForegroundColor Green
        Write-Host ""
        python llama_watermark_demo.py $modelName
    }
    "3" {
        Write-Host ""
        Write-Host "启动交互式界面..." -ForegroundColor Green
        Write-Host "提示: 使用 Ctrl+C 可以随时退出" -ForegroundColor Yellow
        Write-Host ""
        python llama_interactive_demo.py --model_name $modelName
    }
    "4" {
        Write-Host ""
        Write-Host "启动批量测试..." -ForegroundColor Green
        Write-Host "警告: 这将需要较长时间运行" -ForegroundColor Yellow
        Write-Host ""
        $confirm = Read-Host "确认继续? (y/N)"
        if ($confirm -eq "y" -or $confirm -eq "Y") {
            python llama_batch_test.py $modelName
        } else {
            Write-Host "已取消" -ForegroundColor Yellow
        }
    }
    "5" {
        Write-Host ""
        Write-Host "安装依赖包..." -ForegroundColor Green
        Write-Host ""
        Write-Host "注意: PyTorch安装需要根据CUDA版本选择" -ForegroundColor Yellow
        Write-Host "1. CUDA 11.8" -ForegroundColor White
        Write-Host "2. CUDA 12.1" -ForegroundColor White
        Write-Host "3. CPU版本" -ForegroundColor White
        Write-Host ""
        $cuda = Read-Host "请选择 (1-3)"
        
        Write-Host ""
        switch ($cuda) {
            "1" {
                Write-Host "安装CUDA 11.8版本..." -ForegroundColor Green
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            }
            "2" {
                Write-Host "安装CUDA 12.1版本..." -ForegroundColor Green
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            }
            "3" {
                Write-Host "安装CPU版本..." -ForegroundColor Green
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            }
            default {
                Write-Host "无效选择" -ForegroundColor Red
                exit 1
            }
        }
        
        Write-Host ""
        Write-Host "安装其他依赖..." -ForegroundColor Green
        pip install -r requirements_llama.txt
        
        Write-Host ""
        Write-Host "✓ 依赖安装完成!" -ForegroundColor Green
    }
    "6" {
        Write-Host ""
        Write-Host "打开使用文档..." -ForegroundColor Green
        if (Test-Path "LLAMA_DEMO_README.md") {
            notepad LLAMA_DEMO_README.md
        } else {
            Write-Host "✗ 找不到文档文件" -ForegroundColor Red
        }
    }
    {$_ -eq "Q" -or $_ -eq "q"} {
        Write-Host ""
        Write-Host "退出程序" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host ""
        Write-Host "✗ 无效的选择" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "完成" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
