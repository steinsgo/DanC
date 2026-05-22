# Ancient Character OCR — 端到端古文字识别系统

## 方案概述

两阶段端到端 OCR 系统，从完整拓片图像中检测并识别古文字。

| 阶段 | 模型 | 说明 |
|------|------|------|
| 检测 | YOLO11m | 单类别字符检测，输入 1280×1280 |
| 识别 | ResNet50 | 4114 类古文字分类，ImageNet 预训练 + 微调 |

## 环境依赖

| 依赖 | 版本 |
|------|------|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.1 |
| cuDNN | 8 |
| PyTorch | 2.2.0 |
| torchvision | 0.17.0 |
| ultralytics | ≥8.1.0 |
| opencv-python-headless | ≥4.8.0 |
| Pillow | ≥10.0.0 |
| NumPy | <2 |

## 模型权重位置（镜像内）

```
/app/models/
├── best_det.pt      # YOLO11m 检测模型
├── best_rec.pt      # ResNet50 识别模型
└── char_dict.json   # 字符映射字典
```

## 推理入口

- 启动脚本：`/app/run.sh`
- 推理代码：`/app/src/run_inference.py`
- 输入目录：`/saisdata/13/eval/images/`（平台挂载）
- 输出文件：`/saisresult/prediction.json`

## 推理流程

1. YOLO11m 检测图像中所有字符位置，输出 `[x, y, w, h]` 边界框
2. 裁剪每个字符区域，Resize 到 128×128
3. ResNet50 分类器识别字符内容
4. 汇总为 `{image_id: [{bbox, text}, ...]}` 格式输出

## 训练方式

### 检测器训练
- 数据：跨域训练数据 6000 张拓片图 + XML 标注，转换为 YOLO 格式
- 模型：YOLO11m，单 GPU（V100-32GB）
- 参数：epochs=150, imgsz=1280, batch=8, AdamW lr=1e-3
- 增强：mosaic=0.5, degrees=5, 禁用翻转（字符方向固定）

### 识别器训练
- 数据：从训练图裁剪单字图像，按字符类别组织 ImageFolder 结构
- 模型：ResNet50，ImageNet 预训练权重微调
- 参数：epochs=100, img_size=128, batch=128, AdamW lr=2e-4
- 正则化：Dropout(0.3), weight_decay=0.05, label_smoothing=0.2
- 增强：RandomPerspective, RandomErasing, ColorJitter, RandomAffine
- 多卡：torchrun DDP（2× V100-32GB）

## 外部数据使用

- ImageNet 预训练 ResNet50 权重（torchvision 官方）
- 未使用甲骨文预训练数据、PDF 文献数据等其他外部数据

## 项目结构

```
├── Dockerfile              # Docker 镜像构建文件
├── run.sh                  # 容器入口脚本
├── requirements.txt        # Python 依赖
├── models/                 # 模型权重（构建时 COPY 进镜像）
│   ├── best_det.pt
│   ├── best_rec.pt
│   └── char_dict.json
└── scripts/
    ├── run_inference.py    # 推理脚本
    ├── train_yolo.py       # 检测器训练
    ├── train_recognizer.py # 识别器训练
    ├── crop_train_chars.py # 字符裁剪
    └── train_all.sh        # 全流程训练脚本
```

## 随机性控制

训练阶段使用 `seed=42` 固定随机种子。推理阶段无随机操作，结果完全确定。