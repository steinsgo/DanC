# Ancient Character OCR

> 从斑驳拓片中定位笔画，从千年字形中还原字符。

面向古文字拓片场景的端到端 OCR 比赛项目。系统以完整拓片图像为输入，
在复杂底纹、残损笔画和密集排布中完成字符检测与单字识别，
最终生成比赛要求的 `prediction.json`。

## 方案概述

当前提交方案采用紧凑的两阶段识别链路：

| 阶段 | 模型 | 任务 |
| --- | --- | --- |
| 检测 | YOLO11m | 在整幅拓片中扫描字符候选区域，输出单字边界框 |
| 识别 | ResNet50 | 对检测框裁剪出的古文字字形完成类别判别 |

端到端推理流程如下：

1. 读取平台挂载的完整拓片图像。
2. 使用 YOLO11m 在整图尺度上检测字符位置，输出 `[x, y, w, h]`。
3. 将检测框裁剪为单字图像，送入 ResNet50 古文字分类器。
4. 汇总字符框与识别文本，写入 `/saisresult/prediction.json`。

## 比赛输入输出

容器启动后直接对接评测平台约定路径：

| 类型 | 路径 |
| --- | --- |
| 输入图片 | `/saisdata/13/eval/images/` |
| 输出结果 | `/saisresult/prediction.json` |

输出文件顶层为图片 ID 到预测列表的映射：

```json
{
  "image_id": [
    {
      "bbox": [843, 2087, 93, 89],
      "text": "天"
    }
  ]
}
```

其中：

- `bbox` 使用像素坐标 `[x, y, w, h]`；
- `text` 为单字识别结果；
- 输出 JSON 使用 UTF-8 编码。

## 提交镜像

提交镜像遵循即启即推理的执行方式。容器入口由 `run.sh` 拉起推理脚本：

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 构建比赛提交镜像 |
| `run.sh` | 容器启动入口 |
| `scripts/run_inference.py` | 端到端推理实现 |

镜像内需要包含以下模型文件：

```text
/app/models/
|-- best_det.pt
|-- best_rec.pt
`-- char_dict.json
```

构建镜像前应确认 `models/` 中已放入训练好的检测权重、识别权重和字符映射文件，
使镜像在评测容器中无需额外配置即可完成推理。

## 环境依赖

提交镜像基于 PyTorch CUDA runtime，核心依赖如下：

| 依赖 | 版本或约束 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.2.0 |
| torchvision | 0.17.0 |
| CUDA | 12.1 |
| ultralytics | `>=8.1.0` |
| Pillow | `>=10.0.0` |
| NumPy | `<2` |
| opencv-python-headless | `>=4.8.0` |

## 训练流程

### 检测器

检测器以完整拓片为观察尺度，学习在噪声纹理、残缺字形和多行排布中寻找字符区域。
跨域训练数据的 XML 标注会先转换为 YOLO 单类别检测标签，再用于字符检测训练。

相关脚本：

- `scripts/parse_xml_to_yolo.py`：将 XML 标注转换为 YOLO 数据集；
- `scripts/train_yolo.py`：训练 YOLO11m 检测器。

### 识别器

识别器聚焦单字形态本身。训练标注框会被裁剪为单字图像，样本按字符类别组织为
`ImageFolder` 目录结构，再由 ResNet50 学习不同古文字字形之间的判别边界。

相关脚本：

- `scripts/build_char_dict.py`：构建字符映射；
- `scripts/crop_train_chars.py`：从拓片图像中裁剪单字训练样本；
- `scripts/train_recognizer.py`：训练 ResNet50 识别器；
- `scripts/train_all.sh`：串联训练与 sanity check 的脚本。

当前识别训练脚本支持通过 `--pretrained` 加载 ResNet50 预训练权重。

## 项目结构

```text
.
|-- Dockerfile
|-- run.sh
|-- requirements.txt
|-- models/
|   |-- char_dict.json
|   |-- best_det.pt
|   `-- best_rec.pt
`-- scripts/
    |-- run_inference.py
    |-- parse_xml_to_yolo.py
    |-- train_yolo.py
    |-- build_char_dict.py
    |-- crop_train_chars.py
    |-- train_recognizer.py
    `-- train_all.sh
```

## 数据说明

当前提交推理链路依赖比赛训练数据生成的检测器、识别器和字符映射文件。
仓库中的训练辅助脚本同时保留了预训练和外部数据扩展入口，便于继续探索
跨域迁移、长尾字符覆盖和古文字形态建模；实际提交应以赛事允许的数据范围
和最终训练记录为准。

## 致谢

感谢赛事组织方将古文字识别这一兼具历史厚度与工程挑战的任务带入评测场景，
并提供任务定义、评测平台与数据规范。

感谢复旦大学出土文献与古文字研究中心为本赛题相关拓片数据提供支持，
让模型能够面对真实古文字材料中的复杂纹理、残损与字形差异。

感谢 HUST-OBC 等开放数据与古文字智能识别研究工作提供参考，
推动古文字数字化从图像存档进一步走向结构化理解。

感谢 PyTorch、torchvision、Ultralytics YOLO、Pillow 与 OpenCV 等开源项目。
这些工具构成了本项目从数据处理、模型训练到容器推理的工程底座。
