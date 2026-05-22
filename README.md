# Ancient Character OCR

> Detect the glyph. Recover the character. Preserve the page.

`Ancient Character OCR` is an end-to-end competition system for ancient
character recognition on full-page rubbing images. It turns a page-level image
with dense texture, damaged strokes, and highly variable glyph forms into a
structured set of character locations and transcriptions.

The project is built around a practical two-stage pipeline:

```text
full rubbing image
        |
        v
YOLO11m character detector
        |
        v
character crops
        |
        v
ResNet50 glyph recognizer
        |
        v
prediction.json
```

## Highlights

- Page-level detection and single-character recognition in one inference path.
- Competition-ready JSON output with character bounding boxes and text labels.
- YOLO-format detection data conversion from XML page annotations.
- Character-crop classification workflow for ancient glyph recognition.
- Docker entrypoint aligned with the evaluation platform mount contract.

## System Design

| Stage | Model | Responsibility |
| --- | --- | --- |
| Detection | YOLO11m | Locate every visible character candidate on the full rubbing page |
| Recognition | ResNet50 | Classify each cropped glyph into the character vocabulary |
| Packaging | Docker + shell entrypoint | Run evaluation inference without manual intervention |

The detector treats character localization as a single-class object detection
task. The recognizer then operates on glyph crops instead of entire text lines,
which keeps the inference path aligned with the annotation format and the final
evaluation target.

## Inference Contract

The evaluation container reads images from the platform mount and writes one
result file:

| Resource | Path |
| --- | --- |
| Evaluation images | `/saisdata/13/eval/images/` |
| Prediction file | `/saisresult/prediction.json` |

Each prediction entry contains one bounding box and one recognized character:

```json
{
  "ZHJWD060612-000001-GUJINGJUYING195": [
    {
      "bbox": [843, 2087, 93, 89],
      "text": "天"
    },
    {
      "bbox": [808, 2147, 97, 96],
      "text": "王"
    }
  ]
}
```

Output requirements:

- `bbox` uses integer pixel coordinates in `[x, y, w, h]` format.
- `text` is the recognized single-character transcription.
- The output file is UTF-8 encoded JSON.

## Submission Runtime

The Docker image starts [`run.sh`](run.sh), which calls the inference pipeline
in [`scripts/run_inference.py`](scripts/run_inference.py).

The runtime image expects these model artifacts:

```text
/app/models/
|-- best_det.pt
|-- best_rec.pt
`-- char_dict.json
```

Artifact roles:

| Artifact | Role |
| --- | --- |
| `best_det.pt` | Trained YOLO11m character detector |
| `best_rec.pt` | Trained ResNet50 character recognizer |
| `char_dict.json` | Mapping between classifier labels and character text |

Before building the submission image, place the trained detector, recognizer,
and character dictionary in `models/`.

## Training Workflow

### Detection

The detection workflow converts page-level XML annotations into YOLO labels.
Rectangle annotations are used directly as bounding boxes; polygon annotations
are reduced to their enclosing boxes for character detection.

```text
PNG + XML annotations
        |
        v
scripts/parse_xml_to_yolo.py
        |
        v
YOLO detection dataset
        |
        v
scripts/train_yolo.py
```

### Recognition

The recognition workflow builds a character vocabulary, crops annotated glyphs
from training pages, and trains a classifier over the resulting image folders.

```text
training annotations
        |
        v
char_dict.json + cropped glyph dataset
        |
        v
scripts/train_recognizer.py
```

Relevant scripts:

| Script | Purpose |
| --- | --- |
| `scripts/build_char_dict.py` | Build character mappings for recognition |
| `scripts/crop_train_chars.py` | Crop labeled single-character images |
| `scripts/train_recognizer.py` | Train the ResNet recognizer |
| `scripts/train_all.sh` | Chain training steps and sanity inference |

The recognizer training script supports optional ResNet50 pretraining through
`--pretrained`.

## Repository Layout

```text
.
|-- Dockerfile
|-- run.sh
|-- requirements.txt
|-- models/
|   |-- char_dict.json
|   |-- best_det.pt
|   `-- best_rec.pt
|-- sanity_test/
`-- scripts/
    |-- run_inference.py
    |-- parse_xml_to_yolo.py
    |-- train_yolo.py
    |-- build_char_dict.py
    |-- crop_train_chars.py
    |-- train_recognizer.py
    `-- train_all.sh
```

## Documentation Map

| Document | Scope |
| --- | --- |
| [`README.md`](README.md) | Architecture, runtime contract, training workflow, repository guide |
| [`赛题介绍.md`](赛题介绍.md) | Task background, data overview, metrics, baseline notes |
| [`2026 初赛代码提交规范.md`](2026%20初赛代码提交规范.md) | Docker submission contract and prediction format |
| [`镜像提交平台页面分析（结构化文档）.md`](镜像提交平台页面分析（结构化文档）.md) | Platform page notes and submission checklist context |
| [`YOLO_DEPENDENCY_SOLUTION.md`](YOLO_DEPENDENCY_SOLUTION.md) | Notes on YOLO dependency troubleshooting |

## Environment

The submission image is based on a PyTorch CUDA runtime. Core dependencies are:

| Component | Version or constraint |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.2.0 |
| torchvision | 0.17.0 |
| CUDA | 12.1 |
| ultralytics | `>=8.1.0` |
| Pillow | `>=10.0.0` |
| NumPy | `<2` |
| opencv-python-headless | `>=4.8.0` |

## Data Note

The submission inference path consumes trained artifacts produced from the
competition workflow. Training utilities keep room for continued experiments
with pretraining, cross-domain transfer, and long-tail character coverage.
Final training records and submitted artifacts should remain consistent with
the data policy of the competition.

## Acknowledgements

This project is built for a task where cultural heritage material meets modern
vision engineering.

Thanks to the competition organizers for defining the benchmark, evaluation
contract, and platform workflow for ancient character OCR.

Thanks to the Center for Research on Chinese Excavated Classics and
Paleography at Fudan University for the rubbing-data support associated with
the task.

Thanks to HUST-OBC and the broader ancient-character recognition community for
the open research context that continues to advance paleographic digitization.

Thanks to PyTorch, torchvision, Ultralytics YOLO, Pillow, and OpenCV for the
open-source foundation behind the data pipeline, model training, inference
runtime, and container submission path.
