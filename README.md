# Image Captioning Evaluation 

## Hardware & Software Requirements
### Minimum requirements
To run the benchmark, your system must meet the following baseline requirements:

- **GPU: ≥ 12 GB VRAM**

  - Required if you want to run CLIP‑Image‑Score.

  - The benchmark in general requires GPU acceleration for all metrics.

- **CUDA: 13.0 or above**

- **Python: 3.13.0 or above**

### Recommended Configuration
For smooth execution—especially for metrics involving image generation—we recommend:

- **GPU**: NVIDIA **RTX 3090** (24 GB VRAM) or higher
This significantly reduces runtime for compute‑heavy metrics such as CLIP‑Image‑Score.

### If Your GPU Has Less Than 12 GB VRAM
You can still use the benchmark, but:

Running **CLIP‑Image‑Score** will be extremely slow or may not fit in memory.

We strongly recommend using our pre‑generated image datasets, created from candidate captions from:

- Flickr8k

- Composite

- Polaris (test set)

These datasets can be downloaded [here]()

Using the pre‑generated images allows you to run the full evaluation pipeline even on lower‑VRAM GPUs.


## Dataset Setup

Download the image datasets and extract them into the `data/` directory.

Expected folder structure:

```text
Root/
├── data/
│   ├── flickr8k/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── composite/
│   │   ├── image1.jpg
│   │   └── ...
│   │
│   └── polaris/
│       ├── image1.jpg
│       └── ...
│
├── datasets/
├── metrics/
├── compute_all_metric.py
└── README.md
```

### Flickr8K

Download the Flickr8K image archive [here](
https://drive.google.com/file/d/1kBBhZKFLoUWqKXdtBfXxSFmvFD8DIaFz/view?usp=sharing)

Extract all images into:
`
data/flickr8k/
`

---

### Composite

Download the Composite dataset image archive [here](
https://drive.google.com/file/d/1gbi_7RjHrifV9EtPHcrzt_RFBcRe31T8/view?usp=sharing)

Extract all images into:
`
data/composite/
`

---

### Polaris

Download the Polaris dataset image archive [here](
https://drive.google.com/file/d/1pIQEVRWzGm5YiwuuZ3R07nqmKkbkE2aE/view?usp=sharing)

Extract all images into:
`
data/polaris/
`

---

> **Notes**
> - Ensure that image files are placed directly inside each dataset folder.
> - Do not create additional nested directories after extraction.
> - The benchmark expects image paths in this exact structure.
> - Missing or incorrectly placed images will cause dataset loading failures.

## Checkpoint Setup

Some metrics require pretrained checkpoint files before evaluation. Download the required files below and place them in the expected locations.

Expected structure:

```text
Root/
├── checkpoints/
│   ├── PAC_ViT-B-32.pth
│   ├── PAC_ViT-L-14.pth
│   ├── PAC++_clip_ViT-B-32.pth
│   ├── PAC++_clip_ViT-L-14.pth
│   ├── umic.pt
│   ├── faster_rcnn_from_caffe_attr_original.pkl
│   └── yuigawada/
│       └── reprod/
│           ├── reprod.ckpt
│           └── hparams.yaml
```

| Metric | Required checkpoint file(s) | Download URL | Expected location |
|--------|-----------------------------|--------------|------------------|
| PACScore | `PAC_ViT-B-32.pth`, `PAC_ViT-L-14.pth` | [PAC_ViT-B/32](https://drive.google.com/file/d/1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae/view?usp=drive_link), [PAC_ViT-L/14](https://drive.google.com/file/d/1G1DAGQf5fW2U3u7K3Dn-eCC6koMDyvsU/view?usp=drive_link) | `checkpoints/` |
| PACScore++ | `PAC++_clip_ViT-B-32.pth`, `PAC++_clip_ViT-L-14.pth` | [PAC++ ViT-B/32](https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-B-32.pth), [PAC++ ViT-L/14](https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth) | `checkpoints/` |
| UMIC | `umic.pt` | [umic.pt](https://archive.org/download/umic_data/umic.pt) | `checkpoints/` |
| UMIC (feature extraction) | `faster_rcnn_from_caffe_attr_original.pkl` | [Detectron2 Faster R-CNN checkpoint](http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl) | `checkpoints/` |
| Polos | `reprod.ckpt`, `hparams.yaml` | [reprod.zip](https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip) | `checkpoints/yuigawada/reprod/` |

> **Notes**
> - Metrics not listed above either do not require manual checkpoint setup or automatically download pretrained weights during first use.
> - For **Polos**, download `reprod.zip`, extract it, and preserve the original folder structure under:
`
checkpoints/yuigawada/reprod/
`
> - Both `reprod.ckpt` and `hparams.yaml` are required for Polos.
> - CLIP backbone weights used by CLIP-based metrics are automatically downloaded by the corresponding libraries.
> - Ensure filenames and folder structure match exactly, as some metrics rely on hardcoded checkpoint paths.

## Environment Setup

Run the setup script for your operating system.

### Windows

```bash
setup.bat
```

### Linux / macOS

```bash
chmod +x setup.sh
./setup.sh
```

## Running Evaluations

The benchmark is executed through `compute_all_metric.py`.

General syntax:

```bash
python compute_all_metric.py \
    --dataset <dataset_name> \
    --metrics_name <metric_name> \
    --clip_model <clip_backbone>
```

### Arguments

| Argument              | Description | Supported values |
|-----------------------|-------------|------------------|
| `--dataset`           | Dataset to evaluate | `flickr8k`, `composite`, `polaris` |
| `--metrics_name`      | Metric to compute | `standard`, `bert-score`, `bert-score++`, `blip2-score`, `umic-score`, `clip-score`, `pac-score`, `pac-score++`, `mid-score`, `polos` |
| `--clip_model`        | CLIP backbone used by CLIP-based metrics | `ViT-B/32`, `ViT-L/14` |

> **Note:** The `--clip_model` argument is only used for the CLIPScore family metrics:
> - `clip-score`
> - `pac-score`
> - `pac-score++`
>
> For all other metrics, this argument is ignored and can be set to any valid value.
---

### Example Commands

#### Traditional Metrics

Compute all standard rule-based metrics (BLEU, ROUGE, METEOR, CIDEr, SPICE):

```bash
python compute_all_metric.py \
    --dataset polaris \
    --metrics_name standard
```

---

#### BERT-based Metrics

```bash
python compute_all_metric.py \
    --dataset polaris \
    --metrics_name bert-score
```

```bash
python compute_all_metric.py \
    --dataset polaris \
    --metrics_name bert-score++
```

---

#### CLIP-based Metrics

CLIPScore:

```bash
python compute_all_metric.py \
    --dataset flickr8k \
    --metrics_name clip-score \
    --clip_model "ViT-B/32"
```

PACScore:

```bash
python compute_all_metric.py \
    --dataset flickr8k \
    --metrics_name pac-score \
    --clip_model "ViT-B/32"
```

PACScore++:

```bash
python compute_all_metric.py \
    --dataset flickr8k \
    --metrics_name pac-score++ \
    --clip_model "ViT-B/32"
```

---

#### Heavy Multimodal Metrics

BLIP2Score:

```bash
python compute_all_metric.py \
    --dataset flickr8k \
    --metrics_name blip2-score
```

UMIC:

```bash
python compute_all_metric.py \
    --dataset flickr8k \
    --metrics_name umic-score
```

MID:

```bash
python compute_all_metric.py \
    --dataset composite \
    --metrics_name mid-score
```

Polos:

```bash
python compute_all_metric.py \
    --dataset polaris \
    --metrics_name polos
```

---

### Output

Metric scores are saved as JSONL files under:

```text
asset/
├── flickr8k/
├── composite/
└── polaris/
```

Example:

```text
asset/flickr8k/flickrExpert_scores.jsonl
```

If a metric already exists in the output file, its scores will be updated instead of duplicated.

# How to Contribute

1. Fork this repository and clone it locally.
2. Create a new branch for your changes: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Description of the changes'`.
4. Push to your fork: `git push origin feature-name`.
5. Open a pull request on the original repository by providing a description of your changes.
