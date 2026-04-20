# AI2612 Final Project: Baseline DCGAN

本项目提供一个可直接运行的基础版 DCGAN，用于人脸图像生成，包含：

- `train.py`：训练 DCGAN 并保存 checkpoint / 样本图
- `generate.py`：从训练好的生成器随机采样
- `interpolate.py`：潜变量线性插值
- `evaluate.py`：使用 FID 做基础定量评估

## 1. 环境安装

建议 Python `3.10+`。

```bash
pip install -r requirements.txt
```

## 2. 数据集下载（CelebA / LFW）

### 方案 A：CelebA（推荐）

1. 安装 Kaggle CLI，并在本机配置 `~/.kaggle/kaggle.json`（账号 Token）。
2. 下载并解压：

```bash
kaggle datasets download -d jessicali9530/celeba-dataset -p data/raw
unzip data/raw/celeba-dataset.zip -d data/raw/celeba
```

3. 整理成 `ImageFolder` 结构（训练脚本需要）：

```bash
python scripts/prepare_imagefolder.py --src data/raw/celeba/img_align_celeba/img_align_celeba --dst data/celeba
```

整理后结构应为：

```text
data/celeba/
  faces/
    000001.jpg
    000002.jpg
    ...
```

### 方案 B：LFW（体量更小）

1. 官方下载页：https://vis-www.cs.umass.edu/lfw/
2. 下载 `lfw.tgz` 并解压到 `data/raw/lfw`
3. 整理成统一目录（可选）后训练：

```bash
python scripts/prepare_imagefolder.py --src data/raw/lfw --dst data/lfw
```

## 3. 训练

```bash
python train.py \
  --data-root data/celeba \
  --output-dir outputs \
  --epochs 20 \
  --batch-size 128 \
  --image-size 128
```

输出目录：

- `outputs/checkpoints/latest.pt`
- `outputs/samples/*.png`
- `outputs/training_log.json`

## 4. 随机生成人脸

```bash
python generate.py \
  --checkpoint outputs/checkpoints/latest.pt \
  --out-dir outputs/generated \
  --num-images 64
```

## 5. 潜变量插值

```bash
python interpolate.py \
  --checkpoint outputs/checkpoints/latest.pt \
  --out-dir outputs/interpolation \
  --steps 10
```

## 6. FID 评估

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints/latest.pt \
  --data-root data/celeba \
  --num-gen 5000 \
  --batch-size 64
```

> 说明：首次计算 FID 时，`torchmetrics` 会下载 Inception 权重。

## 7. 项目结构

```text
AI2612-finalProject/
  models/
    dcgan.py
  scripts/
    data.py
    prepare_imagefolder.py
    utils.py
  train.py
  generate.py
  interpolate.py
  evaluate.py
  requirements.txt
```
