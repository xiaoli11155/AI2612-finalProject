# Task: GAN-based Human Face Image Generation

## Overview
Build a human face image generation project based on GANs. The project should reproduce a complete training and evaluation pipeline for generating face images, starting from a baseline GAN model and optionally extending to improved variants.

The main purpose is to:
- understand the adversarial training mechanism of GANs,
- implement a baseline face image generator,
- train it on a public face dataset,
- generate high-quality human face images,
- evaluate generation quality with standard metrics,
- and analyze interpolation and model improvement results.

---

## Objectives

### Core objectives
1. Implement a baseline GAN model for face image generation.
2. Train the model on a selected public face dataset.
3. Generate realistic human face images after training.
4. Perform linear interpolation between two latent codes and visualize the generated image sequence.
5. Evaluate image quality using at least one quantitative metric:
   - FID (Fréchet Inception Distance), or
   - IS (Inception Score).

### Bonus objectives
1. Compare the baseline GAN with an improved GAN variant, such as:
   - StyleGAN,
   - CycleGAN,
   - or another reasonable GAN improvement.
2. Explore methods to reduce mode collapse and improve training stability.
3. Provide analysis of performance differences between the baseline and improved models.

---

## Suggested models

### Baseline
- **DCGAN**
  - Use as the minimum required baseline.
  - Suitable for standard image generation experiments.

### Optional improved models
- **StyleGAN**
  - For higher-quality and more diverse face generation.
- **CycleGAN**
  - If exploring style transfer or domain-to-domain face generation.
- Other reasonable GAN improvements are acceptable if clearly explained.

---

## Suggested datasets

Choose one of the following public face datasets:

### Option 1: CelebA
- More than 200,000 celebrity face images
- Approximate size: 1.3 GB
- Recommended for the main experiment

### Option 2: LFW (Labeled Faces in the Wild)
- Around 13,000 face images
- Approximate size: 118 MB
- Suitable for smaller-scale experiments or quick testing

---

## Functional requirements

The project should include the following capabilities:

1. **Data loading and preprocessing**
   - Download or load the selected dataset
   - Resize and normalize images properly
   - Support batching for training

2. **Model implementation**
   - Implement the generator
   - Implement the discriminator
   - Use adversarial training for optimization

3. **Training pipeline**
   - Train the GAN on the selected dataset
   - Save checkpoints regularly
   - Log training losses and sample outputs during training

4. **Image generation**
   - Generate random face images from latent vectors
   - Save generated samples to disk

5. **Latent space interpolation**
   - Select two latent vectors
   - Perform linear interpolation
   - Generate and save the interpolation sequence

6. **Evaluation**
   - Compute FID or IS on generated images
   - Report the evaluation result clearly

7. **Result visualization**
   - Save training curves if applicable
   - Save generated image grids
   - Save interpolation results

---

## Deliverables

The final project should contain:

### 1. Source code
A complete runnable project, with a clear directory structure.

Suggested structure:
```text
project/
├── data/
├── models/
├── outputs/
│   ├── samples/
│   ├── interpolation/
│   └── checkpoints/
├── scripts/
├── train.py
├── generate.py
├── evaluate.py
├── interpolate.py
├── requirements.txt
└── README.md