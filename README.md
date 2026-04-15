# what-are-we-really-measuring

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2302.09462)
[![Paper](https://img.shields.io/badge/Elsevier-CIBM-blue)]([https://doi.org/10.1016/j.compbiomed.2023.106791](https://doi.org/10.1016/j.neucom.2026.133679))

This repository contains the official implementation of our paper ["What are we really measuring? Rethinking dataset bias in web-scale natural image collections via unsupervised semantic clustering"](https://www.sciencedirect.com/science/article/abs/pii/S0925231226010763)   (Neurocomputing, 2026).
## Overview

In this work, we re-evaluate the prevailing method for measuring dataset bias, which is training a classifier to distinguish between datasets. We demonstrate that high classification accuracy, widely interpreted as evidence of meaningful semantic differences, is often driven by low level resolution artifacts rather than true semantic content.

Through controlled experiments, we show that models exploit these artifacts even when trained on procedurally generated images with no semantic information. Standard image augmentations fail to suppress these artifacts. To address this fundamental flaw, we propose an unsupervised framework that clusters semantically rich features from DINOv2, bypassing supervised classification entirely.

Our approach reveals that when analysis is constrained to semantics, the high separability reported by supervised methods largely vanishes, dropping to near chance levels. These findings offer a modern re evaluation of how dataset separability should be interpreted. They suggest that unsupervised clustering provides a more reliable measure of semantic bias in web scale natural image collections.

## Key Findings

**Resolution fingerprints:** Web scale image datasets (YFCC, CC, DataComp, WIT, LAION) exhibit distinct native resolution distributions. These act as dataset specific "fingerprints" that persist even after resizing to a common dimension.

**Artifact exploitation:** Deep models learn to exploit resolution induced interpolation artifacts. They achieve strong dataset classification performance (84-87% accuracy) even on images with no semantic content.

**Supervised magnification:** The supervised objective forces discrimination at any cost. It magnifies weak artifact signals present in the feature space into strong classification cues.

**Unsupervised resilience:** Unsupervised clustering has no access to dataset labels and cannot be instructed to discriminate. It only groups images based on the natural structure of the feature space. This makes it less susceptible to non semantic artifacts.

**Minimal semantic bias:** When artifacts are controlled for, the apparent dataset separability largely vanishes. Our unsupervised method achieves near chance accuracy (46.95% on YCD), which is comparable to the human performance (45.4%) reported in "[A Decade's Battle on Dataset Bias: Are We There Yet?](https://arxiv.org/abs/2403.08632)" (Liu and He, ICLR 2025).

**Methodological re evaluation:** The traditional interpretation of high dataset classification accuracy as evidence of semantic bias should be reconsidered. Unsupervised semantic clustering offers a more principled alternative for measuring true semantic differences between web scale natural image collections.

## Contact

For any inquiries or questions regarding the paper or code, please feel free to contact us directly via email:

- Amir Hossein Saleknia: salekniaamir@gmail.com
- Mohammad Sabokrou: mohammad.sabokrou@oist.jp
