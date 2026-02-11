# PuzzleBot — Personalized Chess Puzzle Generator

## Overview

PuzzleBot is an end-to-end deep learning system for generating personalized chess training puzzles. The system analyzes user games, detects blunders using Stockfish evaluations, classifies the underlying tactical motif using convolutional neural networks, and surfaces targeted puzzles for skill improvement.

The project was trained on **3.08 million labeled chess positions** and evaluated across multiple architectures, achieving ~50% test accuracy across 13 tactical motif classes, with a **~7% improvement using ResNet over a CNN baseline**.

---

## System Pipeline

1. **Game Analysis**
   - User games are parsed and evaluated with Stockfish.
   - Blunders are identified based on evaluation swings.

2. **Position Extraction**
   - Blunder positions are extracted and encoded as model inputs.

3. **Motif Classification**
   - A CNN or ResNet classifies the position into one of 13 tactical motifs.
   - These motifs guide targeted puzzle recommendations.

4. **Personalized Puzzle Generation**
   - Users receive puzzles aligned with their tactical weaknesses.

---

## Model Architecture

Two primary architectures were implemented and compared:

### CNN Baseline
- Convolutional feature extraction
- Fully connected classifier
- ~50% test accuracy across 13 classes

### ResNet Variant
- Residual blocks for improved gradient flow
- Deeper representational capacity
- ~7% accuracy improvement over CNN baseline

---

## Training Details

- **Dataset Size:** 3.08M positions
- **Classes:** 13 tactical motifs
- **Class Imbalance Handling:** Focal loss + weighted sampling
- **Optimization:** AdamW
- **Scheduling:** Cosine annealing
- **Augmentation:** Mixup
- **Framework:** PyTorch

Training was performed in a Google Colab environment due to compute requirements.

---

## Results

- CNN baseline: ~50% test accuracy  
- ResNet: +7% relative improvement  
- Demonstrated stable convergence under significant class imbalance  
- Successfully integrated into a GUI-driven puzzle generation workflow  

Given the subtle spatial differences between tactical motifs and heavy imbalance across 13 classes, results demonstrate meaningful signal capture rather than trivial classification.

A detailed technical report including experimental analysis is available in the `/docs` directory.

---

## My Contributions

This was a team project. My primary contributions included:

- Designing and implementing the CNN and ResNet model architectures
- Building the full PyTorch training pipeline
- Implementing class imbalance handling (focal loss, weighted sampling)
- Integrating Stockfish-based blunder detection logic
- Designing motif classification logic
- Running experiments and evaluating model performance
- Contributing to puzzle generation workflow integration

Teammates contributed to dataset preprocessing and repository refactoring.

---

## Limitations & Future Work

- Heavy class imbalance remains challenging
- Some tactical motifs exhibit subtle spatial differences
- Potential improvements:
  - Attention mechanisms
  - Hybrid CNN-attention architectures
  - Temporal modeling of move sequences
  - Curriculum learning strategies

---

## Tech Stack

- Python
- PyTorch
- CNN / ResNet
- Stockfish
- NumPy / Pandas
- Google Colab

---

## Project Report

The full project report, including methodology and experimental results, is available under PuzzleBotReport.pdf.
