# MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](https://arxiv.org)
[![MomaGraph-Bench](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/cheryyunl/MomaGraph-Bench)
[![MomaGraph-R1](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/cheryyunl/MomaGraph-R1)

📌 This is the official implementation and benchmark evaluation repository of **MomaGraph**, focusing on evaluating the model's capability in **embodied task planning** by leveraging **state-aware unified scene graphs**.

## Overview

We provide two evaluation modes to assess the effectiveness of the generated scene graphs on **MomaGraph-Bench**:

1.  **Graph-then-Plan (`eval.py`)**: Requires the model to first generate a structured Scene Graph (JSON format) as a Chain-of-Thought (CoT) before answering the multiple-choice question. This validates whether the model can ground its reasoning in the scene structure.
2.  **Direct QA Evaluation (`eval_direct.py`)**: A baseline mode where the model answers the question directly without explicit scene graph generation.

## Setup

Ensure you have `vllm` and `datasets` installed:

```bash
pip install vllm datasets
```

## Usage

### 1. Graph-then-Plan (Main Method)

Run the evaluation with Scene Graph generation on MomaGraph-Bench:

```bash
python3 eval.py --model_path /path/to/your/model --dataset_name cheryyunl/momagraph-bench
```

- **Input**: Multi-view images + Task Instruction + Question.
- **Output**: A JSONL file containing the generated scene graph, final choice, and accuracy metrics.
- **Logic**: The model is prompted to analyze objects, spatial relations, and functional relationships before making a decision.

### 2. Direct QA Baseline

Run the direct baseline for comparison:

```bash
python3 eval_direct.py --model_path /path/to/your/model --dataset_name cheryyunl/momagraph-bench
```

## Citation

```bibtex
@article{momagraph2025,
  title={MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning},
  author={..},
  journal={arXiv preprint},
  year={2025}
}
```
