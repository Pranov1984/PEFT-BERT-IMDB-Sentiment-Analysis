# PEFT-BERT-IMDB-Sentiment-Analysis
GitHub Repository README.md Write-up
Here is a clean, professional draft of the README.md section summarizing this use case, ready to be copied directly into your repository:

# Parameter-Efficient Fine-Tuning of BERT using QLoRA

This repository demonstrates how to fine-tune a pre-trained BERT model (`bert-base-uncased`) for binary sentiment classification on IMDB movie reviews using **QLoRA (Quantized Low-Rank Adaptation)**. We contrast parameter-efficient fine-tuning (PEFT) with conventional full fine-tuning to evaluate hardware memory efficiency, checkpoint size, training runtime, and classification accuracy.

## ★ Key Results & Performance Comparison

By leveraging 4-bit NF4 quantization along with standard LoRA adapters (configured with rank $r=16$, $\alpha=32$ on self-attention layers), we achieved nearly identical classification accuracy while significantly lowering memory demands:

| Metric | Full Fine-Tuning | QLoRA (PEFT) | Improvement / Trade-off |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 109,483,778 (100%) | 886,274 (0.80%) | **124x fewer parameters** |
| **Model Weights (GPU)** | 418 MB | 132 MB | **3.2x reduction** in model footprint |
| **Peak Training Memory** | 3,092 MB | 1,857 MB | **1.7x lower peak VRAM** |
| **Saved Checkpoint Size** | 418 MB | 3.39 MB | **123x smaller checkpoint** |
| **Test Accuracy** | **90.39%** | **89.48%** | Negligible accuracy loss (~0.91 points) |
| **Training Time (2 epochs)** | 2.9 minutes | 3.2 minutes | ~10% compute overhead (on-the-fly dequantization) |

---

## ☑ Project Highlights

* **Quantized Base Model:** Loaded the BERT encoder in 4-bit NormalFloat (NF4) precision to drastically minimize weight memory consumption on the GPU.
* **PEFT Configuration:** Implemented low-rank adapters targeting the attention modules (`query`, `key`, `value`) while keeping the base BERT layers frozen.
* **Validation & Testing:** Logged real-time loss and classification metrics directly to **Weights & Biases (W&B)** and evaluated the final checkpoints over 10,000 unseen IMDB test reviews.
* **Portability:** Shows how to export and load just the **3.39 MB** adapter weights back onto a fresh, frozen 4-bit base model in production.

## ⚙ Requirements & Dependencies

To replicate this project, install the following packages:
```bash
pip install transformers peft bitsandbytes accelerate datasets wandb
\`\`\`
