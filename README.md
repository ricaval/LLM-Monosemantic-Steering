# LLM-Monosemantic-Steering
## Steering LLMs with Semantic Feature Injection: An Exploration of Domain Precision

### Overview
This project explores the effect of activating extracted semantic features inside a language model's residual stream to improve domain-specific accuracy, using medical QA as the testbed.

Inspired by Anthropic's *Scaling Monosemanticity* paper, we adapt the technique of sparse autoencoder feature extraction and activation to investigate whether it can:

- Reduce hallucinations.
- Improve factual precision.
- Enhance model focus on complex domain questions.

### Research Questions
- Can injecting domain-relevant features reduce hallucination rates on medical questions?
- Does feature steering improve performance compared to prompt-only methods?
- How robust are these effects across formal and informal question phrasing?

### Methods

- **Model**: Mistral-7B Base.
- **Data**: 
  - Medical QA dataset (for final evaluation).
  - Curated external medical prompts (for feature extraction).
- **Technique**:
  1. Capture residual stream activations at a middle layer (Block 12).
  2. Train a Sparse Autoencoder (SAE) to extract interpretable feature vectors.
  3. Identify features associated with high factuality medical answers.
  4. Inject these features at inference.
- **Baselines**:
  - Unmodified model.
  - Model with medical tone prompting.
  - Model with direct semantic feature injection.

## Status
Structure and Environment setup in progress.

## Notes
- CUDA 12.8 targeted.
- Python 3.11 environment.
- Lightweight design for <=6h full training/eval runs.

## Citation / Inspiration
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet]([https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/) by Anthropic.

---

*This is an early-stage project, subject to change.*
