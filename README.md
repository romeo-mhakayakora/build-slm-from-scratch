# Build a Small Language Model From Scratch

> A 7-hour hands-on workshop by [Vizuara](https://www.youtube.com/@vizuara) — build, train, and run inference on a GPT-style Small Language Model entirely from scratch, no pre-trained weights, no shortcuts.

This project follows the methodology introduced in the Microsoft Research paper [**TinyStories: How Small Can Language Models Be and Still Speak Coherent English?**](https://arxiv.org/pdf/2305.07759) The core thesis: you do not need internet-scale data or trillion-parameter models to build a language model that genuinely understands and generates coherent English. A highly curated, domain-specific dataset and a sub-100M parameter architecture is enough.

By the end of this workshop you will have:
- Implemented a full **GPT-2 style Transformer** from scratch in PyTorch
- Trained it on 2 million short stories using an **A100 GPU**
- Run **autoregressive inference** and generated coherent English text
- Understood every single architectural decision from tokenization to the output head

---

## Repository Structure

```
build-slm-from-scratch/
│
├── 01-preprocessing-and-tokenization/
│   ├── notes.md                  ← Tokenization theory, BPE, input/output pairs,
│   │                                memory mapping, self-supervised learning
│   └── code/
│
├── 02-architecture/
│   ├── notes.md                  ← Input block, token & position embeddings,
│   │                                self-attention, QKV, masking, output head,
│   │                                Word2Vec, contextual embeddings
│   └── code/
│
├── 03-pre-training-and-inference/
│   ├── notes.md                  ← Loss function, backpropagation, gradient
│   │                                accumulation, training loop, inference,
│   │                                domain adaptation, BioGPT
│   └── code/
│
├── slm-model/                    ← Pre-trained model weights + inference notebook
│   ├── README.md
│   ├── Vizuara_AI_Labs_Small_Language_Model_Scratch_Final_(2) (1).ipynb
│   ├── best_model_params.zip.part001
│   └── best_model_params.zip.part002
│
├── resources/
│   ├── papers.md                 ← All papers referenced across the course
│   └── glossary.md               ← Key terms: SLM, BPE, autoregressive, etc.
│
├── .gitignore
└── README.md
```

---

## Course Parts

| Part | Topic | Key Concepts |
|---|---|---|
| **01** | Preprocessing + Tokenization | BPE, vocabulary, memory mapping, input/output pairs, self-supervised learning |
| **02** | Architecture | Token embeddings, positional embeddings, multi-head attention, QKV, causal masking, output head |
| **03** | Pre-training + Inference | Cross-entropy loss, backpropagation, gradient accumulation, AdamW, learning rate scheduling, autoregressive generation |

Notes for each part live in their respective folder. They are written to be **standalone and complete** — you can read any part without needing the others open.

---

## Quick Start — Run the Pre-trained Model

If you just want to load the pre-trained model and run inference without training from scratch:

```bash
# 1. Clone the repo
git clone https://github.com/romeo-mhakayakora/build-slm-from-scratch.git
cd build-slm-from-scratch/slm-model

# 2. Reassemble the split model weights
cat best_model_params.zip.part001 best_model_params.zip.part002 > best_model_params.zip
unzip best_model_params.zip
```

Then open `Vizuara_AI_Labs_Small_Language_Model_Scratch_Final_(2) (1).ipynb` in **Google Colab** or locally, load the weights, and run inference.

→ See [`slm-model/README.md`](./slm-model/README.md) for the full step-by-step guide including local setup and the known `google.colab` import issue.

---

## Training From Scratch

> ⚠️ Requires an **A100 GPU**. Training on a T4 takes 8–10 hours and risks OOM crashes.

The full training pipeline is in the notebook inside `slm-model/`. At a high level:

**1. Tokenize the dataset**
```python
# Uses tiktoken GPT-2 BPE encoder — vocab size 50,257
enc = tiktoken.get_encoding("gpt2")
```

**2. Memory-map tokens to disk**
```python
# Writes ~100M token IDs to train.bin and val.bin
# np.memmap lets you read/write without loading everything into RAM
arr = np.memmap("train.bin", dtype=np.uint16, mode="w+", shape=(total_tokens,))
```

**3. Run the training loop**
```python
# The full training lifecycle in 5 steps per batch:
# get_batch → forward pass → compute loss → backprop → update parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**4. Monitor convergence**

Training and validation loss should decrease together. Divergence = overfitting.

---

## Model Architecture

This is a decoder-only Transformer, architecturally equivalent to GPT-2 base, trained from random initialization.

```
Input Token IDs
       │
       ▼
┌─────────────────────────────┐
│        INPUT BLOCK          │
│  Token Embedding  [V × E]   │  V = 50,257  E = 768
│  + Position Embedding [C×E] │  C = 1,024
└────────────┬────────────────┘
             │  [B × T × 768]
             ▼
┌─────────────────────────────┐
│   × 12 TRANSFORMER BLOCKS   │
│                             │
│  ┌─ LayerNorm               │
│  ├─ Masked Multi-Head       │
│  │  Self-Attention (12 hd)  │
│  ├─ Residual connection     │
│  ├─ LayerNorm               │
│  ├─ Feed-Forward (768→3072  │
│  │  →768)                   │
│  └─ Residual connection     │
└────────────┬────────────────┘
             │  [B × T × 768]
             ▼
┌─────────────────────────────┐
│       OUTPUT BLOCK          │
│  Final LayerNorm            │
│  Linear head [768 → 50,257] │
│  Softmax → argmax           │
└────────────┬────────────────┘
             │
             ▼
     Predicted next token
```

| Hyperparameter | Value |
|---|---|
| Parameters | ~50M–70M |
| Embedding dimension | 768 |
| Transformer blocks | 12 |
| Attention heads | 12 |
| Context window | 1,024 tokens |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| FFNN expansion factor | 4× (768 → 3,072 → 768) |
| Optimizer | AdamW with warm-up + cosine decay |
| Training dataset | TinyStories (2M stories, ~100M tokens) |
| Training hardware | A100 GPU |
| Training time | ~1–2 hours |

---

## Notes Index

Each part has a comprehensive `notes.md` written at the level a high-performing student would take — no concept skipped, all papers cited, Q&A and open questions included.

| Part | Topic | Notes |
|---|---|---|
| 1 | Preprocessing + Tokenization | [01-preprocessing-and-tokenization/notes.md](./01-preprocessing-and-tokenization/notes.md) |
| 2 | Architecture | [02-architecture/notes.md](./02-architecture/notes.md) |
| 3 | Pre-training + Inference | [03-pre-training-and-inference/notes.md](./03-pre-training-and-inference/notes.md) |

---

## Papers Referenced

| Paper | Authors | Relevance |
|---|---|---|
| [TinyStories](https://arxiv.org/pdf/2305.07759) | Eldan, Li — Microsoft Research (2023) | Core dataset and SLM philosophy |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. (2017) | Transformer architecture foundation |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Radford et al. — OpenAI (2019) | Architecture template for this model |
| [Word2Vec](https://arxiv.org/abs/1301.3781) | Mikolov et al. — Google (2013) | Foundation of word embeddings |
| [BioGPT](https://arxiv.org/abs/2210.10341) | Luo et al. (2022) | Domain-specific SLM — next session |

---

## What Comes Next

This workshop is Part 1 of a broader series. Upcoming sessions:

- **Replicating GPT-2** — train on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B tokens) using an 8× H100 cluster (~$20–25/hr on RunPod) to match the original GPT-2 benchmark figures
- **BioGPT** — apply this exact SLM framework to the biomedical domain as a template for enterprise domain adaptation
- **Regional Language SLMs** — the Vizuara team has already published open-source SLMs for Hindi, Marathi, and Bangla using this same methodology

---

## Resources

| Resource | Link |
|---|---|
| Vizuara YouTube Playlist | [youtube.com/@vizuara](https://www.youtube.com/@vizuara) |
| Course Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |
| TinyStories Dataset (HuggingFace) | https://huggingface.co/datasets/roneneldan/TinyStories |
| Training Colab Notebook | https://colab.research.google.com/drive/1vZR7FUAhgGuMMzMg-JnDLPU2J1YLdSaG |
| Vizuara Substack | https://vizuara.substack.com/p/from-words-to-vectors-understanding |

---

## Contributing

This is a personal learning repository. If you spot an error in the notes or the README, open an issue or a PR — happy to fix it.

---

*Built by [@romeo-mhakayakora](https://github.com/romeo-mhakayakora) following the Vizuara 7-Hour Workshop.*
