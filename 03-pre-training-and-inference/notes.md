# Part 03 — Pre-training and Inference
### Build a Small Language Model From Scratch
**Course by:** Vizuara (Dr. Raj Dandekar)
**Lecture video:** [YouTube — Part 3: Pre-training and Inference](https://youtu.be/lF1yuipDf3U?si=ht_8pARVepJ3CnPV)
**Miro Board:** https://miro.com/app/board/uXjVIL4LZB0=/
**Google Colab:** https://colab.research.google.com/drive/1vZR7FUAhgGuMMzMg-JnDLPU2J1YLdSaG#scrollTo=K8PgWXb-kjK7

---

## Table of Contents
1. [Overview & Objective](#1-overview--objective)
2. [The Forward Pass — Journey of a Token (Recap)](#2-the-forward-pass--journey-of-a-token-recap)
3. [Constructing the Loss Function](#3-constructing-the-loss-function)
4. [Calculating Loss for an Entire Batch](#4-calculating-loss-for-an-entire-batch)
5. [Backpropagation and the Update Rule](#5-backpropagation-and-the-update-rule)
6. [Parameter Sinks — Where Are the 100M Knobs?](#6-parameter-sinks--where-are-the-100m-knobs)
7. [The Pre-Training Loop & Optimizers](#7-the-pre-training-loop--optimizers)
8. [Overcoming Hardware Constraints — Gradient Accumulation](#8-overcoming-hardware-constraints--gradient-accumulation)
9. [Parameters vs. Hyperparameters](#9-parameters-vs-hyperparameters)
10. [Translating Theory to PyTorch Code](#10-translating-theory-to-pytorch-code)
11. [The Final Stage — Inference](#11-the-final-stage--inference)
12. [Domain Adaptation Strategy](#12-domain-adaptation-strategy)
13. [Scaling Up — Replicating GPT-2 from Scratch](#13-scaling-up--replicating-gpt-2-from-scratch)
14. [Specialized Models — BioGPT](#14-specialized-models--biogpt)

---

## 1. Overview & Objective

**Primary goal of this session:** Set up the training pipeline, define the loss function, and run pre-training and inference for a Small Language Model entirely from scratch on Google Colab.

**Target model size:** ~50M to 70M parameters (under 100M).

**What the model learns:** The form and semantic meaning of the English language — purely from the Next Token Prediction task. No grammar rules, no vocabulary lists — everything is absorbed implicitly.

**Dataset:** TinyStories — 2 million short stories designed for 3-to-4-year-old comprehension. The constrained, specific nature of this dataset is precisely what allows a sub-100M parameter model to converge successfully.

**Hardware constraint:** Training requires an **A100 GPU**. Attempting to train on a standard T4 GPU will take 8–10 hours and risks memory crashes.

---

## 2. The Forward Pass — Journey of a Token (Recap)

Before discussing how the model learns (backpropagation), we must trace how an input becomes a prediction. The model processes sequences in a batched format where the **target output is simply the input shifted right by one token**.

### ① The Input Block

| Step | Operation | Output |
|---|---|---|
| Token ID | Word (e.g., `"friend"`) → integer ID via vocabulary | Integer |
| Token Embedding | ID → high-dimensional vector | 768-dim vector |
| Position Embedding | Token's position (e.g., position #3) → 768-dim vector | 768-dim vector |
| Input Embedding | Token Embedding **+** Position Embedding | 768-dim vector |

The Input Embedding is the "uniform" the token wears before entering the Transformer block — it encodes both **what** the word is and **where** it sits in the sequence.

### ② The Processor Block (Transformer)

- Tokens pass sequentially through **multiple identical Transformer blocks** (e.g., 12 blocks — mirroring the GPT-2 base architecture).
- Each block contains two sub-layers:

**Multi-Head Self-Attention:**
- The **only place** in the entire architecture where a token looks at its neighbors.
- The Input Embedding absorbs surrounding context and exits as a much richer **Context Vector**.
- Masked so no token can attend to future tokens.

**Feed-Forward Neural Network (FFNN):**
- The 768-dim vector is **expanded** into a massive higher-dimensional space (768 → 4×768 = 3072).
- Then **compressed** back down to 768 dimensions.
- This expansion/contraction allows the network to discover complex non-linear patterns and representations that pure attention cannot capture alone.

### ③ The Output Block

| Step | Operation |
|---|---|
| Logits Layer | 768-dim context vector → projected up to vocabulary size (768 → 50,257) |
| Softmax | Converts 50,257 raw logits into a probability distribution summing to 1.0 |
| Prediction | `argmax` — the index with the highest probability is the predicted next token |

**Initially, these predictions are entirely random** — the parameters are randomly initialized, so the model has no knowledge of language yet. Training is what fixes this.

---

## 3. Constructing the Loss Function

To train the randomly initialized model, we must quantify exactly **how wrong** its predictions are. This is done via the **Negative Log-Likelihood Loss** (also called Cross-Entropy Loss).

### Setup

Assume an input sequence of 4 tokens. The model outputs a probability distribution across the entire vocabulary of 50,257 tokens **for each of the 4 positions**.

Because we have training data, we know the **Target Token IDs** that should be predicted next. Say the targets are: `[23, 3881, 11223, 15]`.

In a perfect model, the probability assigned to the correct target token at each position would be **1.0 (100%)**.

### Extracting the Target Probabilities

We do **not** look at which token gets the highest predicted probability. We look strictly at the probability the model assigned to the **correct target token**:

| Position | Target Token ID | Probability Assigned to Correct Target |
|---|---|---|
| 1 | 23 | P₁ = 0.1 |
| 2 | 3881 | P₂ = 0.3 |
| 3 | 11223 | P₃ = 0.4 |
| 4 | 15 | P₄ = 0.1 |

We want all these probabilities to be as close to **1.0** as possible.

### The Negative Log-Likelihood Formula

```
Loss = −( log(P₁) + log(P₂) + log(P₃) + log(P₄) )
```

### Why Use Logarithms?

**① Zero Loss at Perfection:**
- If the model is perfectly accurate, P = 1.0 → log(1.0) = 0.
- Total loss = 0. ✅

**② Heavy Penalization for Errors:**
- As the predicted probability for the correct token drops toward 0, log(P) → −∞.
- The negative sign flips this to a **massive positive loss penalty**.
- A model confidently wrong is penalized far more heavily than a model that is merely uncertain.

**③ Implicit Optimization of All Other Tokens:**
- Because Softmax forces all 50,257 probabilities to sum to 1.0, pushing the correct token's probability toward 1.0 **automatically crushes all 50,256 incorrect tokens toward 0**.
- We do **not** need to calculate separate loss terms for incorrect tokens — the mathematics handles it automatically.

---

## 4. Calculating Loss for an Entire Batch

While calculating loss for a single sequence is the foundation, models are trained on **batches** of multiple sequences simultaneously for stability and efficiency.

### Batch Loss Formula

If a batch has two sequences of 4 tokens each (8 total target predictions: P₁ through P₈):

```
Batch Loss = −(1/8) × ( log(P₁) + log(P₂) + ... + log(P₈) )
```

The individual losses are **averaged** across all token positions in the batch.

### End of the Forward Pass

This marks the completion of the Forward Pass. Strictly defined, the forward pass is:

```
Input batch
   ↓
Pass through SLM architecture
   ↓
Generate predicted next-token probability distributions (logits)
   ↓
Compute loss against ground truth targets
   ↓
Single scalar loss value
```

---

## 5. Backpropagation and the Update Rule

Once the model has a single scalar loss value for the batch, it must adjust its parameters to make that loss smaller in the future.

### Gradient Descent

- The model calculates the **partial derivative (gradient)** of the overall loss with respect to **every single one of its parameters** (e.g., all 100M parameters).
- The gradient points in the direction of **steepest ascent** on the loss landscape.
- We move in the **opposite direction** (steepest descent) to reduce the loss.

### The Update Rule

```
parameter_new = parameter_old − learning_rate × gradient
```

Every parameter is slightly adjusted by subtracting a small fraction of its gradient.

### Batched Gradient Descent

- Parameters are updated **after every single batch** is processed.
- This is more efficient than waiting for the entire dataset (full batch gradient descent) and more stable than updating after every single sample (stochastic gradient descent).

### Epochs

- Going through every batch in the entire dataset exactly **once** = one **Epoch**.
- Traditional ML might run for 50,000 epochs.
- Language models typically only need **2 or 3 epochs** because the datasets are so massive — the sheer volume of data provides enough gradient signal without repeated passes.

---

## 6. Parameter Sinks — Where Are the 100M Knobs?

Every trainable parameter must live somewhere in the architecture. Using three key variables:
- **V** = Vocabulary Size (e.g., 50,257)
- **E** = Embedding Dimension (e.g., 768)
- **C** = Context Size (e.g., 1,024)

| Component | Parameter Count | Example Size |
|---|---|---|
| Token Embedding Matrix | V × E | 50,257 × 768 ≈ 38.6M |
| Position Embedding Matrix | C × E | 1,024 × 768 ≈ 0.79M |
| Multi-Head Attention (Wq, Wk, Wv per block) | 3 × E² (per head, per block) | scales with n_heads × n_blocks |
| FFNN — Expansion Layer | E × 4E | 768 × 3,072 |
| FFNN — Contraction Layer | 4E × E | 3,072 × 768 |
| FFNN total per block | **8E²** | 8 × 768² ≈ 4.7M per block |
| Output Logits Layer | E × V | 768 × 50,257 ≈ 38.6M |

### ⚠️ Key Architectural Insight — The Vocabulary Size Bottleneck

**Vocabulary Size (V) appears in exactly TWO places:**
1. The very beginning — Token Embedding Matrix
2. The very end — Output Logits Layer

These two components together contribute ~77M parameters in GPT-2 base — before a single Transformer block is even counted.

- Making V **too large** → drastically inflates parameter count, expensive to train and run.
- Making V **too small** → harms model performance, can't represent the language well.

### 📄 GPT-2 Vocabulary Size Problem

The original GPT-2 (vocabulary size = 50,257) **struggled significantly with writing Python code** because its subword merging rules were not optimized for code indentation and syntax. GPT-4 later addressed this by increasing the vocabulary size and retraining the tokenizer specifically to handle code structures.

---

## 7. The Pre-Training Loop & Optimizers

The entire lifecycle of pre-training an SLM is a simple **5-step loop** repeated over every batch:

```
① Choose a batch      → Sample random Input/Target pairs from train.bin
② Forward pass        → Get logits from the SLM architecture
③ Compute loss        → Negative log-likelihood against targets
④ Backpropagation     → Calculate gradients for all parameters
⑤ Update parameters  → Gradient descent step
```

Repeat until training loss and validation loss converge.

### Advanced Optimizers — Adam / AdamW

Vanilla gradient descent (fixed learning rate) is **not used** in practice for LLMs. Modern LLMs use **Adam** or **AdamW**, which adaptively scale the learning rate for each parameter individually based on historical gradient information.

### Learning Rate Scheduling

A fixed learning rate throughout training is suboptimal. The correct approach uses a **two-phase schedule:**

| Phase | Behavior | Purpose |
|---|---|---|
| **Warm-up** | Learning rate ramps up quickly at the start | Exploration — quickly discover broad patterns in the loss landscape |
| **Decay** | Learning rate slowly ramps down | Exploitation — settle into a stable, precise local minimum |

Without warm-up, the model can make destructively large parameter updates early on. Without decay, it overshoots the minimum at the end of training.

---

## 8. Overcoming Hardware Constraints — Gradient Accumulation

### The Memory Problem

Large batch sizes (e.g., 1,024) provide much smoother, stable gradient updates. However, keeping an entire batch of 1,024 input/output matrices in GPU memory at once will instantly trigger an **Out of Memory (OOM) error** on most hardware.

### The Solution — Gradient Accumulation

Instead of processing the full batch at once, break it into **micro-batches** (e.g., chunks of 32):

```
Process micro-batch 1 (32 sequences)
   → Compute loss
   → Compute gradients
   → DO NOT update parameters yet — accumulate gradients in memory

Process micro-batch 2 (32 sequences)
   → Compute gradients
   → ADD to accumulated gradients

... repeat for all micro-batches ...

Once all micro-batches processed:
   → Average the accumulated gradients
   → Make ONE parameter update
```

**Result:** You achieve the **mathematical stability of a 1,024 batch size** while only ever keeping **32 sequences** in GPU memory at any given time. The parameter update is identical to what a true 1,024 batch would have produced.

---

## 9. Parameters vs. Hyperparameters

A critical distinction that separates a top-tier AI engineer from a beginner:

| Type | Definition | Examples |
|---|---|---|
| **Parameters** | Values the model **learns** and tunes during training via gradient descent | All weights in embedding matrices, attention projections, FFNN layers — the ~100M "knobs" |
| **Hyperparameters** | Architectural constraints **fixed by the engineer** before training begins — the model never touches these | See table below |

### Key Hyperparameters in This SLM

| Hyperparameter | Variable Name | Role |
|---|---|---|
| Embedding Dimension | `n_embed` | Size of every token's vector representation |
| Context Size | `block_size` | Maximum tokens the model can look at at once |
| Vocabulary Size | `vocab_size` | Number of unique tokens in the tokenizer |
| Number of Transformer Blocks | `n_layer` | Depth of the network |
| Number of Attention Heads | `n_head` | Number of parallel attention perspectives |
| Learning Rate Schedule | warm-up & decay rates | Controls how fast/slow parameters update |
| Dropout Rate | `dropout` | Percentage of neurons randomly deactivated during training to prevent overfitting |

---

## 10. Translating Theory to PyTorch Code

### The `get_batch` Function

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y
```

Randomly samples data from `.bin` files to assemble the **X** (Input) and **Y** (Target) tensors based on `batch_size` and `block_size`.

### The Forward Pass — 5 Lines of Code

```python
# Inside GPT.forward():
tok_emb = self.transformer.wte(idx)          # Token embeddings
pos_emb = self.transformer.wpe(pos)          # Position embeddings
x = tok_emb + pos_emb                        # Input embeddings
for block in self.transformer.h:             # Loop through all Transformer blocks
    x = block(x)
x = self.transformer.ln_f(x)                # Final LayerNorm
logits = self.lm_head(x)                    # Output Head → [B, T, vocab_size]
```

### The Loss Calculation

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

PyTorch's built-in `F.cross_entropy` **automatically combines both:**
1. The Softmax transformation
2. The Negative Log-Likelihood loss calculation

These do not need to be written separately.

### Monitoring Convergence

- The training loop prints **training loss** and **validation loss** every 500 batches.

**What to look for:**

| Pattern | Meaning |
|---|---|
| Training loss and validation loss both decrease together | ✅ Model is learning correctly |
| Validation loss spikes while training loss keeps falling | ❌ **Overfitting** — model is memorizing training data, not generalizing |

---

## 11. The Final Stage — Inference

Once pre-training is complete, **all parameters are frozen**. Backpropagation stops entirely.

### The Autoregressive Generation Loop

```
① Provide a starting prompt: e.g., "A little girl"
   ↓
② Pass through the frozen architecture
   ↓
③ Model outputs probability distribution over 50,257 tokens
   ↓
④ Select the most likely next token (argmax or sampling)
   ↓
⑤ Append the new token to the end of the input sequence
   ↓
⑥ Feed the new, longer sequence back into the model
   ↓
⑦ Repeat until desired length or end-of-sequence token
```

### Results

Live execution in Colab generated coherent, grammatically correct stories:
> *"Once upon a time there was a pumpkin..."*

### The Key Insight

Despite **never being explicitly taught** grammar, vocabulary, syntax, or sentence structure, the model naturally inferred all the rules of English language — **strictly by playing the "predict the next word" game** on 2 million simple stories.

### 📄 Paper Replicated — TinyStories
- By achieving coherent text generation, the course successfully **replicates the core premise** of the TinyStories paper:
> *"You do not need trillion-parameter models to generate cohesive text if the training dataset is sufficiently focused and constrained."*

### Extension — Regional Indian Languages
- The instructor's team expanded this SLM framework to **regional Indian languages** (Hindi, Marathi, Bangla).
- Published as open-source work — demonstrating the methodology is language-agnostic, not just English-specific.

---

## 12. Domain Adaptation Strategy

The codebase is **fully generalizable**. To build an SLM for any specialized industry (Medical, Legal, Civil Engineering, Finance):

### The Two-Part Dataset Rule

You **cannot** just feed the model domain-specific textbooks directly. The model must first learn the structure of language before it can learn domain jargon.

```
Part 1: Foundational English data
        → Teaches the model general grammar, sentence structure, and common vocabulary

Part 2: Highly specific domain articles and texts
        → Teaches the model industry terminology, concepts, and writing style
```

### Tune the Hyperparameters

- **Vocabulary size:** Adjust if the domain has heavy specialized terminology (e.g., medical Latin terms, legal phrases, code syntax).
- **Context window:** Increase if documents in the domain are long (e.g., legal contracts, research papers).
- **Number of blocks / embedding dimension:** Scale with dataset size and compute budget.

---

## 13. Scaling Up — Replicating GPT-2 from Scratch

The next objective after this course is to **fully replicate the original GPT-2 benchmark results** (specifically NanoGPT).

### Dataset Upgrade — FineWeb-Edu

| Dataset | Scale | Purpose |
|---|---|---|
| TinyStories | ~100M tokens | Learn English grammar and form |
| **FineWeb-Edu** | **10 Billion tokens** | Full world knowledge — ~100× larger |

FineWeb-Edu represents a massive culmination of world knowledge — the kind of diverse, high-quality data needed to replicate GPT-2's general language capability.

### Hardware Upgrade — 8× H100 Cluster

| Hardware | Use Case | Cost |
|---|---|---|
| A100 (Google Colab Pro) | This course — TinyStories SLM | ~$1–2/hr on Colab Pro |
| **8× H100 (RunPod)** | GPT-2 replication on 10B tokens | **~$20–25/hour** |

Training on 10 billion tokens requires chaining **eight H100 GPUs** together simultaneously — a production-scale infrastructure setup.

---

## 14. Specialized Models — BioGPT

### 📄 Paper: BioGPT
> *"BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining"*
> Submitted: 19 Oct 2022 (v1), last revised 3 Apr 2023 (v3)

- A well-known SLM built specifically for **biomedical data and research**.
- Demonstrates exactly how the SLM methodology can be applied to a highly specialized, technically dense domain.

### What the Course Will Do With BioGPT
- The original BioGPT open-source repository is highly complex.
- The instructor has broken down the architecture and adapted it into a **streamlined Google Colab notebook** using the foundational concepts from this bootcamp.
- This will serve as a **definitive template** for adapting SLMs to any enterprise domain.

### Addressing Past Confusion
- In previous "Build an LLM" tutorials from this channel, the class loaded **pre-trained weights from HuggingFace**.
- In this bootcamp, the model is **initialized randomly and trained 100% from scratch** — no borrowed weights, no fine-tuning.

### Homework Before Next Session
- [ ] Read the **original GPT-2 Paper** — understand the benchmark figures the class will attempt to replicate.
- [ ] Read the **BioGPT Paper** — understand the mechanics of domain-specific SLM training.
- [ ] Ensure sufficient budget on **RunPod** if planning to follow along with the 8× H100 cluster, or rely on **Colab Pro** for the BioGPT implementation.

---

## Resources

| Resource | Link |
|---|---|
| Lecture Video (YouTube) | https://youtu.be/lF1yuipDf3U?si=ht_8pARVepJ3CnPV |
| Google Colab Notebook | https://colab.research.google.com/drive/1vZR7FUAhgGuMMzMg-JnDLPU2J1YLdSaG#scrollTo=K8PgWXb-kjK7 |
| Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |
| TinyStories Paper (ArXiv) | https://arxiv.org/pdf/2305.07759 |
| GPT-2 Paper | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| BioGPT Paper (ArXiv) | https://arxiv.org/abs/2210.10341 |

---

## Questions to Revisit
- [ ] Why does `F.cross_entropy` in PyTorch apply Softmax internally — does this mean we should NOT apply Softmax before calling it or we get double-Softmax?
- [ ] What exactly does AdamW fix compared to Adam — what is "weight decay" and why does it help LLMs specifically?
- [ ] In gradient accumulation, does the order of micro-batches matter? What happens if micro-batches are not i.i.d. samples?
- [ ] Why do language models only need 2–3 epochs when traditional ML needs thousands — is this purely a dataset size argument or is there something architectural?
- [ ] How does the warm-up rate and decay rate get chosen — is this tuned empirically or are there principled guidelines?
- [ ] In the autoregressive generation loop, what is the difference between argmax decoding (greedy) and sampling with temperature — and when do you use each?
- [ ] What are the exact benchmark figures GPT-2 achieved that the next session will try to replicate?
