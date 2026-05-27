# Part 02 — Assembling the Model Architecture
### Build a Small Language Model From Scratch
**Course by:** Vizuara (Dr. Raj Dandekar)
**Lecture video:** [YouTube — Part 2: Assemble the Model Architecture](https://www.youtube.com/watch?v=your-part2-link)
**Miro Board:** https://miro.com/app/board/uXjVIL4LZB0=/

---

## Table of Contents
1. [Recap — Data Processing & Setup](#1-recap--data-processing--setup)
2. [The High-Level Architecture — Three Blocks](#2-the-high-level-architecture--three-blocks)
3. [Deep Dive — The Input Block](#3-deep-dive--the-input-block)
   - [A. Token Embeddings](#a-token-embeddings)
   - [B. Position Embeddings](#b-position-embeddings)
   - [C. Creating the Final Input Embedding](#c-creating-the-final-input-embedding)
4. [Tensor Shape Mental Model](#4-tensor-shape-mental-model)
5. [Self-Attention — The Processor Block (Foundations)](#5-self-attention--the-processor-block-foundations)
   - [Q, K, V — Queries, Keys, Values](#q-k-v--queries-keys-values)
   - [The Attention Matrix — QKᵀ](#the-attention-matrix--qk)
   - [Why Scale by √dₖ?](#why-scale-by-dk)
6. [Decoding "Masked Multi-Head" Attention](#6-decoding-masked-multi-head-attention)
   - [Why "Multi-Head"?](#a-why-multi-head)
   - [Why "Masked"?](#b-why-masked-causal-attention)
7. [The Output Block — Predicting the Next Token](#7-the-output-block--predicting-the-next-token)
8. [Translating Theory to Code — PyTorch Implementation](#8-translating-theory-to-code--pytorch-implementation)
9. [Hardware & Enterprise Constraints](#9-hardware--enterprise-constraints)
10. [Supplementary Reading — From Words to Vectors (Vizuara Substack)](#10-supplementary-reading--from-words-to-vectors-vizuara-substack)
11. [Q&A — Do We Train the Tokenizer Alongside the Model?](#11-qa--do-we-train-the-tokenizer-alongside-the-model)

---

## 1. Recap — Data Processing & Setup

**Goal of the course:** Build a Small Language Model (10M–100M parameters) pre-trained on the TinyStories dataset (stories aimed at 3-to-4-year-olds) to generate coherent English text.

**The Core Task:** Language models learn the form and semantic meaning of language purely as a byproduct of the **Next Token Prediction** task. We never explicitly teach them grammar or logic — it is all absorbed through the repetition of predicting the next token.

### Pre-processing Pipeline (Quick Recap)

**① Tokenization:**
- Uses **Subword-based tokenization** — specifically the **Byte Pair Encoder (BPE)** from GPT-2.
- Avoids the pitfalls of strict word-based tokenization (OOV problem) and character-based tokenization (context window explosion).
- Maintains a vocabulary of characters, words, and subwords, assigning each a unique **Token ID**.
- GPT-2 vocabulary size = **50,257**.

**② Memory Mapping:**
- To prevent RAM crashes and speed up training, tokenized data is batched and saved directly to the disk as `.bin` files.
- e.g., `train.bin` contains ~100 million token IDs.
- Uses `np.memmap` to read/write from disk as if it were in RAM.

**③ Input / Target Pairs:**
- Governed by `block_size` (Context Window Size) and `batch_size`.
- The target output is simply the input sequence **shifted right by one token**.
- Example (context size = 4):
  - Input: `"One day a little"`
  - Target: `"day a little girl"`
- This creates **4 distinct prediction tasks** in one block simultaneously.

---

## 2. The High-Level Architecture — Three Blocks

The journey from raw input tokens to predicted next tokens passes through **three distinct schematic blocks:**

```
Token IDs  →  [ INPUT BLOCK ]  →  [ PROCESSOR BLOCK ]  →  [ OUTPUT BLOCK ]  →  Next Token Prediction
```

| Block | Role | Focus |
|---|---|---|
| **Input Block** | Transforms integer Token IDs into rich mathematical vectors | This lecture |
| **Processor Block** | Multi-Head Self-Attention — the "brain" that learns relationships between tokens | Next lecture |
| **Output Block** | Projects the final vectors back to vocabulary space to predict the next token | Next lecture |

### 📄 Key Paper: "Attention Is All You Need" (2017)
- The seminal paper that sparked the revolution in modern language modeling.
- Introduced the **Multi-Head Attention** mechanism that forms the crucial **Processor Block** of the architecture.
- Every modern LLM/SLM is built on this foundation.

---

## 3. Deep Dive — The Input Block

While the Processor Block gets the glory, an inefficient Input Block ruins the entire model. The Input Block handles the transformation of integer Token IDs into **rich, mathematical vectors** that carry both semantic meaning and positional information.

The Input Block has two components that are combined:
1. **Token Embeddings** — encodes *what* the word means
2. **Position Embeddings** — encodes *where* the word sits in the sequence

---

### A. Token Embeddings

**The Problem:** Raw Token IDs (e.g., Token #5) encode absolutely **no semantic meaning**. They are just arbitrary integers — the number 5 tells the model nothing about what word it represents or how it relates to other words.

**The Solution:** Represent every token as a **high-dimensional vector**.

- In a properly trained model, words with similar meanings will physically **cluster closer together** in vector space.
- Examples:
  - `"cat"` and `"kitten"` → close in vector space
  - `"refrigerator"` and `"microwave"` → close in vector space
  - `"cat"` and `"democracy"` → far apart in vector space

### 📄 Key Paper: Word2Vec — Google (2014)
- A foundational breakthrough proving that neural networks could effectively map semantic word meanings to high-dimensional vectors.
- Established the concept that geometric relationships in vector space encode real-world semantic relationships.

### The Token Embedding Matrix

- Initially, every token is assigned a **completely random vector**. These values are **trainable parameters** — the model tunes them over training.
- **Dimensions:** `Vocabulary Size × Embedding Dimension`

| Model | Vocabulary Size | Embedding Dimension | Matrix Size | Parameters introduced |
|---|---|---|---|---|
| GPT-2 | 50,257 | 768 | [50,257 × 768] | ~38 million |
| Our SLM | 50,257 | varies | varies | varies |

- The **~38 million parameters** from just the Token Embedding Matrix alone illustrates why even "small" models have tens of millions of parameters before any processing begins.

---

### B. Position Embeddings

**The Problem:** Token Embeddings do not understand **sequence order**. Consider:

> *"The dog chased the cat, it could not catch it."*

The two instances of `"it"` refer to completely different entities — the first `"it"` refers to the cat, the second to the dog. Standard token embeddings would assign the exact same vector to both instances of `"it"` because they are the same word. The model has no way to distinguish them without positional information.

**The Solution:** Create a **secondary matrix** to encode the spatial position of every token in the sequence.

### The Position Embedding Matrix

- Unlike the Token Embedding Matrix, the number of rows here is determined strictly by the **Context Size** (how many tokens the model can look at at once).
- **Dimensions:** `Context Size × Embedding Dimension`

| Model | Context Size | Embedding Dimension | Matrix Size |
|---|---|---|---|
| GPT-2 | 1,024 | 768 | [1,024 × 768] |
| Our SLM | block_size | embedding_dim | [block_size × embedding_dim] |

- Position 0 gets its own 768-dimensional vector, Position 1 gets its own, and so on up to the maximum context size.
- These are also **trainable parameters** — the model learns the best positional representations during training.

---

### C. Creating the Final Input Embedding

To prepare data for the Processor Block, the model merges token and positional information into a single vector for every token.

**The process for each token:**

```
Step 1: Lookup Token
        → Retrieve the token's 768-dim vector from the Token Embedding Matrix

Step 2: Lookup Position
        → Retrieve that token's position's 768-dim vector from the Position Embedding Matrix

Step 3: Add them together
        → Token Embedding Vector + Position Embedding Vector = Input Embedding Vector
```

**Result:** Every single token is now represented by a rich **768-dimensional Input Embedding vector** that encodes:
- ✅ **What** the word means (semantic meaning from Token Embeddings)
- ✅ **Where** the word is located in the sequence (positional meaning from Position Embeddings)

These vectors are now ready to be passed into the **Processor Block**.

**Final Input Embedding Tensor shape:**
```
(Batch Size, Context Length, Embedding Dimension)
e.g., (32, 1024, 768)
```

---

## 4. Tensor Shape Mental Model

A critical skill when working with transformers is reasoning about tensor shapes intuitively. The key insight is to think **"outer container → inner contents"**.

### The Container Analogy

```
(B, T, D)  →  B containers (batch)
               └── each contains T tokens (sequence)
                    └── each token is a D-dimensional vector (features)
```

**Practical example:** `(32, 1024, 4096)` means:
- 32 sequences (batch)
- each sequence has 1,024 tokens
- each token is a vector of size 4,096

### General Rule
> Dimensions go from **broader grouping → finer detail**

| Domain | Tensor Shape | Reading |
|---|---|---|
| Transformers | `(batch, sequence, features)` | batch → tokens → embedding values |
| Images | `(batch, height, width, channels)` | batch → spatial → color |
| Video | `(batch, frames, pixels, channels)` | batch → time → spatial → color |

### Where Context Length × Context Length Appears

The `T × T` shape appears **inside self-attention**, NOT in the embedding tensor. This is a common source of confusion.

| Tensor | Shape | What it represents |
|---|---|---|
| Input Embedding | `(B, T, D)` | Each token as a D-dim vector |
| Attention Matrix (QKᵀ) | `(B, H, T, T)` | Every token attending to every other token |

Where `H` = number of attention heads.

**Visualizing the attention shape** using the container analogy:
```
(B, H, T, T) →  batch
                 └── attention head
                      └── query token
                           └── scores against every key token
```

The last `T × T` is literally: *"for each token, how much attention goes to every other token"* — which is why attention memory cost grows **quadratically** with sequence length. This is the core scalability bottleneck of the Transformer architecture.

---

## 5. Self-Attention — The Processor Block (Foundations)

### Q, K, V — Queries, Keys, Values

Each token produces **three vectors** via learned linear projections:

| Vector | Symbol | Intuition |
|---|---|---|
| **Query** | qᵢ | "What am I looking for?" |
| **Key** | kᵢ | "What do I contain / offer?" |
| **Value** | vᵢ | "What information do I pass forward if attended to?" |

**Example:** For the sequence `"The cat sat"`:
- Token `"sat"` produces a query vector that strongly matches the key vector for `"cat"`.
- This produces a high attention score → `"sat"` attends heavily to `"cat"`.
- The model learns that the verb relates back to its subject — without being explicitly told.

---

### The Attention Matrix — QKᵀ

**Setup:**

Both Q and K are matrices where **each row is a vector for one token:**

```
Q = [ q₁ ]       K = [ k₁ ]
    [ q₂ ]           [ k₂ ]
    [ q₃ ]           [ k₃ ]
```

**The transpose trick:**

Transposing K makes key vectors become **columns:**

```
Kᵀ = [ k₁ᵀ  k₂ᵀ  k₃ᵀ ]
```

**Matrix multiplication gives all pairwise attention scores simultaneously:**

```
QKᵀ = [ q₁·k₁  q₁·k₂  q₁·k₃ ]
      [ q₂·k₁  q₂·k₂  q₂·k₃ ]
      [ q₃·k₁  q₃·k₂  q₃·k₃ ]
```

Each cell `(i, j)` = dot product `qᵢ · kⱼ` = **"How much should token i attend to token j?"**

**Grid visualization:**

```
              Keys
           k₁  k₂  k₃
Queries q₁ [  ·   ·   · ]
        q₂ [  ·   ·   · ]
        q₃ [  ·   ·   · ]
```

Summary:
- **Queries stay as rows**
- **Keys are transposed into columns**
- Multiplication produces all token-to-token attention scores in one elegant operation

> **Important — Causal Masking:** For a **decoder-only (causal) transformer**, the attention matrix is **masked** — the upper triangle is set to `-∞` before softmax. This ensures `e^(-∞) = 0` after softmax, meaning no token can attend to future tokens. The model cannot "cheat" by looking ahead at words not yet generated.

---

### Why Scale by √dₖ?

The full self-attention formula is:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

Where `dₖ` = the dimension of the key/query vectors.

**The Problem Without Scaling:**

Suppose `dₖ = 1024`. The dot product `q · k` sums together 1,024 multiplications. As dimension increases, the magnitude of the dot product grows roughly proportional to `√dₖ`. With large dimensions, attention logits become very large numbers.

**Why Large Logits Break Softmax:**

Softmax is extremely sensitive to large input values:

```
Healthy:  softmax([2, 3, 4])       → [0.09, 0.24, 0.67]  ← smooth, well-distributed
Broken:   softmax([200, 300, 400]) → [≈0,   ≈0,   ≈1  ]  ← collapsed to one-hot
```

When softmax collapses:
- Attention becomes **overly sharp** — the model only attends to a single token
- Gradients become **near-zero** → vanishing gradient problem
- Training becomes **unstable and slow**

**The Fix:**

Dividing by `√dₖ` normalizes the variance of dot products back to a reasonable range before softmax:

```
Higher-dimensional vectors → bigger dot products → divide by √dₖ to compensate
```

This keeps:
- ✅ Softmax smooth and well-distributed across tokens
- ✅ Gradients healthy throughout training
- ✅ Training stable regardless of embedding dimension size

**Core intuition:** Higher-dimensional vectors naturally produce bigger dot products — we compensate by shrinking them proportionally before softmax.

---

## 6. Decoding "Masked Multi-Head" Attention

The full name of the mechanism used in the Processor Block is **Masked Multi-Head Attention**. Each word in that title has a critical architectural meaning.

### A. Why "Multi-Head"?

**The Problem:** A single attention head (one set of Wq, Wk, Wv matrices) only captures **one perspective** of language. However, language is highly ambiguous.

> *"The artist painted the portrait of a woman with a brush."*
> - Is the artist painting **using** a brush?
> - Or is it a portrait of a woman **who is holding** a brush?

A single attention head cannot resolve both interpretations simultaneously.

**The Solution:** Use **multiple attention heads** (e.g., 12 heads in GPT-2) running **in parallel**. Each head acts as an independent "perspective" learner, capturing different syntactic or semantic relationships simultaneously. The outputs of all heads are then concatenated and projected back down.

### B. Why "Masked" (Causal Attention)?

**The Problem:** When training for Next Token Prediction, the model is given a sequence (e.g., `"The next day is bright"`) and must predict token T+1 based only on tokens 1 through T. If the attention matrix calculates scores for the entire sequence simultaneously, token 1 (`"The"`) would be able to physically **"see"** token 2 (`"next"`) during the calculation. This is cheating — the model would be looking at the answer during the exam.

**The Solution — Masking:** All values representing future tokens (all elements strictly **above the diagonal** of the attention matrix) are forced to `-∞`, which becomes exactly `0` after Softmax (`e^(-∞) = 0`).

**Result:**
```
Token 1 → attends to: Token 1 only
Token 2 → attends to: Tokens 1, 2
Token 3 → attends to: Tokens 1, 2, 3
Token 4 → attends to: Tokens 1, 2, 3, 4
```

No token can ever attend to a future token.

### 📄 Modern Innovations on Attention
- **DeepSeek — Multi-Head Latent Attention (MLA):** Focuses optimizations directly on the Attention pathway to improve efficiency without losing context.
- **Mixture of Experts (MoE):** Architectural innovation targeting the Feed-Forward pathways within each Transformer block for efficiency gains.

---

## 7. The Output Block — Predicting the Next Token

After passing through all Transformer blocks, each token has an incredibly rich contextual representation — but it is still a vector in 768-dimensional space. The model must now translate these math vectors back into actual English words.

### Step 1 — Final Layer Normalization
- The `[4 × 768]` matrix undergoes one final normalization to stabilize the mean and variance before prediction.
- This is the same LayerNorm used throughout the Transformer blocks, applied one last time.

### Step 2 — The Output Head (Linear Projection)
- The matrix is multiplied by a final neural network layer that **expands** the dimension from the Embedding Dimension up to the full Vocabulary Size.

```
[4 × 768]  →  Output Head (Linear Layer)  →  [4 × 50,257]
```

- **Meaning:** For each of the 4 token positions, the model now has a list of **50,257 raw numerical scores** called **logits** — one score per word in the vocabulary.

### Step 3 — Softmax & Decoding

```
[4 × 50,257] logits  →  Softmax  →  [4 × 50,257] probabilities
```

- Softmax converts the 50,257 raw logits into a **probability distribution** that sums to 100%.
- The model takes the **argmax** — the index with the highest probability.
- It looks up that index in the original BPE vocabulary dictionary to recover the actual text token.

**Concrete example:**
- If index `5,000` has the highest probability for position 1, and index `5,000` maps to `"the"` in the BPE dictionary → the model predicts `"the"` as the next token.

### Full End-to-End Flow (Summary)

```
Token IDs
   ↓
Token Embedding Matrix  +  Position Embedding Matrix
   ↓ (addition)
Input Embedding Tensor  [B × T × D]
   ↓
× 12 Transformer Blocks (Masked Multi-Head Attention + FFN + LayerNorm + Residuals)
   ↓
Final Layer Norm
   ↓
Output Head  [T × D]  →  [T × Vocab Size]
   ↓
Softmax  →  Probability Distribution over 50,257 tokens
   ↓
argmax  →  Predicted Next Token
```

---

## 8. Translating Theory to Code — PyTorch Implementation

The full architecture maps directly to modular PyTorch classes that mirror the whiteboard theory exactly:

| Class | What it implements |
|---|---|
| `MLP` | The Feed-Forward Neural Network inside each Transformer block: Expansion layer → Activation (GeLU) → Contraction layer |
| `LayerNorm` | Mean subtraction and variance division to stabilize activations |
| `CausalSelfAttention` | Wq, Wk, Wv projections + causal masking of future tokens + weighted summation to produce context vectors |
| `Block` | One complete Transformer layer: CausalSelfAttention + MLP + LayerNorm + **Residual (Shortcut) connections** |
| `GPT` | The grand assembly: initializes Token & Position Embedding matrices → loops through N `Block` layers sequentially → applies the final Output Head linear projection |

---

## 9. Hardware & Enterprise Constraints

### Compute Requirements

Training even a Small Language Model (10M–100M parameters) requires significant compute:

| Hardware | Training Time | Risk |
|---|---|---|
| **A100 GPU** (Google Colab Pro) | ~1–2 hours | Low |
| **T4 GPU** (standard Colab) | ~8–9 hours | High — memory crash likely |

**Enterprise takeaway:** You cannot reliably train or run robust models in enterprise environments on T4 hardware. Upgraded GPU clusters are mandatory for production-grade SLM pre-training.

### Context Windows — Clarification

When models like **LLaMA** advertise a *"200k Context Window"*, it simply means their input sequence size (`block_size`) allows for **200,000 parallel tokens per batch row**. This does **not** mean any fundamental architectural change — it just means the Attention matrices are proportionally enormous, since attention memory scales as:

```
O(T²)   where T = context length
```

A 200k context window means attention matrices of size `[200,000 × 200,000]` — which is why long-context models require specialized hardware and memory-efficient attention variants (e.g., Flash Attention).

---

---

## 10. Supplementary Reading — From Words to Vectors (Vizuara Substack)

**Source:** [From Words to Vectors: Understanding Word Embeddings in NLP](https://vizuara.substack.com/p/from-words-to-vectors-understanding)
**Author:** Mayank Pratap Singh (Vizuara)

This article deep-dives into the full evolution of how words became vectors — the foundation that makes the Token Embedding Matrix in our SLM possible.

---

### Why Turn Words into Numbers?

Computers operate on numbers, not text. To make a machine understand language, we must **encode** words into numerical representations. This process evolved through several increasingly powerful approaches:

---

### Stage 1 — Integer Encoding (Token IDs)

Assign each word a unique numeric ID:
- `"apple"` → 1, `"banana"` → 2, `"cat"` → 3

**Fatal flaw:** These numbers carry **no semantic meaning**. The model has no clue that `"cat"` and `"dog"` are more related to each other than to `"banana"`. They are just arbitrary labels.

---

### Stage 2 — One-Hot Encoding

Each word is represented as a binary vector of length = vocabulary size, with a single `1` at the word's position and `0` everywhere else:
- `"apple"` → `[1, 0, 0, 0, 0]`
- `"banana"` → `[0, 1, 0, 0, 0]`
- `"cat"` → `[0, 0, 1, 0, 0]`

**Improvement:** Removes the false numerical ranking between words.

**Two fatal flaws:**
- **Sparse & huge:** With a 600,000-word vocabulary, each vector is 600,000 numbers long with only one `1`. Extremely inefficient in memory and computation.
- **Still no semantic meaning:** Every word's vector is equally different from every other word's. `"cat"` is as "unrelated" to `"dog"` as it is to `"table"` — which is wrong.

---

### Stage 3 — Bag of Words (BoW)

Represents an entire piece of text by counting word occurrences across the vocabulary:

**Improvement:** Captures which words are present and how often.

**Fatal flaw:** Completely **ignores word order**. The sentences `"dog bites man"` and `"man bites dog"` produce identical BoW vectors — but mean completely different things.

---

### The Core Insight — The Distributional Hypothesis

> *"You shall know a word by the company it keeps."* — J.R. Firth

This is the foundational linguistic principle behind all modern embeddings: **words that appear in similar contexts tend to have similar meanings.**

**Example:** If you see *"The glorp is barking and wagging its tail"*, you can infer `"glorp"` is probably a dog — purely from context. The surrounding words carry the meaning.

Applied to machine learning: if `"doctor"` and `"nurse"` both frequently appear near `"hospital"` and `"patient"`, a model can learn they are semantically related — without being told explicitly.

---

### Stage 4 — Word2Vec (Google, 2013)

Word2Vec operationalizes the Distributional Hypothesis by training a simple neural network to predict words from their context (or context from words). The learned internal weights become the word vectors.

**Two training strategies:**

**① CBOW (Continuous Bag of Words)**
- Input: surrounding context words
- Task: predict the missing center word
- Example: `["The", "cat", "on", "the", "mat"]` → predict `"sat"`
- The model must learn which words tend to fill which contexts → vectors encode meaning.

**② Skip-Gram**
- Input: one center word
- Task: predict the surrounding context words
- Example: given `"coffee"` → predict `"morning"`, `"mug"`, `"bean"` nearby
- The reverse of CBOW — learns to map a word to its likely neighborhood.

**The result:** After training on millions of sentences, words that share similar contexts cluster together in vector space. Words that share no context are geometrically distant.

---

### The Famous Analogy — King − Man + Woman ≈ Queen

One of the most striking demonstrations of what Word2Vec learns:

```
vector("king") − vector("man") + vector("woman") ≈ vector("queen")
```

This works because the model has learned that the geometric **direction** from `"man"` to `"king"` (royalty) is the same direction as from `"woman"` to `"queen"`. Semantic relationships are encoded as directions in vector space.

Equivalently: `queen − woman = king − man` — the relationship between "queen" and "woman" (royalty) is the same as between "king" and "man".

---

### Measuring Similarity — Cosine Similarity

Once words are vectors, similarity is measured by **cosine similarity** — the angle between two vectors, not their magnitude.

| Angle | Cosine Value | Meaning |
|---|---|---|
| Small angle (pointing same direction) | Close to 1.0 | Words are semantically similar |
| 90° (perpendicular) | 0 | No particular similarity |
| 180° (opposite directions) | -1 | Potentially opposite meanings |

**Why angle over magnitude?** Some words appear more frequently and get larger vector magnitudes — but what encodes meaning is the *direction* (the pattern of values across dimensions), not the size. Cosine similarity normalizes magnitude out.

**Example:** Computing cosine similarity between `"dog"` and all other words might yield:
- `"cat"` → 0.8
- `"wolf"` → 0.75
- `"banana"` → 0.2

This aligns perfectly with intuition.

---

### Interpretability — What Do the Dimensions Mean?

Word vectors are typically 300–768+ dimensions. Each dimension is a real-valued number, but they are **learned in an unsupervised way** — no dimension is explicitly labeled.

However, by visualizing vectors as heatmaps (negative values = purple, positive = red), patterns emerge:

**Case Study 1 — Living vs. Non-Living** (`man`, `woman`, `boy`, `girl`, `banana`, `water`):
- In dimension 6, all living entities show strong negative values; non-living objects do not.
- Hypothesis: dimension 6 may encode something related to animacy or agency.

**Case Study 2 — Emotions** (`happy`, `joyful`, `sad`):
- `happy` and `joyful` show nearly identical patterns across most dimensions.
- At dimension 6: `happy` = 0.04, `joyful` = 0.17, `sad` = 0.86 — a clear divergence.
- Hypothesis: dimension 6 may encode emotional polarity.

**Case Study 3 — Life Stages** (`baby`, `child`, `teenager`, `adult`):
- All four words have notably similar vectors, sharing patterns at dimensions 2, 3, and 5.
- At dimension 8: `baby` = 1.30 → `child` = 1.17 → `teenager` = 0.98 → `adult` = 0.79 — a descending trend.
- Hypothesis: may encode something like "available free time" or developmental stage.

**Key takeaway:** We cannot definitively label individual dimensions. Interpretability comes from **relative comparisons and observed patterns**, not absolute values. But consistent patterns confirm the model is encoding real semantic structure.

---

### Stage 5 — Contextual Embeddings (Transformers: BERT, GPT)

**The remaining flaw with Word2Vec:** Every word gets **one static vector** regardless of context. But many words are **polysemous** — they have multiple meanings depending on use.

**Classic example:**
- *"Mayank is sitting quietly on the river **bank**, watching the water flow."*
- *"Mayank is robbing the **bank** downtown with a mask on his face."*

Word2Vec assigns the exact same vector to `"bank"` in both sentences — it has to pick one point in space that awkwardly averages both meanings.

**The Transformer solution — Contextual Embeddings:**
- Models like **BERT** and **GPT** generate a **fresh vector for each token occurrence** based on the full sentence it appears in.
- `"bank"` in the finance sentence gets a vector close to `"money"`, `"deposit"`, `"account"`.
- `"bank"` in the river sentence gets a vector close to `"water"`, `"river"`, `"edge"`.
- Same word, two completely different vectors — determined by context via the self-attention mechanism.

This is exactly the problem that the **Processor Block (Multi-Head Self-Attention)** in our SLM solves. The Token Embedding Matrix gives each token a starting static vector — but the Transformer layers then **update and refine** those vectors based on the full surrounding context.

---

### Evolution Summary

| Method | Semantic Meaning | Handles OOV | Context-Aware | Practical |
|---|---|---|---|---|
| Integer IDs | ❌ | ❌ | ❌ | ✅ |
| One-Hot | ❌ | ❌ | ❌ | ❌ (sparse) |
| Bag of Words | ❌ | ❌ | ❌ | ⚠️ |
| Word2Vec | ✅ (static) | ❌ | ❌ | ✅ |
| Transformers (BERT/GPT) | ✅ (contextual) | ✅ | ✅ | ✅ |

---

## 11. Q&A — Do We Train the Tokenizer Alongside the Model?

**Short answer: No. The tokenizer is trained separately first, then frozen. The model trains on its own after.**

### Tokenizer Training (done once, upfront)
- The BPE tokenizer is trained on the raw text corpus **before** the model ever sees any data.
- It learns which character pairs to merge based purely on **frequency statistics** in the text — no neural network, no gradients, no backpropagation involved.
- Once trained, the vocabulary (e.g., 50,257 tokens in GPT-2) is **locked / frozen**.
- In this course specifically, we are not training the tokenizer from scratch — we **reuse the pre-built GPT-2 BPE tokenizer** via `tiktoken`. This step is already done for us.

### Model Training (after tokenization)
- Once the tokenizer is frozen, the entire corpus is tokenized into token IDs and written to `train.bin` / `val.bin` — **this happens once as a preprocessing step**.
- Only then does actual model training begin — gradient descent tuning the Token Embedding Matrix, Position Embedding Matrix, Q/K/V projection weights, MLP weights, Output Head, etc.
- The tokenizer is **never touched again** during model training.

### Why You Cannot Train Them Together
- The tokenizer's vocabulary defines the **shape of the Token Embedding Matrix** (`Vocabulary Size × Embedding Dimension`).
- If the vocabulary kept changing during training, the embedding matrix dimensions would change — **breaking the entire architecture mid-training**.
- The model needs a **stable, fixed input space** to learn from.

### The Full Pipeline in Order
```
Raw Text
   ↓
① Train / load tokenizer  (BPE — frequency-based statistics, no gradients)
   ↓
② Tokenize entire corpus → train.bin / val.bin  (done once, offline)
   ↓
③ Train the model  (gradient descent on frozen vocabulary)
```

### Key distinction
These are fundamentally **two different types of "training":**
- Tokenizer → uses **frequency statistics** (no neural network)
- Model → uses **backpropagation and gradient descent**

They happen **sequentially, never simultaneously**.

---

## Resources

| Resource | Link |
|---|---|
| Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |
| "Attention Is All You Need" (2017) | https://arxiv.org/abs/1706.03762 |
| Word2Vec Paper — Google (2014) | https://arxiv.org/abs/1301.3781 |
| TinyStories Paper (ArXiv) | https://arxiv.org/pdf/2305.07759 |
| From Words to Vectors — Vizuara Substack | https://vizuara.substack.com/p/from-words-to-vectors-understanding |

---

## Questions to Revisit
- [ ] How exactly are the Q, K, V projection matrices initialized and what are their dimensions relative to the embedding dimension?
- [ ] What is the difference between **Single-Head** and **Multi-Head** attention — why does splitting into multiple heads help?
- [ ] In the causal mask, what precisely happens numerically: `-∞` → softmax → `e^(-∞) = 0`?
- [ ] Position Embeddings in GPT-2 are learned — but the original Transformer used fixed **sinusoidal encodings**. What are the tradeoffs between learned vs fixed?
- [ ] The attention matrix is `(B, H, T, T)` — how does memory scale as context length grows and why is this the core challenge of long-context models?
- [ ] What does the **Output Block** look like — how does the final 768-dim vector get projected back to 50,257 vocabulary logits to produce the final next-token prediction?
