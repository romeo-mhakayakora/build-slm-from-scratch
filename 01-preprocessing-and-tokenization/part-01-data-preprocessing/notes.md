# Part 01 — Data Pre-processing
### Build a Small Language Model From Scratch
**Course by:** Vizuara (Dr. Raj Dandekar)
**Lecture video:** [YouTube — Part 1: Data Pre-processing](https://youtu.be/KOXaH7-0gEE?si=O9zq91_TH3wH08aB)
**Core paper:** [TinyStories — ArXiv 2305.07759](https://arxiv.org/pdf/2305.07759)
**Colab notebook:** [Google Colab](https://colab.research.google.com/drive/1k4G3G5MxYLxawmPfAknUN7dbbmyqIdQv#scrollTo=vFkgAjyMR8fa)

---

## Table of Contents
1. [First Principles — The "Alien" Thought Experiment](#1-first-principles--the-alien-thought-experiment)
2. [What is a Large Language Model (LLM)?](#2-what-is-a-large-language-model-llm)
3. [Why Must LLMs Be So "Large"?](#3-why-must-llms-be-so-large)
4. [The Shift to Small Language Models (SLMs)](#4-the-shift-to-small-language-models-slms)
5. [What Actually is a "Parameter"?](#5-what-actually-is-a-parameter)
6. [The Dataset Strategy — TinyStories](#6-the-dataset-strategy--tinystories)
7. [Data Pre-processing Phase 1 — Tokenization](#7-data-pre-processing-phase-1--tokenization)
8. [Sub-word Tokenization & Byte Pair Encoding (BPE)](#8-sub-word-tokenization--byte-pair-encoding-bpe)
9. [Data Engineering — Writing Tokens to Disk](#9-data-engineering--writing-tokens-to-disk)
10. [The Core Task — Creating Input / Output Pairs](#10-the-core-task--creating-input--output-pairs)
11. [The Mechanics of Training — Tuning the Parameters](#11-the-mechanics-of-training--tuning-the-parameters)
12. [Key Terminology](#12-key-terminology)
13. [Final Insights & Big Picture](#13-final-insights--big-picture)

---

## 1. First Principles — The "Alien" Thought Experiment

**Setup:** Imagine a highly intelligent alien who knows mathematics but zero English. They are given a massive stack of English books and asked to complete the sentence: *"Your effort will take you [blank]"*.

### The Statistical / Frequency Approach
- The alien's first instinct is to search the dataset for the exact phrase and count which word follows it most often.

**Why this approach fails in practice:**

**Problem 1 — Computationally Intractable:**
- English vocabulary ≈ 40,000 words.
- Predicting 1 word ahead → 40,000 options.
- Predicting 2 words → 40,000 × 40,000 combinations.
- Predicting 3 words → 1.6 trillion options.
- Predicting 20 words → more combinations than there are particles in the universe.
- Pure frequency counting scales impossibly.

**Problem 2 — Out-of-Distribution Sentences:**
- If the exact sentence sequence has never appeared in the dataset, the frequency approach completely breaks down — there is no prior data to count.

### The Physics Analogy (Galileo's Leaning Tower)
- Dropping objects to observe descent times gives you limited observational data points.
- But deriving a mathematical model: **t = √(2h / g)** lets you predict descent time for *any* height, including unobserved ones.
- **Takeaway:** We cannot statistically predict the next word using pure frequency. We need to build a *model* that takes an input sequence and mathematically estimates the probability distribution of what the next word should be.

---

## 2. What is a Large Language Model (LLM)?

An LLM is exactly that mathematical engine. It has **three defining characteristics:**

**① Next Token Prediction Engine**
- Takes a sequence of tokens (words or sub-words) and predicts what token comes next.

**② Probabilistic Nature**
- Output is not deterministic.
- The model generates a **probability distribution** over all possible next tokens.
- It typically selects the token with the highest probability.
- It generates text **one single token at a time**.

**③ Billions of Parameters**
- Parameters are the "tunable knobs" of the model.
- A simple linear equation (y = ax + b) has 2 parameters.
- GPT-3 → 175 billion parameters.
- GPT-4 → estimated 1 trillion parameters.

---

## 3. Why Must LLMs Be So "Large"?

Scaling up the parameter count and computational compute solves two major problems:

### A. Emergent Properties (The Technical Reason)
- As model size and computational power scale, model performance improves.
- At specific **"takeoff points,"** models suddenly acquire complex capabilities they were **never explicitly trained for** — e.g., arithmetic, logical reasoning, language translation.
- Tech companies build massive models hoping to unlock even more advanced emergent properties, pushing toward **Artificial General Intelligence (AGI)**.
- https://cset.georgetown.edu/article/emergent-abilities-in-large-language-models-an-explainer/

### B. Learning Form and Meaning (The Intuitive Reason)
LLMs are only trained to predict the next token. We **never** explicitly teach them grammar, sentence structure, or real-world facts.

- **Form:** Through the brute-force repetition of next-token prediction, the model implicitly learns English grammar (e.g., subject-verb-object syntax).
- **Meaning:** It learns a "world model." For example, the sentence *"Blue electrons eat fish"* is grammatically flawless, but an LLM will not generate it because it mathematically understands that this sequence carries no logical meaning in reality.
- Absorbing the vast, unstructured nuances of both human grammar and real-world logic requires an **immense parameter count**.

---

## 4. The Shift to Small Language Models (SLMs)

**Traditional LLM Philosophy:** To learn language comprehensively, you need all the text data on the internet and 100+ billion parameters to process it.

**SLMs challenge this notion.**

### The SLM Hypothesis
- If an enterprise doesn't need a model to know *everything* in the world (astrophysics, Reddit threads, cooking recipes), can they build a highly capable model strictly for a specific domain?

### The Solution
- By using a **smartly constructed, domain-specific dataset**, you can still teach a model the form and meaning of language without the massive overhead of generalized data.

### Target Size
- An SLM typically operates in the range of **10 million to 100 million parameters** — a roughly **1,000× reduction** in size compared to massive frontier LLMs.

---

## 5. What Actually is a "Parameter"?

- Think of them as **"tunable knobs"** in a black-box architecture.
- In linear regression (y = ax + b), `a` and `b` are the 2 parameters.
- In an SLM/LLM, imagine millions of nested mathematical functions:
  ```
  y = f(g(h(...(x))))
  ```
- The **weights within these nested layers** and **token embeddings** are the parameters.
- Training = the process of tuning all these knobs so the model's output matches the desired target.

---

## 6. The Dataset Strategy — TinyStories

To teach a model English grammar and logic, you don't need a massive corpus of complex Reddit threads or PhD astrophysics papers. You just need **representative data**.

### The TinyStories Paper
> *"How small can language models be and still speak coherent English?"* — Microsoft Research

- Instead of scraping the web, researchers generated a **synthetic dataset of 2 million short stories** designed specifically for a **3-to-4-year-old reading level**.
- **Why this is brilliant:** A toddler's story still perfectly preserves the essential elements of natural language (grammar, vocabulary, basic facts, and reasoning), but severely limits breadth and complexity.
- By training an SLM strictly on these 2 million rows, the model learns the "rules" of English without needing billions of parameters to memorize random world trivia.

### Applying This to the Enterprise (e.g., Healthcare, Finance)
- You can apply this exact logic to domain-specific SLMs (e.g., JP Morgan training a model on internal financial data).
- **Requirement:** You need roughly **1 million to 10 million rows** of domain-specific textual data.
- **Critical distinction:** This must be **raw text / context**, not just input/output Q&A pairs — the model still needs to learn the structural language of that specific domain.

---

## 7. Data Pre-processing Phase 1 — Tokenization

**The fundamental problem:** Computers do not understand text; they only understand numbers.

**Tokenization defined:** Taking raw text, breaking it down into smaller chunks, and assigning a unique numerical **Token ID** to each chunk via a vocabulary dictionary.

---

### Method 1: Word-Based Tokenization

**Mechanism:** Every unique word gets its own Token ID.
- e.g., Token 1 = "this", Token 2 = "is", Token 3 = "a" …

**Flaw 1 — Computational Latency:**
- English vocabulary ≈ 600,000 words.
- A vocabulary dictionary this large requires massive processing time during the model's next-token decoding phase.

**Flaw 2 — Out of Vocabulary (OOV) Problem:**
- If the model encounters a typo in the real world (e.g., `"univeristy"` instead of `"university"`), it will not find a matching Token ID in its vocabulary list.
- The model breaks / fails on that word entirely.

---

### Method 2: Character-Based Tokenization

**Mechanism:** Every single letter / character becomes its own token.
- e.g., `"C"`, `"A"`, `"T"` — each is a separate token.

**Advantage:**
- Vocabulary size shrinks drastically (down to 26 English letters or 255 ASCII characters).
- Completely solves the OOV problem — any misspelled word is just a sequence of known characters.

**Fatal Flaw 1 — Destroys Linguistic Essence:**
- Words carry inherent meaning and structure (e.g., "cat" vs "kitten").
- Breaking language down to mere letters strips away the structural advantages of text, making it incredibly hard for the model to learn semantic relationships.

**Fatal Flaw 2 — Context Window Explosion:**
- If every letter is a token, the number of tokens required to represent a single sentence skyrockets.
- The model will quickly run out of "memory" (its Context Window) before it can process a meaningful amount of text.

---

## 8. Sub-word Tokenization & Byte Pair Encoding (BPE)

Sub-word tokenization offers the **best of both worlds** — it avoids the OOV problem by keeping individual characters as a fallback, but aggressively reduces sequence length by grouping common letter combinations.

### Byte Pair Encoding (BPE) — The Algorithm

**Step 1:** Start at the character level. Look at all individual characters in the text and count their frequencies.

**Step 2 — Merge Frequent Pairs:** Find the pair of adjacent characters that appear most frequently together (e.g., `"t"` and `"h"`) and merge them into a brand new token: `"th"`.

**Step 3 — Iterate:** Repeat this process over and over. Common suffixes like `-est`, `-tion`, `-ly`, and highly frequent whole words eventually become their own single tokens.

**The Result:**
- Highly common words / sub-words get dedicated tokens.
- Rare words are pieced together using smaller sub-words or single characters.
- This optimizes vocabulary size perfectly.
- Example: The **GPT-2 tokenizer** uses a vocabulary size of exactly **50,257**.
- https://vizuara.substack.com/p/understanding-byte-pair-encoding


### Tokenization Quirks (Visualized via TikToken)
https://tiktokenizer.vercel.app/

| Quirk | Detail |
|---|---|
| **Case Sensitivity** | `"egg"`, `"Egg"`, and `"EGG"` map to completely different Token IDs |
| **Spacing** | A blank space at the start of a word binds to the word. `" egg"` is a different token than `"egg"` |
| **Numbers** | Tokenizers often slice numbers at arbitrary places — this is the foundational reason why LLMs are notoriously bad at mathematics |


---

## 9. Data Engineering — Writing Tokens to Disk

After passing 2 million rows of TinyStories through the BPE tokenizer, you are left with a **massive array of numbers (Token IDs)**. This cannot be kept in active memory (RAM).

### Batching & Disk Storage
- The array of token IDs is split into batches (e.g., **80% training / 20% validation**).
- These batches are written sequentially directly to the hard drive as **binary files** (`train.bin` and `val.bin`).

### `np.memmap` — Memory Mapping (Critical Production Trick)
- Allows the script to read/write massive arrays to the disk **as if they were in RAM**, without actually crashing the computer's memory.
- Essential for working with datasets that exceed available RAM.

### Special Tokens (Production Note)
- Though not explicitly used in this toy model, production models append **Special Tokens** like `<EOS>` (End of Sequence) between stories.
- This lets the model mathematically know when one context ends and a new one begins.

---

## 10. The Core Task — Creating Input / Output Pairs

This is the most critical concept linking token IDs to actual LLM training. The goal is **Next Token Prediction** — we must transform our raw data stream into a format that teaches this.

### Setting the Parameters

| Parameter | Description | Example Value |
|---|---|---|
| **Context Window** | Maximum number of tokens the model is allowed to "look at" at one time | 4 tokens |
| **Batch Size** | Number of independent sequences processed simultaneously | 4 sequences |

### The "Shift by One" Mechanism

Take a sample sequence of 5 tokens from the data:
```
"one"  "day"  "a"  "little"  "girl"
```

- **Input Tensor (x):** Take the first 4 tokens → `[one, day, a, little]`
- **Target Tensor (y):** Shift the input right by 1 → `[day, a, little, girl]`

### Why This is Brilliant — 4 Tasks in 1

Although this looks like a single input/output pair, it forces the model to perform **4 simultaneous next-token prediction tasks:**

| Input seen so far | Target |
|---|---|
| `"one"` | `"day"` |
| `"one day"` | `"a"` |
| `"one day a"` | `"little"` |
| `"one day a little"` | `"girl"` |

### How Training Works
1. Initially, the 10–100 million parameters are **randomized** — the model might predict `"rainbow"` instead of `"day"`.
2. Over thousands of batches, **gradient descent** tweaks the parameters so the predictions begin to match the shifted target tensors.
3. This repetitive mathematical matching is how the model **implicitly absorbs English grammar and logic** — without ever being told the rules explicitly.

---

## 11. The Mechanics of Training — Tuning the Parameters

### The Prediction Gap
- Initially, parameters are completely random.
- When fed the input `"one day a little"`, the model might predict a mathematically random token like `"rainbow"`.
- The ground truth target is `"girl"`.
- The objective of training is to use **gradient descent and backpropagation** to tweak the model's parameters until predictions consistently match the ground truth.

### Batch Construction

A batch is a **matrix** of input / output pairs:
- **Rows** = Batch Size (number of independent sequences processed at once)
- **Columns** = Context Window (number of tokens in each sequence)

### Random Sampling Strategy
- Input sequences are selected at **random points** (`torch.randint`) across the dataset, rather than strictly sequentially.
- Random sampling drastically improves the model's **generalization ability** (how well it performs on unseen data).

---

## 12. Key Terminology

### Self-Supervised Learning
- We do **not** manually label this dataset.
- The model generates its own "labels" (targets) simply by taking the input sequence and shifting it to the right by one token.
- **The data supervises itself.**

### Autoregressive
- The output of one step becomes part of the input for the next step.
- Example: If input `"one"` predicts `"day"`, the new input sequence becomes `"one day"` to predict `"a"`.

---


### Hardware Optimizations Explained

| Optimization | What It Does |
|---|---|
| `pin_memory=True` | Locks the tensor's memory in RAM. Allows significantly faster data transfer from CPU → GPU |
| `non_blocking=True` | Allows the CPU to continue doing other tasks (like preparing the next batch) asynchronously while the current batch is being moved to the GPU |

---

## 13. Final Insights & Big Picture

### The "Leap of Faith" in Data Prep
- It seems mathematically absurd that feeding a computer a giant bag of shifted number arrays teaches it language.
- Yet the simple task of next-token prediction forces the neural network to **reverse-engineer the underlying rules of human grammar, form, and logical meaning**.

### SLMs vs. Emergent Behavior
- Massive models (100B+ parameters) are built to chase **"emergent behaviors"** — magical abilities the model wasn't explicitly trained for.
- **SLMs do not care about emergent behavior.**
- The goal of an SLM is to be **hyper-efficient and highly accurate at one specific task / domain**.

### Context Window Dependency
- The "magic" of an LLM/SLM relies heavily on its **Attention Mechanism** (covered in the next lecture).
- For Attention to work optimally, a full logical thought (like a complete short story) should **ideally fit inside a single Context Window**.
- If a story gets awkwardly split across multiple batches, the model loses context and learns incorrectly.

---

## Resources

| Resource | Link |
|---|---|
| Lecture Video (YouTube) | https://youtu.be/KOXaH7-0gEE?si=O9zq91_TH3wH08aB |
| TinyStories Paper (ArXiv) | https://arxiv.org/pdf/2305.07759 |
| Google Colab Notebook | https://colab.research.google.com/drive/1k4G3G5MxYLxawmPfAknUN7dbbmyqIdQv#scrollTo=vFkgAjyMR8fa |
| Miro Board | https://miro.com/app/board/uXjVIL4LZB0=/ |

---

## Questions to Revisit
- [ ] Why is the GPT-2 vocabulary size exactly 50,257? Is this arbitrary or mathematically optimal?
- [ ] What happens to the model's performance if stories are split mid-context across batches?
- [ ] How does `np.memmap` handle concurrent read/write in a multi-GPU training scenario?
- [ ] Why does tokenizing numbers arbitrarily cause math failures — is this fully a tokenization problem or also an architecture problem?
- [ ] What does the Attention Mechanism look like under the hood? (Next lecture)
