# Literature Review: Artificial Token Languages for More Efficient LLMs

## Research Area Overview

Training large language models on artificial, compact token languages represents an emerging approach to improving LLM efficiency without sacrificing performance. The core insight is that traditional tokenization (BPE, WordPiece) is a fixed, heuristic preprocessing step that may not optimally allocate compute. By working directly with bytes and learning dynamic representations, models can potentially achieve better compression-performance trade-offs.

This literature review synthesizes recent research (2024-2025) on byte-level language models, alternative tokenization approaches, and vocabulary optimization strategies.

---

## Key Papers

### 1. Byte Latent Transformer: Patches Scale Better Than Tokens
- **Authors:** Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, et al. (Meta AI)
- **Year:** 2024
- **Source:** arXiv:2412.09871
- **File:** `papers/2412.09871_byte_latent_transformer.pdf`

#### Key Contribution
First byte-level architecture to match tokenization-based LLM performance at scale (up to 8B parameters, 4T bytes). Introduces **entropy-based dynamic patching** that allocates compute based on next-byte uncertainty.

#### Methodology
- **Architecture:** Three-component model
  1. Local Encoder (lightweight, byte-level)
  2. Global Latent Transformer (heavy computation, patch-level)
  3. Local Decoder (lightweight, generates bytes)
- **Patching:** Dynamic grouping based on entropy model predictions
- **Hash N-gram Embeddings:** Capture byte context without fixed vocabulary
- **Cross-attention:** Pools byte representations into patches

#### Datasets Used
- Llama 2 dataset (2T tokens)
- BLT-1T (1T tokens from various public sources)
- Evaluated on: WikiText, C4, GitHub, ARC, HellaSwag, PIQA, MMLU, MBPP, HumanEval

#### Results
- Matches Llama 3 training FLOP-controlled performance
- Up to 50% fewer inference FLOPs at equivalent performance
- Significantly better on:
  - Noisy inputs (HellaSwag variants)
  - Character-level tasks (CUTE benchmark: 54.1% vs 27.5%)
  - Low-resource translation (FLORES-101)

#### Code Available
Yes: https://github.com/facebookresearch/blt

#### Relevance to Our Research
**Critical.** This is the state-of-the-art for byte-level LLMs. Demonstrates that:
1. Byte-level models CAN match token-based models at scale
2. Dynamic patching outperforms fixed tokenization
3. Learned representations (hash n-grams) beat fixed vocabularies
4. Model size and patch size can be scaled simultaneously with fixed inference budget

---

### 2. SpaceByte: Towards Deleting Tokenization from Large Language Modeling
- **Authors:** Kevin Slagle
- **Year:** 2024
- **Source:** arXiv:2404.14408
- **File:** `papers/2404.14408_spacebyte.pdf`

#### Key Contribution
Byte-level decoder that closes performance gap with subword models by inserting larger transformer blocks after space-like bytes (whitespace, punctuation).

#### Methodology
- Patches on natural linguistic boundaries (spaces)
- Hierarchical processing: more compute after important bytes
- No fixed vocabulary

#### Results
- Outperforms MEGABYTE (fixed-stride patching)
- Roughly matches tokenized transformers within fixed compute budgets
- Simpler than BLT (no cross-attention or hash embeddings)

#### Code Available
Not explicitly stated

#### Relevance to Our Research
Shows that **simple rule-based patching** (spaces) can be effective. Demonstrates importance of aligning patches with semantic boundaries rather than arbitrary byte positions.

---

### 3. MrT5: Dynamic Token Merging for Efficient Byte-level Language Models
- **Authors:** Kallini, Murty, Manning, Potts, Csordás
- **Year:** 2024
- **Source:** arXiv:2410.20771
- **File:** `papers/2410.20771_mrt5.pdf`

#### Key Contribution
Integrates **learned deletion mechanism** in encoder to dynamically shorten sequence length. Reduces sequences by up to 75% with minimal performance loss.

#### Methodology
- Encoder-decoder architecture (T5-style)
- Learned gating mechanism decides which tokens to keep/merge
- Dynamic compression adapts to content complexity

#### Results
- Significant gains in inference runtime
- Minimal effect on downstream performance (XNLI, TyDi QA)
- 75% sequence length reduction possible

#### Code Available
Not explicitly stated

#### Relevance to Our Research
Alternative approach to dynamic compression. Instead of deciding when to invoke expensive computation (like BLT), decides which information to preserve. Could be complementary to artificial token design.

---

### 4. Tokenization Is More Than Compression
- **Authors:** Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter, Tanner
- **Year:** 2024 (EMNLP)
- **Source:** arXiv:2402.18376
- **File:** `papers/2402.18376_tokenization_more_than_compression.pdf`

#### Key Contribution
**Challenges conventional wisdom** that compression drives tokenizer effectiveness. Introduces PathPiece (minimal-token segmenter) and shows fewer tokens ≠ better performance.

#### Methodology
- Trained 64 language models (350M-2.4B parameters)
- Varied tokenization across all three phases:
  1. Pre-tokenization
  2. Vocabulary construction
  3. Segmentation
- PathPiece: segments text into minimal tokens for given vocabulary

#### Key Findings
1. **Compression does not equal performance**
2. Pre-tokenization is most important phase
3. BPE vocabulary initialization provides benefits
4. Fewer tokens doesn't improve downstream tasks

#### Datasets Used
- Standard pretraining corpora
- 64 models made publicly available

#### Code Available
Models released (check paper for links)

#### Relevance to Our Research
**Critical insight:** Don't optimize purely for compression. The goal is efficient allocation of model capacity, not minimal token count. This supports BLT's approach of dynamic patching based on complexity rather than fixed compression ratio.

---

### 5. Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling
- **Authors:** Huang, Zhu, Wu, Zeng, Wang, Min, Zhou
- **Year:** 2025 (ICML)
- **Source:** arXiv:2501.16975
- **File:** `papers/2501.16975_over_tokenized_transformer.pdf`

#### Key Contribution
Decouples input and output vocabularies. Shows that **larger input vocabularies** improve performance via log-linear relationship with training loss.

#### Methodology
- Separate optimization of embedding (input) vs unembedding (output)
- Multi-gram tokens in input vocabulary
- Input vocab can be 128× larger with <5% overhead

#### Results
- 400M parameter model with 12.8M input vocab matches 1B baseline
- 2.5× effective scaling
- 3-5× reduction in training steps for convergence
- Memory overhead <5% despite 128× larger vocab

#### Relevance to Our Research
Complements artificial token language idea. Suggests that:
1. Input and output should be optimized separately
2. Large input vocabularies don't hurt (unlike conventional wisdom)
3. Multi-gram representations help

Could inform design of artificial language: optimize expressiveness of input tokens separately from output generation.

---

### 6. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations
- **Authors:** Deiseroth, Brack, Schramowski, Kersting, Weinbach
- **Year:** 2024 (revised 2025)
- **Source:** arXiv:2406.19223
- **File:** `papers/2406.19223_tfree.pdf`

#### Key Contribution
Embeds words through **sparse activation patterns over character triplets** without reference corpus. Achieves 85% parameter reduction on embedding layers.

#### Methodology
- Character triplet encoding
- Sparse representations
- No fixed vocabulary
- Corpus-independent

#### Results
- Competitive downstream performance
- 85% parameter reduction on embeddings
- Significant improvements in cross-lingual transfer

#### Relevance to Our Research
Alternative to vocabulary-based embeddings. Shows that **learned, sparse representations** can replace fixed vocabularies while improving efficiency and cross-lingual transfer.

---

### 7. Training LLMs over Neurally Compressed Text
- **Authors:** Lester, Lee, Alemi, Pennington, Roberts, Sohl-Dickstein, Constant (Google DeepMind)
- **Year:** 2024
- **Source:** arXiv:2404.03626
- **File:** `papers/2404.03626_neurally_compressed_text.pdf`

#### Key Contribution
Investigates training on **highly compressed text** using neural compression (arithmetic coding). Proposes Equal-Info Windows for uniform compression blocks.

#### Methodology
1. Train byte-level LM (M1) to predict next byte probabilities
2. Use M1 to compress corpus via Arithmetic Coding
3. Chunk compressed bitstream into equal-info windows
4. Train second LM (M2) over compressed representation

#### Results
- Outperforms byte-level baselines
- Faster inference (fewer autoregressive steps)
- Still underperforms subword tokenizers at equivalent parameter counts
- Provides analysis of learnability factors

#### Relevance to Our Research
Shows potential of **learned compression** for tokenization. However, standard neural compression creates hard-to-learn representations. Suggests that compression scheme must be co-designed with learning objective.

---

### 8. From Language Models over Tokens to Language Models over Characters
- **Authors:** Vieira, LeBrun, Giulianelli, Gastaldi, DuSell, Terilla, O'Donnell, Cotterell
- **Year:** 2024 (ICML 2025)
- **Source:** arXiv:2412.03719
- **File:** `papers/2412.03719_tokens_to_characters.pdf`

#### Key Contribution
Provides **algorithms to convert token-level models to character-level distributions**, addressing mismatch between internal token processing and user character strings.

#### Methodology
- Exact and approximate conversion algorithms
- Converts token-level probabilities to character-level
- Evaluated on 4 public models

#### Results
- Accurately approximates character-level distribution
- Improves compression rates (bits per byte)
- Faster than direct character-level modeling

#### Relevance to Our Research
Addresses practical issue: users interact with characters, models use tokens. Conversion algorithms could be useful for evaluating artificial token languages against character-level baselines.

---

### 9. Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing
- **Authors:** Neitemeier, Deiseroth, Eichenberg, Balles
- **Year:** 2025
- **Source:** arXiv:2501.10322
- **File:** `papers/2501.10322_hierarchical_transformers.pdf`

#### Key Contribution
**Hybrid architecture** with character encoders/decoders and word-level backbone. Matches tokenizer performance with greater robustness. Tested up to 7B parameters.

#### Methodology
- Lightweight character-level encoder → word representations
- Word-level transformer backbone
- Character-level decoder → output bytes
- No fixed vocabulary

#### Results
- Matches traditional tokenizer performance
- Significantly greater robustness to input perturbations
- Faster training on new languages

#### Relevance to Our Research
Shows benefits of **hierarchical processing**: lightweight byte/char processing with powerful word-level reasoning. Could inform multi-scale artificial token design.

---

## Common Methodologies Across Papers

### Patching Strategies
1. **Fixed-stride** (MEGABYTE) - Simple but suboptimal
2. **Space-based** (SpaceByte) - Aligns with linguistic boundaries
3. **Entropy-based** (BLT) - Adapts to content complexity
4. **Learned gating** (MrT5) - Model decides what to compress

### Architectural Patterns
1. **Local-Global split** (BLT, MEGABYTE, Hierarchical) - Lightweight byte processing + powerful patch/word processing
2. **Cross-attention** (BLT, Hierarchical) - Pool byte info into higher-level representations
3. **Hash embeddings** (BLT) - Vocabulary-free context encoding
4. **Sparse activations** (T-FREE) - Efficient representation without fixed vocab

### Evaluation Metrics
1. **Bits-per-byte (BPB)** - Tokenizer-independent perplexity
2. **Downstream tasks** - Standard NLP benchmarks (GLUE, SuperGLUE, etc.)
3. **Robustness** - Noisy input variants
4. **Character-level understanding** - CUTE benchmark, orthographic tasks
5. **Multilingual** - FLORES for translation, token count ratios across languages

---

## Standard Baselines

From the literature, standard baselines for comparison:

### Token-based Baselines
1. **BPE** (GPT-2, GPT-3)
   - Vocab: 50k-100k tokens
   - Average: 3-4 bytes/token

2. **WordPiece** (BERT)
   - Vocab: 30k tokens
   - Balanced linguistic/compression trade-off

3. **SentencePiece** (T5, mT5)
   - Vocab: 32k tokens
   - Language-independent

4. **Llama 2 Tokenizer**
   - Vocab: 32k tokens
   - 3.7 bytes/token average

5. **Llama 3 Tokenizer**
   - Vocab: 128k tokens
   - 4.4 bytes/token average
   - Better multilingual coverage

### Byte-level Baselines
1. **MEGABYTE**
   - Fixed-stride patching
   - Patch size: 4-8 bytes

2. **ByT5**
   - No patching (processes every byte)
   - Very expensive but simple

3. **Character-level Transformers**
   - Process individual characters
   - Long sequence lengths

---

## Datasets in the Literature

### Language Modeling
1. **WikiText-103** - Standard LM benchmark
2. **enwiki8** - Byte-level standard (100M bytes)
3. **text8** - Character-level variant (27 chars)
4. **The Pile** - Diverse 825GB corpus
5. **C4** - Cleaned Common Crawl (750GB)

### Multilingual
1. **FLORES-101/200** - 101-200 languages, parallel translation
2. **mC4** - Multilingual Common Crawl
3. **OSCAR** - Multilingual web corpus

### Evaluation Benchmarks
1. **CUTE** - Character-level understanding
2. **HellaSwag** (+ noisy variants) - Commonsense + robustness
3. **MMLU** - Massive multitask understanding
4. **HumanEval / MBPP** - Code generation
5. **ARC-Easy/Challenge** - Reasoning
6. **PIQA** - Physical commonsense

---

## Gaps and Opportunities

### Identified Gaps

1. **Limited exploration of artificial grammars**
   - Most work uses bytes or existing tokens
   - Little work on designing optimal symbolic systems

2. **No systematic vocabulary optimization**
   - BPE/WordPiece use compression heuristics
   - Over-Tokenized Transformer scales vocab but doesn't redesign it

3. **Patching is still heuristic**
   - Even BLT's entropy-based patching uses separate entropy model
   - Not end-to-end optimized

4. **Multilingual inequity persists**
   - Latin scripts still favored
   - Low-resource languages require more tokens

5. **Theory-practice gap**
   - "Tokenization Is More Than Compression" shows compression ≠ performance
   - But unclear what DOES make good tokenization

### Research Opportunities

1. **Design artificial token languages**
   - Optimize symbols for LLM learning (not human readability)
   - Could have uniform complexity across "words"
   - Information-theoretically optimal encoding

2. **End-to-end learned tokenization**
   - Don't separate tokenization from model training
   - Joint optimization of representation and model

3. **Multilingual-first design**
   - Equal efficiency across scripts
   - Universal byte-to-concept mapping

4. **Compression-performance theory**
   - Formal characterization of good tokenization
   - Beyond heuristics

5. **Scaling laws for byte-level models**
   - BLT assumes Llama 3 scaling laws apply
   - May need different compute-optimal ratios

---

## Recommendations for Our Experiment

Based on the literature review:

### Recommended Datasets

**Primary (Must-have):**
1. **enwiki8** (100M bytes) - Standard byte-level benchmark, enables direct comparison
2. **FLORES-200** - Multilingual evaluation, test efficiency across scripts
3. **CUTE** - Character-level understanding, critical for byte-level models

**Secondary (Important):**
4. **WikiText-103** - Standard LM benchmark
5. **C4 sample** - Web text diversity
6. **HellaSwag** - Robustness testing (create noisy variants)

**Tertiary (If resources permit):**
7. **The Pile (subset)** - Domain diversity
8. **MMLU** - Downstream evaluation

### Recommended Baselines

**Must compare against:**
1. **BLT** - Current state-of-the-art byte-level
2. **Llama 3 BPE** - Strong token-based baseline
3. **SentencePiece BPE** - Standard production tokenizer

**Secondary baselines:**
4. **MEGABYTE** - Fixed-stride patching
5. **Character-level Transformer** - Naive byte processing

### Recommended Metrics

**Primary:**
1. **Bits-per-byte (BPB)** - Tokenizer-independent performance
2. **Training FLOPs** - Efficiency during training
3. **Inference FLOPs** - Efficiency during generation
4. **Tokens-per-byte** - Compression ratio (but remember: compression ≠ performance!)

**Secondary:**
5. **Downstream task accuracy** - ARC, HellaSwag, MMLU
6. **Robustness** - Performance on noisy inputs
7. **Character-level accuracy** - CUTE benchmark
8. **Multilingual equity** - Token count ratios across languages (FLORES)

### Methodological Considerations

1. **FLOP-controlled experiments** are essential
   - Don't just compare model sizes
   - Control for training and inference compute

2. **Separate training and inference efficiency**
   - A model might be slow to train but fast at inference (or vice versa)

3. **Test on diverse data**
   - Language modeling alone isn't enough
   - Need downstream tasks, robustness, multilingual

4. **Consider scaling trends**
   - What works at 1B parameters may not work at 8B+
   - Plot scaling curves, not just final performance

5. **Document everything**
   - Tokenization details
   - Training hyperparameters
   - Data preprocessing
   - Enables reproducibility

---

## Key Insights for Artificial Token Language Design

Based on the literature:

1. **Dynamic > Static**
   - BLT's dynamic patching beats fixed tokenization
   - Allocate compute based on complexity, not fixed rules

2. **Learned > Heuristic**
   - Hash n-grams (BLT) beat fixed vocabularies
   - Neural compression shows promise but needs better design

3. **Compression ≠ Performance**
   - Don't optimize solely for minimal tokens
   - Goal is efficient capacity allocation

4. **Vocabulary Scaling Helps**
   - Larger input vocabularies show log-linear improvements
   - Input and output vocabs should be optimized separately

5. **Byte-level is Viable**
   - BLT proves byte-level can match tokens at scale
   - Robustness and multilingual benefits

6. **Hierarchical Processing Works**
   - Local-global split is effective
   - Lightweight byte processing + powerful reasoning

7. **Cross-lingual Transfer Matters**
   - T-FREE shows sparse representations help transfer
   - Design for universality, not English optimization

---

## Open Questions

1. What makes a "good" tokenization beyond compression?
2. Can we design tokens that are easier for LLMs to learn?
3. What's the information-theoretic optimum for LLM tokenization?
4. How should artificial tokens differ from natural language?
5. Can we achieve true multilingual parity in tokenization?

---

## Conclusion

The literature shows a clear trend away from fixed, heuristic tokenization toward learned, dynamic representations. BLT represents the current state-of-the-art, but significant opportunities remain:

- **Design artificial languages** optimized for LLM learning
- **End-to-end optimization** of representation and model
- **Multilingual-first** approach for equity
- **Theory** of what makes good tokenization

Your research hypothesis—that compact, artificial token languages can improve LLM efficiency—is well-supported by recent trends. The key is to go beyond compression and optimize for **learnability** and **efficient capacity allocation**.

