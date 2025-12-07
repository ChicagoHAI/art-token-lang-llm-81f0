# Downloaded Papers

This directory contains research papers on artificial token languages, efficient tokenization, and alternatives to traditional subword tokenization for LLMs.

## Core Papers on Byte-Level and Token-Free Models

### 1. Byte Latent Transformer: Patches Scale Better Than Tokens
- **File:** `2412.09871_byte_latent_transformer.pdf`
- **Authors:** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et al. (Meta AI)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2412.09871
- **Why relevant:** Proposes a tokenizer-free architecture that learns from raw bytes using dynamically-sized patches based on entropy. Scales up to 8B parameters on 4T bytes, matching tokenization-based LLM performance with improved efficiency.
- **Key contribution:** Dynamic patching strategy that adapts to content complexity
- **Code:** Not publicly available

### 2. SpaceByte: Towards Deleting Tokenization from Large Language Modeling
- **File:** `2404.14408_spacebyte.pdf`
- **Authors:** Kevin Slagle
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2404.14408
- **Why relevant:** Byte-level decoder that closes the performance gap with subword models by inserting larger transformer blocks after specific bytes (like spaces).
- **Key contribution:** Hierarchical processing at byte level with selective computation
- **Code:** Not publicly available

### 3. MrT5: Dynamic Token Merging for Efficient Byte-level Language Models
- **File:** `2410.20771_mrt5.pdf`
- **Authors:** Julie Kallini, Shikhar Murty, Christopher D. Manning, et al.
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2410.20771
- **Why relevant:** Integrates token deletion mechanism to dynamically shorten sequences by up to 75% with minimal performance loss. Directly addresses efficiency of byte-level models.
- **Key contribution:** Learned gating mechanism for token merging
- **Code:** Not publicly available

### 4. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations
- **File:** `2406.19223_tfree.pdf`
- **Authors:** Björn Deiseroth, Manuel Brack, Patrick Schramowski, et al.
- **Year:** 2024 (revised 2025)
- **arXiv:** https://arxiv.org/abs/2406.19223
- **Why relevant:** Directly embeds words through sparse activation patterns over character triplets without reference corpora. Achieves 85% parameter reduction on embedding layers.
- **Key contribution:** Corpus-free sparse character triplet embeddings
- **Code:** Not publicly available

### 5. Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing
- **File:** `2501.10322_hierarchical_transformers.pdf`
- **Authors:** Pit Neitemeier, Björn Deiseroth, Constantin Eichenberg, Lukas Balles
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2501.10322
- **Why relevant:** Combines character-level and word-level processing, matching tokenizer performance while showing greater robustness. Tested up to 7B parameters.
- **Key contribution:** Hybrid architecture with character encoders/decoders and word-level backbone
- **Code:** Not publicly available

## Papers on Tokenization Theory and Optimization

### 6. Tokenization Is More Than Compression
- **File:** `2402.18376_tokenization_more_than_compression.pdf`
- **Authors:** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, et al.
- **Year:** 2024 (EMNLP 2024)
- **arXiv:** https://arxiv.org/abs/2402.18376
- **Why relevant:** Challenges the assumption that compression drives tokenizer effectiveness. Introduces PathPiece and tests whether fewer tokens improve performance (they don't).
- **Key contribution:** Empirical evidence that compression ≠ performance; 64 trained models released
- **Code:** Models publicly available (check paper)

### 7. Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling
- **File:** `2501.16975_over_tokenized_transformer.pdf`
- **Authors:** Hongzhi Huang, Defa Zhu, Banggu Wu, et al.
- **Year:** 2025 (ICML 2025)
- **arXiv:** https://arxiv.org/abs/2501.16975
- **Why relevant:** Decouples input and output vocabularies, showing that larger input vocabularies improve performance via log-linear relationship. Achieves 2.5× effective scaling.
- **Key contribution:** Separate optimization of input vs output vocabulary; 128× larger vocabularies with <5% overhead
- **Code:** Not publicly available

### 8. From Language Models over Tokens to Language Models over Characters
- **File:** `2412.03719_tokens_to_characters.pdf`
- **Authors:** Tim Vieira, Ben LeBrun, Mario Giulianelli, et al.
- **Year:** 2024 (ICML 2025)
- **arXiv:** https://arxiv.org/abs/2412.03719
- **Why relevant:** Provides algorithms to convert token-level models to character-level distributions, addressing the mismatch between internal token processing and user character strings.
- **Key contribution:** Exact and approximate conversion algorithms; improved compression rates
- **Code:** Not publicly available

### 9. Training LLMs over Neurally Compressed Text
- **File:** `2404.03626_neurally_compressed_text.pdf`
- **Authors:** Brian Lester, Jaehoon Lee, Alex Alemi, et al. (Google DeepMind)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2404.03626
- **Why relevant:** Investigates training on highly compressed text using neural compression. Proposes Equal-Info Windows for effective learning. Shows potential but underperforms subword tokenizers.
- **Key contribution:** Analysis of learnability in compressed representations
- **Code:** Not publicly available

## Additional Papers (Previously Downloaded)

### 10. BPE is Suboptimal for Language Model Pretraining
- **File:** `2004.03720_bpe_suboptimal.pdf`
- **arXiv:** https://arxiv.org/abs/2004.03720
- **Why relevant:** Early critique of BPE for pretraining

### 11. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization
- **File:** `2106.12672_charformer.pdf`
- **arXiv:** https://arxiv.org/abs/2106.12672
- **Why relevant:** Character-level transformer with learned subword boundaries

### 12. Rethinking Tokenization
- **File:** `2403.00417_rethinking_tokenization.pdf`
- **arXiv:** https://arxiv.org/abs/2403.00417
- **Why relevant:** General critique of current tokenization approaches

### 13. Unpacking Tokenization
- **File:** `2403.06265_unpacking_tokenization.pdf`
- **arXiv:** https://arxiv.org/abs/2403.06265
- **Why relevant:** Comprehensive analysis of tokenization effects

### 14. A Theory of Tokenization
- **File:** `2404.08335_theory_of_tokenization.pdf`
- **arXiv:** https://arxiv.org/abs/2404.08335
- **Why relevant:** Theoretical foundations of tokenization

### 15. Retok: Efficient Tokenizer
- **File:** `2410.04335_retok_efficient_tokenizer.pdf`
- **arXiv:** https://arxiv.org/abs/2410.04335
- **Why relevant:** Proposes more efficient tokenization approach

## Summary Statistics

- **Total Papers:** 15
- **Focus Areas:**
  - Byte-level and character-level models: 5 papers
  - Tokenization theory and optimization: 4 papers
  - Alternative tokenization methods: 6 papers
- **Publication Years:** 2020-2025 (majority from 2024-2025)
- **Key Venues:** EMNLP 2024, ICML 2025

## Key Themes

1. **Byte-level processing** is gaining traction as a viable alternative to tokenization
2. **Compression does not equal performance** - more tokens doesn't necessarily hurt
3. **Vocabulary scaling** shows log-linear improvements (especially for input vocab)
4. **Hierarchical architectures** combine benefits of character and word-level processing
5. **Learned dynamic representations** (patches, merging) outperform fixed tokenization
