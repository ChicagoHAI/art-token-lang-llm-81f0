# Downloaded Papers

This directory contains research papers relevant to the study of artificial token languages for more efficient LLMs.

## Papers Overview

Total papers downloaded: 8

### 1. Toward a Theory of Tokenization in LLMs
- **File**: `2404.08335_theory_of_tokenization.pdf`
- **Authors**: Bostrom et al.
- **Year**: 2024 (April)
- **arXiv**: [2404.08335](https://arxiv.org/abs/2404.08335)
- **Size**: 3.2 MB
- **Why relevant**: Provides theoretical foundation for understanding how tokenization affects transformer behavior. Investigates tokenization from information-theoretic perspective on simple data generating processes. Shows that with tokenization, transformers break through learning barriers and achieve near-optimal probability modeling.

### 2. ReTok: Replacing Tokenizer to Enhance Representation Efficiency in Large Language Models
- **File**: `2410.04335_retok_efficient_tokenizer.pdf`
- **Authors**: Various
- **Year**: 2024 (October)
- **arXiv**: [2410.04335](https://arxiv.org/abs/2410.04335)
- **Size**: 555 KB
- **Why relevant**: Directly addresses tokenizer replacement for improved efficiency. Proposes methods to enhance model representation and processing efficiency while maintaining performance. Shows significant reduction in inference time in specific domains - highly aligned with our research hypothesis.

### 3. Rethinking Tokenization: Crafting Better Tokenizers for Large Language Models
- **File**: `2403.00417_rethinking_tokenization.pdf`
- **Authors**: Various
- **Year**: 2024 (March)
- **arXiv**: [2403.00417](https://arxiv.org/abs/2403.00417)
- **Size**: 611 KB
- **Why relevant**: Proposes the Less-is-Better (LiB) model which autonomously learns integrated vocabulary with subwords, words, and multiword expressions. Effectively reduces both token count and vocabulary types. Demonstrates performance improvements over existing BPE tokenizers - directly supports compact token language approach.

### 4. Byte Pair Encoding is Suboptimal for Language Model Pretraining
- **File**: `2004.03720_bpe_suboptimal.pdf`
- **Authors**: Bostrom & Durrett
- **Year**: 2020
- **arXiv**: [2004.03720](https://arxiv.org/abs/2004.03720)
- **Size**: 344 KB
- **Why relevant**: Critiques standard BPE approach and proposes Unigram LM tokenization as alternative. Shows that alternative tokenization methods can match or outperform BPE. Provides foundation for exploring non-BPE artificial token languages.

### 5. Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance
- **File**: `2403.06265_unpacking_tokenization.pdf`
- **Authors**: Various
- **Year**: 2024 (March)
- **arXiv**: [2403.06265](https://arxiv.org/abs/2403.06265)
- **Size**: 2.8 MB
- **Why relevant**: Demonstrates correlation between tokenizer compression ability and downstream model performance. Controlled study showing compression is reliable intrinsic indicator of tokenization quality - validates our hypothesis that efficient token languages improve model performance.

### 6. Charformer: Fast Character Transformers via Gradient-based Subword Tokenization
- **File**: `2106.12672_charformer.pdf`
- **Authors**: Various
- **Year**: 2021 (June)
- **arXiv**: [2106.12672](https://arxiv.org/abs/2106.12672)
- **Size**: 800 KB
- **Why relevant**: Proposes gradient-based subword tokenization module (GBST) that automatically learns latent subword representations. Achieves 28%-100% speedup over byte-level and subword transformers while maintaining quality. Shows learned tokenization can be more efficient than fixed vocabularies.

### 7. Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling
- **File**: `2501.16975_overtokenized_transformer.pdf`
- **Authors**: Various
- **Year**: 2025 (January - very recent!)
- **arXiv**: [2501.16975](https://arxiv.org/abs/2501.16975)
- **Size**: 3.8 MB
- **Why relevant**: Recent work showing log-linear relationship between input vocabulary size and training loss. Demonstrates larger vocabularies consistently enhance performance. Provides engineering solutions to mitigate computational overhead (< 5% additional cost). Suggests vocabulary optimization is critical efficiency lever.

### 8. Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models
- **File**: `2501.10322_hierarchical_transformers.pdf`
- **Authors**: Various
- **Year**: 2025 (January - very recent!)
- **arXiv**: [2501.10322](https://arxiv.org/abs/2501.10322)
- **Size**: 1.5 MB
- **Why relevant**: Demonstrates hierarchical approach to tokenization combining byte and word-level processing. Shows matched performance to subword tokenizers with greater robustness at scales up to 7B parameters. Explores multi-level token representations as alternative to single-level artificial languages.

## Key Themes Across Papers

### Compression and Efficiency
Papers 2, 3, 5, and 7 directly investigate the relationship between tokenization compression and model efficiency/performance.

### Alternative Tokenization Methods
Papers 4, 6, and 8 propose alternatives to standard BPE tokenization, including Unigram LM, learned GBST, and hierarchical approaches.

### Theoretical Foundations
Paper 1 provides information-theoretic framework for understanding tokenization's role in transformer learning.

### Recent Trends (2024-2025)
Most papers are very recent (2024-2025), indicating this is an active and rapidly evolving research area. Papers 7 and 8 from January 2025 represent the latest thinking on vocabulary scaling and hierarchical tokenization.

## Research Implications

These papers collectively support the hypothesis that:
1. Tokenization significantly impacts model efficiency and performance
2. Standard BPE is not optimal and alternatives exist
3. Compression efficiency correlates with model quality
4. Both learned and designed tokenization approaches show promise
5. Vocabulary optimization is a key leverage point for model efficiency
6. The field is actively exploring artificial/optimized token languages

## Next Steps for Our Research

Based on this literature:
- Consider Unigram LM or learned tokenization (GBST) as baselines
- Investigate compression metrics as evaluation criteria
- Explore multi-level/hierarchical token representations
- Benchmark against both standard BPE and recent efficient alternatives
- Consider vocabulary size as optimization parameter (log-linear relationship)
