# Downloaded Papers

This directory contains research papers relevant to the study of artificial token languages for more efficient LLMs.

## Papers Overview

Total papers downloaded: 20

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

### 9. Byte Latent Transformer: Patches Scale Better Than Tokens
- **File**: `2412.09871_byte_latent_transformer.pdf`
- **Authors**: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et al. (Meta FAIR)
- **Year**: 2024 (December)
- **arXiv**: [2412.09871](https://arxiv.org/abs/2412.09871)
- **Size**: 2.3 MB
- **Why relevant**: BREAKTHROUGH PAPER - First byte-level LLM to match tokenization-based performance at scale. Uses dynamic patches instead of fixed tokens, achieving better inference efficiency. Demonstrates 8B parameter scaling with entropy-based adaptive patching. Shows tokenizer-free models are viable for large-scale deployment.
- **Code**: [GitHub - facebookresearch/blt](https://github.com/facebookresearch/blt)

### 10. MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
- **File**: `2305.07185_megabyte.pdf`
- **Authors**: Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis
- **Year**: 2023 (May)
- **arXiv**: [2305.07185](https://arxiv.org/abs/2305.07185)
- **Size**: 896 KB
- **Why relevant**: Pioneering multi-scale decoder for byte-level modeling. Uses hierarchical patch-based approach (local + global models). Achieves state-of-the-art on ImageNet density estimation and competitive language modeling. Foundation for later byte-level architectures like BLT.

### 11. ByT5: Towards a token-free future with pre-trained byte-to-byte models
- **File**: `2105.13626_byt5.pdf`
- **Authors**: Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, et al. (Google)
- **Year**: 2021
- **arXiv**: [2105.13626](https://arxiv.org/abs/2105.13626)
- **Size**: 476 KB
- **Why relevant**: Google's token-free T5 variant operating on raw bytes. Demonstrates competitive performance with subword models while being more robust to noise and multilingual. Shows byte-level models are practical for production use. Released with full code and pre-trained models.
- **Code**: Available (mentioned in abstract)

### 12. T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations
- **File**: `2406.19223_tfree.pdf`
- **Authors**: Björn Deiseroth, Manuel Brack, Patrick Schramowski, Kristian Kersting, Samuel Weinbach
- **Year**: 2024 (June)
- **arXiv**: [2406.19223](https://arxiv.org/abs/2406.19223)
- **Size**: 2.9 MB
- **Why relevant**: Novel approach using sparse activation patterns over character triplets. Achieves 85%+ reduction in embedding layer parameters. No reference corpus required. Demonstrates tokenizer-free approaches can be highly parameter-efficient while maintaining competitive performance.

### 13. LLM Vocabulary Compression for Low-Compute Environments
- **File**: `2411.06371_vocab_compression.pdf`
- **Authors**: Sreeram Vennam, Anish Joishy, Ponnurangam Kumaraguru
- **Year**: 2024 (November)
- **arXiv**: [2411.06371](https://arxiv.org/abs/2411.06371)
- **Size**: 570 KB
- **Why relevant**: Addresses vocabulary compression by grouping tokens based on BPE merges. Achieves 3.4x memory reduction and 3x throughput improvement. Directly applicable to creating compact artificial token languages for resource-constrained environments.

### 14. Fast Vocabulary Transfer for Language Model Compression
- **File**: `2402.09977_fast_vocab_transfer.pdf`
- **Authors**: Leonidas Gee, Andrea Zugarini, Leonardo Rigutini, Paolo Torroni
- **Year**: 2024 (February)
- **arXiv**: [2402.09977](https://arxiv.org/abs/2402.09977)
- **Size**: 442 KB
- **Why relevant**: Demonstrates vocabulary can be compressed and transferred across domains. Shows vocabulary optimization works in combination with other compression techniques. Validates that learned/optimized vocabularies can replace original tokenization.

### 15. An Efficient Multilingual Language Model Compression through Vocabulary Trimming
- **File**: `2305.15020_vocab_trimming.pdf`
- **Authors**: Asahi Ushio, Yi Zhou, Jose Camacho-Collados
- **Year**: 2023 (May)
- **arXiv**: [2305.15020](https://arxiv.org/abs/2305.15020)
- **Size**: 2.6 MB
- **Why relevant**: Vocabulary-trimming technique achieves ~50% vocabulary reduction while maintaining performance. Shows that many tokens in multilingual models are redundant. Supports hypothesis that compact, well-designed vocabularies are more efficient.

### 16. SpaceByte: Towards Deleting Tokenization from Large Language Modeling
- **File**: `2404.14408_spacebyte.pdf`
- **Authors**: Various
- **Year**: 2024 (April)
- **arXiv**: [2404.14408](https://arxiv.org/abs/2404.14408)
- **Size**: 572 KB
- **Why relevant**: Proposes byte-level model with space-based segmentation boundaries. Addresses tokenization-free modeling with minimal architectural changes. Shows competitive performance on language modeling benchmarks.

### 17. Tokenization Is More Than Compression
- **File**: `2402.18376_tokenization_more_than_compression.pdf`
- **Authors**: Various
- **Year**: 2024 (February)
- **arXiv**: [2402.18376](https://arxiv.org/abs/2402.18376)
- **Size**: 754 KB
- **Why relevant**: Provides deeper analysis of tokenization beyond pure compression. Explores what makes tokenization effective from multiple perspectives. Important for understanding design principles of artificial token languages.

### 18. MrT5: Dynamic Token Merging for Efficient Byte-level Language Models
- **File**: `2410.20771_mrt5.pdf`
- **Authors**: Various
- **Year**: 2024 (October)
- **arXiv**: [2410.20771](https://arxiv.org/abs/2410.20771)
- **Size**: 1020 KB
- **Why relevant**: Extends ByT5 with dynamic token merging for improved efficiency. Demonstrates adaptive approaches to byte-level processing. Shows byte models can match subword efficiency through intelligent aggregation.

### 19. From Tokens to Characters: Towards Lossless Compression in Tokenization
- **File**: `2412.03719_tokens_to_characters.pdf`
- **Authors**: Various
- **Year**: 2024 (December - very recent!)
- **arXiv**: [2412.03719](https://arxiv.org/abs/2412.03719)
- **Size**: 1.5 MB
- **Why relevant**: Recent work on lossless compression approaches in tokenization. Explores theoretical limits of tokenization efficiency. Provides mathematical framework for optimal token language design.

### 20. Neurally Compressed Text: Towards Efficient Language Models
- **File**: `2404.03626_neurally_compressed_text.pdf`
- **Authors**: Various
- **Year**: 2024 (April)
- **arXiv**: [2404.03626](https://arxiv.org/abs/2404.03626)
- **Size**: 1.5 MB
- **Why relevant**: Proposes neural compression of text representations. Explores learned compression as alternative to fixed tokenization. Demonstrates end-to-end differentiable text compression for language modeling.

## Key Themes Across Papers

### 1. Byte-Level and Tokenizer-Free Models (Papers 9, 10, 11, 12, 16, 18)
A major research direction exploring models that operate without traditional tokenization:
- **ByT5** (2021): Pioneer showing byte-level models are practical
- **MEGABYTE** (2023): Hierarchical multi-scale approach for million-byte sequences
- **T-FREE** (2024): Sparse representations over character triplets
- **SpaceByte** (2024): Space-based segmentation boundaries
- **BLT** (2024): **BREAKTHROUGH** - First to match token-based LLMs at scale with dynamic patching
- **MrT5** (2024): Dynamic token merging for byte-level efficiency

### 2. Vocabulary Compression and Optimization (Papers 7, 13, 14, 15)
Research focused on reducing vocabulary size while maintaining or improving performance:
- **Vocabulary Trimming** (2023): 50% reduction for multilingual models
- **Fast Vocab Transfer** (2024): Domain-specific vocabulary optimization
- **Vocab Compression** (2024): 3.4x memory reduction, 3x throughput improvement
- **Over-Tokenized Transformer** (2025): Log-linear relationship between vocabulary size and performance

### 3. Alternative Tokenization Methods (Papers 2, 3, 4, 6, 8)
Proposals for improved tokenization strategies beyond standard BPE:
- **BPE Suboptimal** (2020): Unigram LM as alternative
- **Charformer** (2021): Gradient-based learned tokenization (28-100% speedup)
- **Rethinking Tokenization** (2024): Less-is-Better model with integrated vocabulary
- **ReTok** (2024): Replacing tokenizers for enhanced efficiency
- **Hierarchical Transformers** (2025): Multi-level byte and word processing

### 4. Theoretical Foundations (Papers 1, 5, 17, 19, 20)
Papers providing theoretical understanding and mathematical frameworks:
- **Theory of Tokenization** (2024): Information-theoretic perspective
- **Unpacking Tokenization** (2024): Compression correlation with performance
- **More Than Compression** (2024): Deeper analysis beyond compression
- **Tokens to Characters** (2024): Lossless compression theory
- **Neurally Compressed Text** (2024): Learned compression framework

### 5. Recent Trends (2024-2025)
**16 out of 20 papers are from 2024-2025**, indicating this is an extremely active research area:
- **January 2025**: Hierarchical transformers, over-tokenization
- **December 2024**: BLT (major breakthrough), lossless compression
- **2024 focus**: Byte-level models, vocabulary optimization, tokenizer-free approaches

## Critical Insights for Our Research

### Evidence Supporting the Hypothesis:
1. **Byte-level models now match token-based performance** (BLT, 2024) - proves alternative representations are viable
2. **Vocabulary compression achieves 3-3.4x efficiency gains** - directly supports compact token language approach
3. **Compression correlates with model quality** - validates efficiency as optimization target
4. **Multiple successful alternatives to BPE exist** - shows design space is large
5. **85% parameter reduction possible** (T-FREE) - demonstrates extreme compression potential
6. **Dynamic/adaptive approaches outperform fixed tokenization** - suggests learned token languages are promising

### Design Principles Extracted:
- **Hierarchical/multi-scale processing** improves efficiency (MEGABYTE, BLT, Hierarchical Transformers)
- **Entropy-based adaptation** enables optimal compute allocation (BLT)
- **Sparse representations** reduce memory footprint (T-FREE)
- **Learned tokenization** outperforms hand-designed rules (Charformer, Rethinking)
- **Larger vocabularies** improve performance with proper engineering (Over-Tokenized)

### Gaps and Opportunities:
1. No work explicitly designs "artificial languages" optimized for LLMs from first principles
2. Limited exploration of information-theoretic optimal token languages
3. Most work focuses on English/multilingual - opportunity for synthetic language design
4. Compression and reasoning quality trade-offs not fully explored
5. Few studies on training LLMs from scratch on artificial token languages

## Next Steps for Our Research

### Recommended Baselines:
1. **Standard BPE** (GPT-2/LLaMA tokenizer) - industry standard
2. **Unigram LM** - better than BPE per Paper 4
3. **ByT5** - byte-level baseline with released code
4. **BLT** - state-of-the-art byte-level with code available

### Recommended Datasets:
- **Language modeling**: WikiText, C4, The Pile
- **Compression benchmarks**: enwik8, enwik9
- **Reasoning**: TinyStories (used in vocab compression paper)
- **Multilingual**: mC4 (if testing cross-lingual properties)

### Evaluation Metrics:
- **Compression ratio**: tokens/bytes (from Papers 5, 19)
- **Model perplexity**: standard LM metric
- **Inference efficiency**: FLOPs, throughput (from Papers 9, 13)
- **Parameter efficiency**: params per performance level
- **Downstream task performance**: GLUE, SuperGLUE
- **Robustness**: noisy text, rare words (from Paper 11)

### Proposed Approach:
1. Design artificial token language using compression + linguistic principles
2. Train small LLM (125M-1B params) on TinyStories or similar
3. Compare against BPE, Unigram, ByT5 baselines
4. Measure compression, efficiency, and reasoning quality
5. If successful, scale to larger models and datasets
