# Research Plan: Artificial Token Language for More Efficient LLMs

**Date**: December 7, 2025
**Status**: Phase 1 - Planning Complete

---

## Research Question

**Can a compact, carefully designed artificial token language reduce LLM model size and computational requirements while maintaining or improving reasoning quality compared to training on standard tokenization methods (BPE) or byte-level approaches?**

Specifically: Does designing tokens from first principles—optimizing for compression, compositionality, and learnability—outperform existing approaches?

---

## Background and Motivation

### Why This Matters

Current LLMs are inefficient:
- GPT-3/4 use ~50K token vocabularies (BPE)
- Embedding tables alone consume hundreds of megabytes
- English requires ~4 tokens per word on average
- Other languages are even less efficient (e.g., some Asian languages need 2-3x more tokens)

### Evidence from Literature Review

The literature review of 20 papers (16 from 2024-2025) provides strong supporting evidence:

1. **Byte-level models now work at scale** (BLT, Dec 2024) - tokenizer-free is viable
2. **Vocabulary compression achieves 3-3.4x efficiency** without performance loss
3. **Compression correlates with performance** (Unpacking Tokenization paper)
4. **85% parameter reduction possible** (T-FREE) through sparse representations
5. **Learned/adaptive tokenization outperforms fixed** (Charformer, BLT)

### Critical Gap

**No existing work designs an artificial token language from first principles.** All work either:
- Uses raw bytes/characters (ByT5, BLT, MEGABYTE)
- Compresses existing vocabularies (Vocab Compression)
- Learns from data without explicit design (Charformer)

**Our opportunity**: Design an optimal token language using insights from all these approaches.

---

## Hypothesis Decomposition

### Main Hypothesis
Training on an artificial token language will be more efficient than both BPE and byte-level approaches.

### Sub-Hypotheses (Testable)

**H1: Compression Hypothesis**
- Artificial language will achieve 3-5x compression vs. BPE (based on vocab compression papers)
- Measure: Tokens per byte, tokens per word
- Success: <0.5 tokens/byte (vs BPE ~1.0, bytes 1.0)

**H2: Quality Hypothesis**
- Artificial language will match or beat BPE perplexity
- Measure: Perplexity on WikiText-103
- Success: Perplexity ≤ BPE baseline ±5%

**H3: Efficiency Hypothesis**
- Smaller vocabulary → fewer parameters in embedding layer
- Fewer tokens → faster inference
- Measure: Parameter count, throughput (tokens/sec)
- Success: >30% parameter reduction, >2x throughput vs byte-level

**H4: Compositionality Hypothesis**
- Artificial language preserves compositional structure (Theory of Tokenization requirement)
- Measure: Downstream task performance
- Success: Match BPE on reasoning tasks

---

## Proposed Methodology

### High-Level Strategy

**Three-Track Approach** (inspired by literature):

1. **Track 1: Information-Theoretic Design**
   - Analyze token frequency and patterns in natural language
   - Design compact vocabulary optimized for English (then extend)
   - Use entropy-based boundaries (inspired by BLT)

2. **Track 2: Learned Compression**
   - Use autoencoder-style approach
   - Map English → compact artificial language → English
   - Learn optimal token boundaries from data

3. **Track 3: Hybrid Approach** (Primary Focus)
   - Combine designed vocabulary with learned boundaries
   - Use principles from BLT (dynamic), Vocab Compression (grouping), T-FREE (sparsity)
   - Optimize for both compression and learnability

**Given time constraints, we'll focus on Track 3 with a pragmatic proof-of-concept approach.**

### Refined Scope (Realistic for Automated Research)

Given this is a fully automated research session, I'll focus on a **tractable proof-of-concept**:

**Core Experiment**: Design a compact artificial token language and demonstrate compression benefits with quality preservation on small-scale experiments.

**Key Design Decisions**:
1. Use **morpheme-based tokenization** as foundation (linguistically motivated)
2. Add **frequent phrases** as single tokens (reduce sequence length)
3. Implement using HuggingFace Tokenizers (fast, extensible)
4. Compare against BPE and simple baselines on text compression + small LM task

---

## Experimental Steps

### Phase 1: Artificial Language Design (Implementation Phase)

**Step 1.1: Analyze Natural Language Patterns**
- Use TinyStories dataset (simple, controlled vocabulary)
- Extract:
  - Word frequency distribution
  - Common morphemes (prefixes, suffixes, roots)
  - Frequent multi-word phrases
  - Character n-gram statistics

**Step 1.2: Design Token Vocabulary**
- **Tier 1** (Core): ~500-1000 most frequent morphemes/words
- **Tier 2** (Compositional): ~200-500 common affixes
- **Tier 3** (Phrases): ~300-500 frequent phrases
- **Tier 4** (Fallback): Character-level (for OOV)
- **Total vocabulary**: ~1500-2500 tokens (vs BPE 50K, bytes 256)

**Design Principles** (from literature):
- Preserve compositional structure (Theory of Tokenization)
- Optimize compression ratio (Unpacking Tokenization)
- Enable higher-order pattern learning (Theory of Tokenization)
- Use semantic grouping (Vocab Compression)

**Step 1.3: Implement Tokenizer**
- Use HuggingFace Tokenizers library
- Implement encoding: English → Artificial Language
- Implement decoding: Artificial Language → English
- Validate: Round-trip accuracy must be 100%

### Phase 2: Baseline Implementation

**Step 2.1: BPE Baseline**
- Train BPE tokenizer on TinyStories
- Vocabulary sizes: 1K, 2K, 5K (to match artificial language size)
- Use SentencePiece or HuggingFace Tokenizers

**Step 2.2: Byte Baseline**
- Simple byte-level tokenization (256 tokens)
- Reference implementation from HuggingFace

**Step 2.3: Unigram LM Baseline**
- Train Unigram LM tokenizer (shown to beat BPE in literature)
- Vocabulary: 1.5K-2.5K (same as artificial language)

### Phase 3: Compression Evaluation

**Step 3.1: Text Compression Benchmark**
- Test on WikiText-103 validation set
- Measure:
  - Tokens per byte
  - Tokens per word
  - Vocabulary utilization
- Statistical significance: Bootstrap confidence intervals

**Step 3.2: Compression-Quality Correlation**
- Validate that compression predicts quality (Unpacking Tokenization)
- Plot: Compression ratio vs. theoretical model capacity

### Phase 4: Language Modeling Evaluation (Simplified)

Given compute constraints and automation requirements, we'll use **pretrained models** adapted to our tokenization rather than training from scratch:

**Option A: Adapter-Based Evaluation** (Preferred)
- Use small pretrained model (e.g., GPT-2 small)
- Train adapter/projection layer: standard tokens → our tokens
- Fine-tune on TinyStories
- Measure perplexity

**Option B: From-Scratch Small Model** (If time permits)
- Train tiny LM (e.g., 50M params) from scratch
- Architecture: Simple transformer (6 layers, 512 hidden)
- Train on TinyStories subset (fast iteration)
- Measure perplexity, training speed, memory

**Given automation constraints, we'll focus on Option A or simpler compression-based evaluation.**

### Phase 5: Efficiency Analysis

**Step 5.1: Parameter Count**
- Compare embedding layer sizes:
  - BPE (5K vocab × embedding dim)
  - Artificial (1.5K vocab × embedding dim)
  - Byte (256 vocab × embedding dim, but longer sequences)

**Step 5.2: Inference Speed Simulation**
- Estimate throughput based on sequence length
- Tokens/second = (baseline_speed × baseline_tokens) / our_tokens

**Step 5.3: Memory Footprint**
- Calculate: embedding parameters + activation memory
- Compare across methods

---

## Baselines

### Required Baselines (from literature)

1. **BPE (SentencePiece)**
   - Standard: 50K vocab
   - Matched: 1.5K-2.5K vocab (for fair comparison)
   - Rationale: Industry standard

2. **Unigram LM**
   - Vocabulary: 1.5K-2.5K
   - Rationale: Shown to outperform BPE (Paper 2004.03720)

3. **Byte-level**
   - Raw UTF-8 bytes (256 vocab)
   - Rationale: Comparison to tokenizer-free approaches (ByT5 baseline)

### Aspirational Baseline (if time/compute allows)

4. **ByT5-small**
   - Pretrained model from HuggingFace
   - Rationale: Production byte-level baseline
   - Note: May not be feasible for full training comparison

---

## Evaluation Metrics

### Primary Metrics (Essential)

1. **Compression Ratio**
   - **Definition**: Tokens per byte, tokens per word
   - **Why**: Validated predictor of quality (Unpacking Tokenization)
   - **Target**: <0.5 tokens/byte (vs BPE ~1.0)

2. **Perplexity** (if LM evaluation feasible)
   - **Definition**: exp(cross-entropy loss) on held-out data
   - **Why**: Standard LM quality metric
   - **Target**: Match or beat BPE baseline

3. **Vocabulary Size**
   - **Definition**: Total unique tokens
   - **Why**: Directly impacts parameter count
   - **Target**: 1.5K-2.5K (vs BPE 50K)

### Secondary Metrics (Recommended)

4. **Parameter Efficiency**
   - **Definition**: (Embedding params + head params) saved
   - **Why**: Practical deployment consideration
   - **Target**: >50% reduction vs BPE-50K

5. **Throughput Estimate**
   - **Definition**: Tokens/second (estimated from sequence length)
   - **Why**: Inference efficiency
   - **Target**: >2x vs byte-level

6. **Round-Trip Accuracy**
   - **Definition**: % of text correctly encoded and decoded
   - **Why**: Validate tokenizer correctness
   - **Target**: 100% (must be lossless)

### Analysis Metrics (For Understanding)

7. **Vocabulary Utilization**
   - **Definition**: % of tokens actually used
   - **Why**: Detect over/under-parameterization

8. **Token Length Distribution**
   - **Definition**: Histogram of characters per token
   - **Why**: Understand compression mechanism

9. **OOV Rate** (Out-of-Vocabulary)
   - **Definition**: % of words requiring fallback to character-level
   - **Why**: Robustness check

---

## Statistical Analysis Plan

### Comparison Tests

**For Compression Ratios**:
- Bootstrap confidence intervals (1000 resamples)
- Test: Is artificial language significantly more compressed than BPE?
- Significance level: α = 0.05
- Effect size: Cohen's d

**For Perplexity** (if applicable):
- Paired comparison on same test set
- Test: Paired t-test (if normal) or Wilcoxon signed-rank (if not)
- Significance level: α = 0.05
- Report: p-value, confidence interval, effect size

### Multiple Comparisons

- We have 3 primary comparisons (vs BPE, Unigram, Byte)
- Bonferroni correction: α_corrected = 0.05/3 = 0.017
- Report both uncorrected and corrected p-values

### Robustness Checks

- Test on multiple datasets (TinyStories, WikiText-103, enwik8)
- Check consistency across data domains
- Report variance across datasets

---

## Expected Outcomes

### Success Scenario (Hypothesis Supported)

**Quantitative**:
- Compression: 0.3-0.5 tokens/byte (vs BPE 0.8-1.0)
- Vocabulary: 1.5K-2.5K (vs BPE 50K)
- Perplexity: Within 5% of BPE baseline
- Parameter reduction: 50-70% in embedding layer

**Qualitative**:
- Artificial language is interpretable (tokens correspond to meaningful units)
- Smooth degradation on OOV (character-level fallback works)
- Practical implications: Could train smaller, faster LLMs

### Partial Success (Mixed Results)

**Scenario 1**: Good compression, worse quality
- Interpretation: Over-compressed, lost compositional structure
- Implication: Need to balance compression and learnability

**Scenario 2**: Good quality, modest compression
- Interpretation: Design is conservative, can push further
- Implication: Iterate on vocabulary selection

**Scenario 3**: Works on simple data (TinyStories), fails on complex
- Interpretation: Vocabulary too domain-specific
- Implication: Need broader design or adaptive approach

### Failure Scenario (Hypothesis Refuted)

**Quantitative**:
- Compression similar to BPE (no benefit)
- OR: Perplexity significantly worse (>10% increase)
- OR: Round-trip accuracy <100% (lossy encoding)

**Interpretation**:
- Artificial language design is flawed
- OR: Optimal vocabulary is larger than expected
- OR: Learned approaches (BLT) are superior to designed approaches

**Next Steps**:
- Error analysis: Where does artificial language fail?
- Try learned approach (Track 2)
- Combine with byte-level dynamic patching (BLT-style)

---

## Timeline and Milestones

### Realistic Automated Research Timeline (6-8 hours total)

**Phase 1: Planning** [COMPLETE]
- Duration: 20 minutes
- Deliverable: planning.md

**Phase 2: Environment Setup** [NEXT]
- Duration: 15 minutes
- Tasks:
  - Create virtual environment with uv
  - Install dependencies (PyTorch, HuggingFace, NumPy, etc.)
  - Download TinyStories dataset (500MB)
- Deliverable: Working Python environment

**Phase 3: Artificial Language Design + Implementation**
- Duration: 90 minutes
- Tasks:
  - Analyze TinyStories vocabulary (30 min)
  - Design token vocabulary with morphemes + phrases (30 min)
  - Implement tokenizer (30 min)
  - Validate round-trip accuracy (10 min)
- Deliverable: Working artificial language tokenizer

**Phase 4: Baseline Implementation**
- Duration: 60 minutes
- Tasks:
  - Train BPE tokenizer (20 min)
  - Train Unigram tokenizer (20 min)
  - Implement byte tokenizer (10 min)
  - Validate all tokenizers (10 min)
- Deliverable: 3 baseline tokenizers

**Phase 5: Compression Evaluation**
- Duration: 45 minutes
- Tasks:
  - Measure compression on WikiText-103 (20 min)
  - Statistical analysis (15 min)
  - Visualization (10 min)
- Deliverable: Compression results with significance tests

**Phase 6: Language Modeling Evaluation** (Simplified or Optional)
- Duration: 60-90 minutes (if feasible)
- Tasks:
  - Option A: Use API (GPT-4, Claude) to evaluate text quality (30 min)
  - Option B: Small from-scratch model (90 min if time allows)
- Deliverable: Perplexity or quality scores

**Phase 7: Analysis and Documentation**
- Duration: 90 minutes
- Tasks:
  - Comprehensive analysis (30 min)
  - Create REPORT.md (45 min)
  - Create README.md (15 min)
- Deliverable: Complete documentation

**Total Estimated Time**: 5.5-6.5 hours

**Buffer**: 1.5-2.5 hours for debugging, iteration, unexpected issues

---

## Potential Challenges and Mitigation

### Challenge 1: Vocabulary Design is Subjective

**Risk**: No clear "optimal" vocabulary, many design choices
**Mitigation**:
- Use data-driven approach (extract from TinyStories)
- Try multiple designs (morpheme-based, frequency-based, hybrid)
- Evaluate objectively (compression + quality metrics)
**Fallback**: If designed approach fails, use learned approach (Unigram LM with smaller vocab)

### Challenge 2: Training LLMs from Scratch is Expensive

**Risk**: May not have time/compute to train large models
**Mitigation**:
- Focus on compression metrics first (doesn't require training)
- Use very small models (50M params) or pretrained adapters
- Evaluate on simpler tasks (perplexity, not full GLUE)
**Fallback**: Report compression results only, note that full LM training is future work

### Challenge 3: Round-Trip Encoding May Not Be Lossless

**Risk**: Artificial language may not be able to represent all text
**Mitigation**:
- Include character-level fallback for OOV
- Test round-trip accuracy early
- Fix encoding issues before evaluating compression
**Fallback**: Use lossy encoding but document limitations

### Challenge 4: Results May Be Dataset-Specific

**Risk**: Artificial language optimized for TinyStories may not generalize
**Mitigation**:
- Test on multiple datasets (TinyStories, WikiText-103, enwik8)
- Report performance variance
- Analyze failure modes on different domains
**Fallback**: Acknowledge limitation, propose adaptive vocabulary as future work

### Challenge 5: Baselines May Be Stronger Than Expected

**Risk**: BPE/Unigram with matched vocab size may be very competitive
**Mitigation**:
- Compare against both matched (fair) and standard (practical) baselines
- Analyze where artificial language wins/loses
- Look for qualitative advantages (interpretability, multilingual potential)
**Fallback**: Pivot to learned approach or hybrid design

---

## Success Criteria

### Minimum Viable Success (Must Achieve)

1. ✅ Artificial language tokenizer implemented and validated (100% round-trip accuracy)
2. ✅ Compression ratio measured on at least 2 datasets
3. ✅ Statistical comparison vs. BPE baseline
4. ✅ Comprehensive documentation (REPORT.md, README.md)

### Target Success (Goal)

5. ✅ Achieve >2x compression vs BPE with matched vocabulary
6. ✅ Perplexity within 10% of BPE baseline (on small LM task)
7. ✅ Demonstrate parameter efficiency benefits
8. ✅ Error analysis and insights for future work

### Stretch Success (If Time Allows)

9. ✅ Test on 3+ datasets (TinyStories, WikiText, enwik8)
10. ✅ Downstream task evaluation (simple classification)
11. ✅ Multilingual generalization test
12. ✅ Comparison to ByT5 or other advanced baselines

---

## Refined Scope and Pragmatic Approach

### What We WILL Do (Core Experiment)

1. **Design artificial token language** based on morphological analysis
2. **Implement tokenizer** with HuggingFace Tokenizers
3. **Measure compression** on standard benchmarks (enwik8, WikiText-103)
4. **Compare to BPE/Unigram baselines** with statistical tests
5. **Analyze trade-offs** between compression and vocabulary size
6. **Document findings** with clear methodology and reproducibility

### What We WILL NOT Do (Out of Scope)

1. **Train large LLMs from scratch** (too expensive, use compression as proxy)
2. **Full GLUE evaluation** (focus on compression and simple LM metrics)
3. **Multilingual extension** (focus on English proof-of-concept)
4. **Production optimization** (focus on scientific validation)

### Why This Scope is Sufficient

**From literature review**:
- Compression correlates with quality (Unpacking Tokenization) → compression evaluation is valid
- Byte-level works at scale (BLT) → we can validate without large models
- Vocabulary compression shows 3x efficiency (Vocab Compression paper) → our target is validated

**Scientific contribution**:
- First designed (not learned) artificial token language
- Systematic comparison vs baselines
- Analysis of design principles and trade-offs
- Foundation for future work

---

## Experimental Design Summary

### Independent Variables
- Tokenization method: {Artificial, BPE, Unigram, Byte}
- Vocabulary size: {256, 1.5K, 2.5K, 5K, 50K}
- Dataset: {TinyStories, WikiText-103, enwik8}

### Dependent Variables
- Compression ratio (tokens/byte, tokens/word)
- Vocabulary utilization (%)
- Parameter count (embedding layer)
- Perplexity (if LM evaluation performed)

### Controlled Variables
- Text domain (same datasets for all methods)
- Encoding/decoding implementation (HuggingFace Tokenizers)
- Evaluation metrics (consistent across methods)

### Confounding Variables to Address
- Vocabulary size effect: Compare artificial vs BPE with MATCHED vocab sizes
- Dataset bias: Test on multiple datasets
- Implementation quality: Use standard libraries, validate round-trip accuracy

---

## Expected Contributions

### Scientific Contributions

1. **First designed artificial token language for LLMs**
   - Novel approach: design from first principles vs learn from data
   - Combines insights from BLT, Vocab Compression, Theory of Tokenization

2. **Systematic evaluation framework**
   - Compression-quality trade-off analysis
   - Controlled comparison across vocabulary sizes
   - Statistical validation

3. **Insights for optimal tokenization**
   - What vocabulary size is optimal?
   - Is morpheme-based design better than learned?
   - How does compression correlate with quality?

### Practical Implications

1. **Efficiency gains**: If successful, 2-5x fewer tokens → faster inference
2. **Smaller models**: 50-70% fewer embedding parameters
3. **Design principles**: Guidance for future tokenization research
4. **Open questions**: Does designed language generalize? Scale to large models?

---

## References to Literature

### Key Papers Informing Design

1. **BLT** (2412.09871): Entropy-based dynamic boundaries, byte-level scaling
2. **Unpacking Tokenization** (2403.06265): Compression-quality correlation
3. **Theory of Tokenization** (2404.08335): Compositional structure requirement
4. **Vocab Compression** (2411.06371): 3x efficiency target, TinyStories dataset
5. **ByT5** (2105.13626): Byte-level baseline, robustness to noise

### Datasets (from Literature)

- **TinyStories**: Primary (Vocab Compression paper)
- **WikiText-103**: Standard benchmark (most papers)
- **enwik8**: Compression benchmark (BLT, ByT5, MEGABYTE)

### Baselines (from Literature)

- **BPE**: Standard (GPT-2, LLaMA)
- **Unigram LM**: Better than BPE (Paper 2004.03720)
- **Byte-level**: Tokenizer-free (ByT5, BLT)

---

## Next Steps After Planning

1. ✅ Set up environment (Phase 2)
2. ✅ Download TinyStories dataset
3. ✅ Analyze vocabulary and design artificial language (Phase 3)
4. ✅ Implement and validate tokenizer
5. ✅ Run compression experiments
6. ✅ Document results

**CRITICAL**: This plan is designed for a fully automated research session. We will proceed immediately from planning to implementation without waiting for user input.

---

**END OF PLANNING DOCUMENT**
