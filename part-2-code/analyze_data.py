import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import T5TokenizerFast
from tqdm import tqdm

# ============================================================================
# Spider2TextDataset Class with Data Quantity Tracking
# ============================================================================

class Spider2TextDataset(Dataset):
    def __init__(self, split, tokenizer, max_input_length, max_output_length):
        super().__init__()
        self.split = split
        
        # Read data
        nl_path = f'data/{split}.nl'
        sql_path = f'data/{split}.sql'
        
        print(f"\n📂 Loading {split} data...")
        print(f"   Reading from: {nl_path} and {sql_path}")
        
        with open(nl_path, 'r', encoding='utf8') as f:
            nl = [line.strip() for line in f]
        with open(sql_path, 'r', encoding='utf8') as f:
            sql = [line.strip() for line in f]
        
        # Track original data count
        self.original_count = len(nl)
        print(f"   Original data count: {self.original_count}")
        
        # Store raw examples for later analysis
        self.examples = [{'nl': n, 'sql': s} for n, s in zip(nl, sql)]
        
        # Filter empty or invalid examples (if any)
        valid_examples = []
        filtered_count = 0
        for ex in self.examples:
            if ex['nl'].strip() and ex['sql'].strip():
                valid_examples.append(ex)
            else:
                filtered_count += 1
        
        self.examples = valid_examples
        self.filtered_count = filtered_count
        
        print(f"   Valid examples: {len(self.examples)}")
        if filtered_count > 0:
            print(f"   ⚠️  Filtered out: {filtered_count} empty/invalid examples")
        
        # Tokenization parameters
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        print(f"   Tokenization config:")
        print(f"     • Max input length: {max_input_length}")
        print(f"     • Max output length: {max_output_length}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        
        # Tokenize input (NL)
        input_encoding = self.tokenizer(
            example['nl'],
            padding='max_length',
            max_length=self.max_input_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize output (SQL)
        target_encoding = self.tokenizer(
            example['sql'],
            padding='max_length',
            max_length=self.max_output_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_data_statistics(self):
        """Get statistics about data filtering"""
        return {
            'original_count': self.original_count,
            'valid_count': len(self.examples),
            'filtered_count': self.filtered_count,
            'retention_rate': len(self.examples) / self.original_count * 100 if self.original_count > 0 else 0
        }

# ============================================================================
# Helper Functions
# ============================================================================

def get_vocabulary(texts, tokenizer):
    """Get unique tokens from a list of texts"""
    vocab = set()
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        vocab.update(tokens)
    return vocab

# ============================================================================
# Analysis Functions: BEFORE PREPROCESSING
# ============================================================================

def analyze_raw_data(nl_path, sql_path, tokenizer, split_name):
    """Analyze raw data files before preprocessing"""
    print(f"\n{'─'*80}")
    print(f"📂 Analyzing RAW {split_name.upper()} data")
    print(f"{'─'*80}")
    
    # Read files
    with open(nl_path, 'r', encoding='utf8') as f:
        nl_lines = [line.strip() for line in f]
    with open(sql_path, 'r', encoding='utf8') as f:
        sql_lines = [line.strip() for line in f]
    
    num_examples = len(nl_lines)
    print(f"  • Number of examples: {num_examples}")
    
    # Tokenize and compute statistics
    nl_lengths = []
    sql_lengths = []
    
    print("  • Tokenizing NL sentences...")
    for nl in tqdm(nl_lines, desc="  • Tokenizing NL"):
        tokens = tokenizer.encode(nl, add_special_tokens=False)
        nl_lengths.append(len(tokens))
    
    print("  • Tokenizing SQL queries...")
    for sql in tqdm(sql_lines, desc="  • Tokenizing SQL"):
        tokens = tokenizer.encode(sql, add_special_tokens=False)
        sql_lengths.append(len(tokens))
    
    print("  • Computing vocabulary sizes...")
    nl_vocab = get_vocabulary(nl_lines, tokenizer)
    sql_vocab = get_vocabulary(sql_lines, tokenizer)
    
    print("  ✅ Analysis complete!")
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': np.mean(nl_lengths),
        'mean_sql_length': np.mean(sql_lengths),
        'vocab_nl': len(nl_vocab),
        'vocab_sql': len(sql_vocab),
        'nl_texts': nl_lines,
        'sql_texts': sql_lines,
    }

# ============================================================================
# Analysis Functions: AFTER PREPROCESSING
# ============================================================================

def analyze_dataset(dataset, tokenizer, split_name):
    """Analyze preprocessed dataset"""
    print(f"\n{'─'*80}")
    print(f"🔄 Analyzing PREPROCESSED {split_name.upper()} dataset")
    print(f"{'─'*80}")
    print(f"  • Number of examples: {len(dataset)}")
    
    nl_texts = []
    sql_texts = []
    nl_lengths = []
    sql_lengths = []
    
    print("  • Decoding tokenized data...")
    for i in tqdm(range(len(dataset)), desc="  • Computing statistics"):
        example = dataset[i]
        
        # Decode input
        input_ids = example['input_ids']
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        nl_texts.append(input_text)
        nl_lengths.append(len(tokenizer.encode(input_text, add_special_tokens=False)))
        
        # Decode labels
        labels = example['labels']
        labels_cleaned = labels.clone()
        labels_cleaned[labels_cleaned == -100] = tokenizer.pad_token_id
        sql_text = tokenizer.decode(labels_cleaned, skip_special_tokens=True)
        sql_texts.append(sql_text)
        sql_lengths.append(len(tokenizer.encode(sql_text, add_special_tokens=False)))
    
    print("  ✅ Analysis complete!")
    
    # Compute vocabulary
    nl_vocab = get_vocabulary(nl_texts, tokenizer)
    sql_vocab = get_vocabulary(sql_texts, tokenizer)
    
    return {
        'mean_nl_length': np.mean(nl_lengths),
        'mean_sql_length': np.mean(sql_lengths),
        'vocab_nl': len(nl_vocab),
        'vocab_sql': len(sql_vocab),
        'nl_texts': nl_texts,
        'sql_texts': sql_texts,
    }

# ============================================================================
# Data Quantity Analysis
# ============================================================================

def print_data_quantity_analysis(train_dataset, dev_dataset):
    """Print data quantity changes before and after preprocessing"""
    print("\n" + "="*80)
    print("📊 DATA QUANTITY ANALYSIS (Before vs After Preprocessing)")
    print("="*80)
    
    train_stats = train_dataset.get_data_statistics()
    dev_stats = dev_dataset.get_data_statistics()
    
    print(f"\n{'Dataset':<15} {'Original':<15} {'After Filter':<15} {'Filtered Out':<15} {'Retention Rate':<15}")
    print("─"*80)
    print(f"{'Train':<15} {train_stats['original_count']:<15,} {train_stats['valid_count']:<15,} {train_stats['filtered_count']:<15,} {train_stats['retention_rate']:<14.2f}%")
    print(f"{'Dev':<15} {dev_stats['original_count']:<15,} {dev_stats['valid_count']:<15,} {dev_stats['filtered_count']:<15,} {dev_stats['retention_rate']:<14.2f}%")
    
    total_original = train_stats['original_count'] + dev_stats['original_count']
    total_valid = train_stats['valid_count'] + dev_stats['valid_count']
    total_filtered = train_stats['filtered_count'] + dev_stats['filtered_count']
    total_retention = total_valid / total_original * 100 if total_original > 0 else 0
    
    print("─"*80)
    print(f"{'Total':<15} {total_original:<15,} {total_valid:<15,} {total_filtered:<15,} {total_retention:<14.2f}%")
    
    # Summary
    print("\n📌 Summary:")
    if total_filtered == 0:
        print("   ✅ No data was filtered out during preprocessing")
        print("   ✅ All original examples are retained")
    else:
        print(f"   ⚠️  {total_filtered} examples were filtered out")
        print(f"   • Possible reasons: empty NL or SQL, invalid format")
        print(f"   • Retention rate: {total_retention:.2f}%")
    
    return {
        'train': train_stats,
        'dev': dev_stats,
        'total_original': total_original,
        'total_valid': total_valid,
        'total_filtered': total_filtered,
        'total_retention': total_retention
    }

# ============================================================================
# Truncation Analysis
# ============================================================================

def analyze_truncation(dataset, tokenizer, split_name, max_input_length, max_output_length):
    """Analyze if any examples were truncated during tokenization"""
    print("\n" + "="*80)
    print(f"✂️  TRUNCATION ANALYSIS ({split_name.upper()})")
    print("="*80)
    
    input_truncated = 0
    output_truncated = 0
    
    print(f"\n⏳ Checking {len(dataset)} examples for truncation...")
    
    for i in tqdm(range(len(dataset)), desc="  • Checking"):
        example = dataset.examples[i]
        
        # Check input truncation
        input_tokens = tokenizer.encode(example['nl'], add_special_tokens=True)
        if len(input_tokens) > max_input_length:
            input_truncated += 1
        
        # Check output truncation
        output_tokens = tokenizer.encode(example['sql'], add_special_tokens=True)
        if len(output_tokens) > max_output_length:
            output_truncated += 1
    
    print(f"\n📊 Truncation Statistics:")
    print(f"{'─'*80}")
    print(f"  Input (NL) truncated:   {input_truncated:>6} / {len(dataset):<6} ({input_truncated/len(dataset)*100:>5.2f}%)")
    print(f"  Output (SQL) truncated: {output_truncated:>6} / {len(dataset):<6} ({output_truncated/len(dataset)*100:>5.2f}%)")
    
    if input_truncated > 0 or output_truncated > 0:
        print(f"\n⚠️  Warning: Some examples were truncated!")
        print(f"   • Max input length: {max_input_length}")
        print(f"   • Max output length: {max_output_length}")
        print(f"   • Consider increasing max_length if truncation is significant")
    else:
        print(f"\n✅ No truncation occurred!")
        print(f"   • All examples fit within the specified max_length")
    
    return {
        'input_truncated': input_truncated,
        'output_truncated': output_truncated,
        'input_truncation_rate': input_truncated / len(dataset) * 100 if len(dataset) > 0 else 0,
        'output_truncation_rate': output_truncated / len(dataset) * 100 if len(dataset) > 0 else 0
    }

# ============================================================================
# Length Distribution Analysis
# ============================================================================

def show_length_distribution(dataset, tokenizer, split_name):
    """Show distribution of sequence lengths"""
    print("\n" + "="*80)
    print(f"📏 LENGTH DISTRIBUTION ({split_name.upper()})")
    print("="*80)
    
    input_lengths = []
    output_lengths = []
    
    print(f"\n⏳ Computing lengths for {len(dataset)} examples...")
    
    for i in tqdm(range(len(dataset)), desc="  • Computing"):
        example = dataset.examples[i]
        
        input_tokens = tokenizer.encode(example['nl'], add_special_tokens=True)
        output_tokens = tokenizer.encode(example['sql'], add_special_tokens=True)
        
        input_lengths.append(len(input_tokens))
        output_lengths.append(len(output_tokens))
    
    # Statistics
    print(f"\n📊 Input (NL) Length Distribution:")
    print(f"{'─'*80}")
    print(f"  Min:              {np.min(input_lengths):>6}")
    print(f"  Max:              {np.max(input_lengths):>6}")
    print(f"  Mean:             {np.mean(input_lengths):>6.2f}")
    print(f"  Median:           {np.median(input_lengths):>6.2f}")
    print(f"  Std:              {np.std(input_lengths):>6.2f}")
    print(f"  95th percentile:  {np.percentile(input_lengths, 95):>6.2f}")
    print(f"  99th percentile:  {np.percentile(input_lengths, 99):>6.2f}")
    
    print(f"\n📊 Output (SQL) Length Distribution:")
    print(f"{'─'*80}")
    print(f"  Min:              {np.min(output_lengths):>6}")
    print(f"  Max:              {np.max(output_lengths):>6}")
    print(f"  Mean:             {np.mean(output_lengths):>6.2f}")
    print(f"  Median:           {np.median(output_lengths):>6.2f}")
    print(f"  Std:              {np.std(output_lengths):>6.2f}")
    print(f"  95th percentile:  {np.percentile(output_lengths, 95):>6.2f}")
    print(f"  99th percentile:  {np.percentile(output_lengths, 99):>6.2f}")
    
    return {
        'input': {
            'min': np.min(input_lengths),
            'max': np.max(input_lengths),
            'mean': np.mean(input_lengths),
            'median': np.median(input_lengths),
            'p95': np.percentile(input_lengths, 95),
            'p99': np.percentile(input_lengths, 99),
        },
        'output': {
            'min': np.min(output_lengths),
            'max': np.max(output_lengths),
            'mean': np.mean(output_lengths),
            'median': np.median(output_lengths),
            'p95': np.percentile(output_lengths, 95),
            'p99': np.percentile(output_lengths, 99),
        }
    }

# ============================================================================
# Print Functions
# ============================================================================

def print_table(title, train_stats, dev_stats, include_num_examples=False):
    """Print statistics table"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    print()
    print(f"{'Statistics Name':<55} {'Train':>12} {'Dev':>12}")
    print("─"*80)
    
    if include_num_examples:
        print(f"{'Number of examples':<55} {train_stats['num_examples']:>12,} {dev_stats['num_examples']:>12,}")
    
    print(f"{'Mean sentence length (tokens)':<55} {train_stats['mean_nl_length']:>12.2f} {dev_stats['mean_nl_length']:>12.2f}")
    print(f"{'Mean SQL query length (tokens)':<55} {train_stats['mean_sql_length']:>12.2f} {dev_stats['mean_sql_length']:>12.2f}")
    print(f"{'Vocabulary size (natural language)':<55} {train_stats['vocab_nl']:>12,} {dev_stats['vocab_nl']:>12,}")
    print(f"{'Vocabulary size (SQL)':<55} {train_stats['vocab_sql']:>12,} {dev_stats['vocab_sql']:>12,}")

def show_sample_transformations(stats, num_samples=3):
    """Show sample transformed data"""
    print("\n" + "="*80)
    print(f"📝 SAMPLE TRANSFORMATIONS (First {num_samples} Examples from Train Set)")
    print("="*80)
    
    for i in range(min(num_samples, len(stats['nl_texts']))):
        print(f"\n{'─'*80}")
        print(f"Example {i+1}:")
        print(f"{'─'*80}")
        print(f"  📖 Natural Language:")
        print(f"     {stats['nl_texts'][i]}")
        print(f"\n  🔧 SQL Query:")
        print(f"     {stats['sql_texts'][i]}")

def print_latex_table(title, train_stats, dev_stats, include_num_examples=False):
    """Print LaTeX format table"""
    print("\n" + "="*80)
    print(f"📋 LATEX FORMAT - {title}")
    print("="*80)
    
    if include_num_examples:
        print(f"Number of examples & {train_stats['num_examples']:,} & {dev_stats['num_examples']:,} \\\\")
    
    print(f"Mean sentence length & {train_stats['mean_nl_length']:.2f} & {dev_stats['mean_nl_length']:.2f} \\\\")
    print(f"Mean SQL query length & {train_stats['mean_sql_length']:.2f} & {dev_stats['mean_sql_length']:.2f} \\\\")
    print(f"Vocabulary size (natural language) & {train_stats['vocab_nl']:,} & {dev_stats['vocab_nl']:,} \\\\")
    print(f"Vocabulary size (SQL) & {train_stats['vocab_sql']:,} & {dev_stats['vocab_sql']:,} \\\\")

def print_latex_data_quantity(quantity_stats):
    """Print LaTeX table for data quantity analysis"""
    print("\n" + "="*80)
    print("📋 LATEX FORMAT - DATA QUANTITY TABLE")
    print("="*80)
    print()
    print("% Copy-paste this into your LaTeX document:")
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Data Quantity Before and After Preprocessing}")
    print("\\begin{tabular}{lrrrr}")
    print("\\hline")
    print("Dataset & Original & After Filter & Filtered Out & Retention Rate \\\\")
    print("\\hline")
    
    train = quantity_stats['train']
    dev = quantity_stats['dev']
    
    print(f"Train & {train['original_count']:,} & {train['valid_count']:,} & {train['filtered_count']:,} & {train['retention_rate']:.2f}\\% \\\\")
    print(f"Dev & {dev['original_count']:,} & {dev['valid_count']:,} & {dev['filtered_count']:,} & {dev['retention_rate']:.2f}\\% \\\\")
    print("\\hline")
    print(f"Total & {quantity_stats['total_original']:,} & {quantity_stats['total_valid']:,} & {quantity_stats['total_filtered']:,} & {quantity_stats['total_retention']:.2f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

def print_summary(train_before, dev_before, train_after, dev_after):
    """Print summary of changes"""
    print("\n" + "="*80)
    print("📊 SUMMARY OF CHANGES")
    print("="*80)
    
    print(f"\n🔹 Dataset Sizes:")
    print(f"   Train: {train_before['num_examples']:,} examples")
    print(f"   Dev:   {dev_before['num_examples']:,} examples")
    
    print(f"\n🔹 Average Lengths (Before → After):")
    print(f"   Train NL:  {train_before['mean_nl_length']:.2f} → {train_after['mean_nl_length']:.2f} tokens")
    print(f"   Dev NL:    {dev_before['mean_nl_length']:.2f} → {dev_after['mean_nl_length']:.2f} tokens")
    print(f"   Train SQL: {train_before['mean_sql_length']:.2f} → {train_after['mean_sql_length']:.2f} tokens")
    print(f"   Dev SQL:   {dev_before['mean_sql_length']:.2f} → {dev_after['mean_sql_length']:.2f} tokens")
    
    print(f"\n🔹 Vocabulary Sizes (Before → After):")
    print(f"   Train NL:  {train_before['vocab_nl']:,} → {train_after['vocab_nl']:,} unique tokens")
    print(f"   Dev NL:    {dev_before['vocab_nl']:,} → {dev_after['vocab_nl']:,} unique tokens")
    print(f"   Train SQL: {train_before['vocab_sql']:,} → {train_after['vocab_sql']:,} unique tokens")
    print(f"   Dev SQL:   {dev_before['vocab_sql']:,} → {dev_after['vocab_sql']:,} unique tokens")
    
    # Check for changes
    changes = []
    if abs(train_before['mean_nl_length'] - train_after['mean_nl_length']) > 0.01:
        changes.append("NL length changed")
    if abs(train_before['mean_sql_length'] - train_after['mean_sql_length']) > 0.01:
        changes.append("SQL length changed")
    if train_before['vocab_nl'] != train_after['vocab_nl']:
        changes.append("NL vocabulary changed")
    if train_before['vocab_sql'] != train_after['vocab_sql']:
        changes.append("SQL vocabulary changed")
    
    if changes:
        print(f"\n⚠️  Detected changes: {', '.join(changes)}")
    else:
        print(f"\n✅ No significant changes detected")

# ============================================================================
# Main Function
# ============================================================================

def main():
    print("="*80)
    print("DATA STATISTICS ANALYSIS FOR Q4")
    print("Using T5 Tokenizer (google-t5/t5-small with legacy=False)")
    print("="*80)
    
    # Load tokenizer
    print("\n🔧 Loading T5 tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small', legacy=False)
    print(f"✅ Tokenizer loaded successfully!")
    print(f"   • Model: google-t5/t5-small")
    print(f"   • Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"   • Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")
    
    # Configuration
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 512
    
    # ========================================================================
    # TABLE 1: BEFORE PRE-PROCESSING
    # ========================================================================
    print("\n" + "🔍"*40)
    print("TABLE 1: DATA STATISTICS BEFORE PRE-PROCESSING")
    print("🔍"*40)
    
    train_before = analyze_raw_data('data/train.nl', 'data/train.sql', tokenizer, 'train')
    dev_before = analyze_raw_data('data/dev.nl', 'data/dev.sql', tokenizer, 'dev')
    
    print_table("TABLE 1: Data Statistics Before Pre-processing", 
                train_before, dev_before, include_num_examples=True)
    
    # ========================================================================
    # LOAD DATASETS WITH PREPROCESSING
    # ========================================================================
    print("\n" + "🔍"*40)
    print("LOADING DATASETS WITH PREPROCESSING")
    print("🔍"*40)
    
    print(f"\n🔧 Creating Spider2TextDataset with:")
    print(f"   • Max input length: {MAX_INPUT_LENGTH}")
    print(f"   • Max output length: {MAX_OUTPUT_LENGTH}")
    
    train_dataset = Spider2TextDataset('train', tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH)
    dev_dataset = Spider2TextDataset('dev', tokenizer, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH)
    
    # DATA QUANTITY ANALYSIS
    quantity_stats = print_data_quantity_analysis(train_dataset, dev_dataset)
    
    # LENGTH DISTRIBUTION ANALYSIS
    train_length_dist = show_length_distribution(train_dataset, tokenizer, 'train')
    dev_length_dist = show_length_distribution(dev_dataset, tokenizer, 'dev')
    
    # TRUNCATION ANALYSIS
    train_truncation = analyze_truncation(train_dataset, tokenizer, 'train', 
                                         MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH)
    dev_truncation = analyze_truncation(dev_dataset, tokenizer, 'dev',
                                       MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH)
    
    # ========================================================================
    # TABLE 2: AFTER PRE-PROCESSING
    # ========================================================================
    print("\n" + "🔍"*40)
    print("TABLE 2: DATA STATISTICS AFTER PRE-PROCESSING")
    print("🔍"*40)
    
    train_after = analyze_dataset(train_dataset, tokenizer, 'train')
    dev_after = analyze_dataset(dev_dataset, tokenizer, 'dev')
    
    print_table("TABLE 2: Data Statistics After Pre-processing",
                train_after, dev_after, include_num_examples=False)
    
    # ========================================================================
    # SAMPLE TRANSFORMATIONS
    # ========================================================================
    show_sample_transformations(train_after)
    
    # ========================================================================
    # LATEX OUTPUT
    # ========================================================================
    print("\n" + "📋"*40)
    print("LATEX OUTPUT - READY TO COPY-PASTE INTO YOUR REPORT")
    print("📋"*40)
    
    # Data quantity table
    print_latex_data_quantity(quantity_stats)
    
    # Original tables
    print_latex_table("TABLE 1 (BEFORE PRE-PROCESSING)", 
                      train_before, dev_before, include_num_examples=True)
    
    print_latex_table("TABLE 2 (AFTER PRE-PROCESSING)", 
                      train_after, dev_after, include_num_examples=False)
    
    # ========================================================================
    # COMPREHENSIVE SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print(f"\n🔢 Data Quantity:")
    print(f"{'─'*80}")
    print(f"  Original examples:     {quantity_stats['total_original']:>8,}")
    print(f"  Valid examples:        {quantity_stats['total_valid']:>8,}")
    print(f"  Filtered out:          {quantity_stats['total_filtered']:>8,}")
    print(f"  Retention rate:        {quantity_stats['total_retention']:>7.2f}%")
    
    print(f"\n✂️  Truncation:")
    print(f"{'─'*80}")
    print(f"  Train input truncated:  {train_truncation['input_truncated']:>7} ({train_truncation['input_truncation_rate']:>5.2f}%)")
    print(f"  Train output truncated: {train_truncation['output_truncated']:>7} ({train_truncation['output_truncation_rate']:>5.2f}%)")
    print(f"  Dev input truncated:    {dev_truncation['input_truncated']:>7} ({dev_truncation['input_truncation_rate']:>5.2f}%)")
    print(f"  Dev output truncated:   {dev_truncation['output_truncated']:>7} ({dev_truncation['output_truncation_rate']:>5.2f}%)")
    
    print(f"\n📏 Length Statistics (Train):")
    print(f"{'─'*80}")
    print(f"  Input (NL):   Mean={train_length_dist['input']['mean']:.2f}, Max={train_length_dist['input']['max']}, 95th={train_length_dist['input']['p95']:.2f}")
    print(f"  Output (SQL): Mean={train_length_dist['output']['mean']:.2f}, Max={train_length_dist['output']['max']}, 95th={train_length_dist['output']['p95']:.2f}")
    
    # Original summary
    print_summary(train_before, dev_before, train_after, dev_after)
    
    print("\n" + "="*80)
    print("✨ ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n💡 Key Insights:")
    if quantity_stats['total_filtered'] == 0:
        print("   ✅ No data loss during preprocessing")
    else:
        print(f"   ⚠️  {quantity_stats['total_filtered']} examples filtered out")
    
    if train_truncation['output_truncated'] > 0 or dev_truncation['output_truncated'] > 0:
        total_trunc = train_truncation['output_truncated'] + dev_truncation['output_truncated']
        print(f"   ⚠️  {total_trunc} SQL queries were truncated")
        print(f"   💡 Consider increasing max_output_length beyond {MAX_OUTPUT_LENGTH}")
    else:
        print(f"   ✅ No truncation - max_length={MAX_OUTPUT_LENGTH} is sufficient")
    
    print("\n💡 Next steps:")
    print("   1. Copy the LATEX output above to your report")
    print("   2. Review the data quantity table to confirm no data loss")
    print("   3. Check truncation statistics")
    print("   4. Verify length distributions match expectations")
    print("\n")

if __name__ == '__main__':
    main()