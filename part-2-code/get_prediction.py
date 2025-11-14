import os
import torch
import argparse
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import T5Dataset, test_collate_fn
from torch.utils.data import DataLoader
from utils import compute_metrics, save_queries_and_records
import json

def load_best_model(model_path, device):
    """Load the best model checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading best model from {model_path}")
    
    try:
        # Load T5 model
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"\nCheckpoint Information:")
        
        # Safely print checkpoint info
        epoch = checkpoint.get('epoch', 'N/A')
        best_f1 = checkpoint.get('best_f1', None)
        
        print(f"  - Epoch: {epoch}")
        if best_f1 is not None:
            print(f"  - Best F1: {best_f1:.4f}")
        else:
            print(f"  - Best F1: N/A")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {total_params:,}")
        
        return model, checkpoint
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def generate_predictions(model, dataloader, tokenizer, args, device):
    """Generate predictions for the dataset"""
    model.eval()
    predictions = []
    
    print(f"\nGenerating predictions...")
    print(f"  - Max generation length: {args.max_gen_length}")
    print(f"  - Num beams: {args.num_beams}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Unpack batch
            if len(batch) == 3:  # test_collate_fn
                encoder_input, encoder_mask, initial_decoder_input = batch
            else:  # normal_collate_fn
                encoder_input, encoder_mask = batch[0], batch[1]
            
            encoder_input = encoder_input.to(device)
            encoder_mask = encoder_mask.to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            # Decode each sequence in batch
            for output in outputs:
                pred_sql = tokenizer.decode(output, skip_special_tokens=True)
                predictions.append(pred_sql)
    
    print(f"\n✅ Generated {len(predictions)} predictions")
    return predictions

def evaluate_predictions(predictions, args):
    """Evaluate predictions using compute_metrics"""
    print("\n" + "-"*70)
    print("Evaluating Predictions")
    print("-"*70)
    
    # Setup paths
    split = args.test_split
    experiment_name = args.experiment_name
    
    # Ground truth paths
    gt_sql_path = f'data/{split}.sql'
    gt_record_path = f'records/{split}_gt_records.pkl'
    
    # Model prediction paths
    model_sql_path = os.path.join(args.output_dir, f'{split}_predictions.sql')
    model_record_path = os.path.join(args.output_dir, f'{split}_predictions_records.pkl')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions and generate records
    print(f"Saving predictions to {model_sql_path}")
    save_queries_and_records(predictions, model_sql_path, model_record_path)
    print(f"✅ Predictions saved")
    print(f"✅ Records generated and saved to {model_record_path}")
    
    # Check if ground truth exists
    if not os.path.exists(gt_sql_path):
        print(f"\n⚠️  Ground truth SQL not found: {gt_sql_path}")
        print(f"Skipping evaluation (expected for test set)")
        return None
    
    if not os.path.exists(gt_record_path):
        print(f"\n⚠️  Ground truth records not found: {gt_record_path}")
        print(f"Generating ground truth records...")
        # Load ground truth SQL
        with open(gt_sql_path, 'r') as f:
            gt_sql = [line.strip() for line in f]
        # Generate ground truth records
        save_queries_and_records(gt_sql, gt_sql_path, gt_record_path)
        print(f"✅ Ground truth records generated")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    try:
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        # Calculate error rate
        error_count = sum(1 for msg in error_msgs if msg != "")
        error_rate = error_count / len(error_msgs) if error_msgs else 0
        
        results = {
            'sql_em': sql_em,
            'record_em': record_em,
            'record_f1': record_f1,
            'error_rate': error_rate,
            'num_errors': error_count,
            'total_examples': len(error_msgs)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Test Best T5 Model')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/ft_experiments/baseline_len512/best_model.pt',
                        help='Path to best model checkpoint')
    parser.add_argument('--test_split', type=str, default='dev',
                        choices=['dev', 'test'],
                        help='Which split to test on: dev or test')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--max_gen_length', type=int, default=300,
                        help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for beam search')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save predictions')
    parser.add_argument('--experiment_name', type=str, default='best_model_eval',
                        help='Experiment name for saving files')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("TESTING BEST MODEL")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_path}")
    print(f"Split: {args.test_split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max generation length: {args.max_gen_length}")
    print(f"Num beams: {args.num_beams}")
    print(f"Output directory: {args.output_dir}")
    
    # Load tokenizer
    print("\n" + "-"*70)
    print("Loading Tokenizer")
    print("-"*70)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small', legacy=False)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
    
    # Load best model
    print("\n" + "-"*70)
    print("Loading Model")
    print("-"*70)
    model, checkpoint = load_best_model(args.model_path, device)
    
    # Load dataset
    print("\n" + "-"*70)
    print(f"Loading {args.test_split.upper()} Dataset")
    print("-"*70)
    test_dataset = T5Dataset(
        data_folder='data',
        split=args.test_split
    )
    
    # Create dataloader
    collate_fn = test_collate_fn
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"✅ Dataset ready: {len(test_dataset)} examples, {len(test_loader)} batches")
    
    # Generate predictions
    print("\n" + "-"*70)
    print("Generating Predictions")
    print("-"*70)
    predictions = generate_predictions(model, test_loader, tokenizer, args, device)
    
    # Evaluate predictions
    results = evaluate_predictions(predictions, args)
    
    # Print results
    if results:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"\n📊 Metrics:")
        print(f"  • Record F1:       {results['record_f1']:.4f} ({results['record_f1']*100:.2f}%)")
        print(f"  • Record EM:       {results['record_em']:.4f} ({results['record_em']*100:.2f}%)")
        print(f"  • SQL EM:          {results['sql_em']:.4f} ({results['sql_em']*100:.2f}%)")
        print(f"  • Error Rate:      {results['error_rate']:.4f} ({results['error_rate']*100:.2f}%)")
        print(f"  • Errors:          {results['num_errors']} / {results['total_examples']}")
        
        # Save results to JSON
        results_file = os.path.join(args.output_dir, f'{args.test_split}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {results_file}")
    
    # Show sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 5)")
    print("="*70)
    for i, pred in enumerate(predictions[:5], 1):
        print(f"\n{i}. {pred[:150]}{'...' if len(pred) > 150 else ''}")
    
    # Final summary
    print("\n" + "="*70)
    print("TESTING COMPLETE ✅")
    print("="*70)
    
    model_sql_path = os.path.join(args.output_dir, f'{args.test_split}_predictions.sql')
    model_record_path = os.path.join(args.output_dir, f'{args.test_split}_predictions_records.pkl')
    
    print(f"\n📁 Output files:")
    print(f"  • SQL Predictions:  {model_sql_path}")
    print(f"  • Record Predictions: {model_record_path}")
    
    if results:
        results_file = os.path.join(args.output_dir, f'{args.test_split}_results.json')
        print(f"  • Evaluation Results: {results_file}")
    
    print(f"\n💡 To check predictions:")
    print(f"  head -20 {model_sql_path}")
    print(f"  wc -l {model_sql_path}")
    
    if results:
        print(f"\n🎯 Best Metric: Record F1 = {results['record_f1']:.4f}")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()