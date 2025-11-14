import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import T5TokenizerFast
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=10,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--max_gen_length', type=int, default=128,
                        help="Maximum length for generation")
    parser.add_argument('--num_beams', type=int, default=4,
                        help="Number of beams for beam search")
    
    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    experiment_name = args.experiment_name
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/dev_gt_records.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, tokenizer,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping at epoch {epoch}")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    num_batches = 0

    for encoder_input, encoder_mask, labels in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels 
        )
        
        loss = outputs.loss 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        
        if scheduler is not None: 
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
        
def eval_epoch(args, model, dev_loader, tokenizer, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO
    model.eval()
    
    all_predictions = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for encoder_input, encoder_mask, labels in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate predictions
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_predictions.extend(predictions)
            
            # Print sample predictions from first batch
            if num_batches == 1: 
                print("\n" + "="*80)
                print("Sample Predictions (First Batch):")
                print("="*80)
                for i in range(min(3, len(predictions))):
                    print(f"\nExample {i+1}:")
                    print(f"Generated SQL: {predictions[i]}")
                print("="*80 + "\n")
                
            del encoder_input, encoder_mask, labels, outputs, generated
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    # Save predictions and compute metrics
    save_queries_and_records(all_predictions, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    error_count = sum(1 for msg in model_error_msgs if msg != "")
    error_rate = error_count / len(model_error_msgs) if model_error_msgs else 0
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate


        
def test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader, desc="Testing"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(predictions)

    save_queries_and_records(all_predictions, model_sql_path, model_record_path)
    print(f"Test predictions saved to {model_sql_path}")


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader, tokenizer,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
