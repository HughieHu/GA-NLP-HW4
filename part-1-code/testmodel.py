# 在 import 部分之后，其他函数之前添加这个诊断函数
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def diagnose_model(model, eval_dataloader, device, num_batches=5):
    """
    诊断模型行为，判断是随机猜测还是系统性错误
    """
    print("\n" + "=" * 80)
    print("🔬 模型诊断开始")
    print("=" * 80)
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= num_batches:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
    
    # 统计分析
    from collections import Counter
    pred_counter = Counter(all_predictions)
    label_counter = Counter(all_labels)
    
    print(f"\n📊 预测分布（前 {num_batches} 个批次）:")
    print(f"   标签 0 (负面): {pred_counter[0]} 次")
    print(f"   标签 1 (正面): {pred_counter[1]} 次")
    
    print(f"\n📊 真实标签分布:")
    print(f"   标签 0 (负面): {label_counter[0]} 个")
    print(f"   标签 1 (正面): {label_counter[1]} 个")
    
    # 检查是否总是预测同一个类别
    if len(pred_counter) == 1:
        only_pred = list(pred_counter.keys())[0]
        print(f"\n⚠️  警告：模型总是预测类别 {only_pred}！")
        print("   这表明模型已经崩溃或未正确训练。")
        return "collapsed"
    
    # 计算预测准确率
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    accuracy = correct / len(all_predictions)
    
    print(f"\n✅ 局部准确率: {accuracy:.4f} ({correct}/{len(all_predictions)})")
    
    # 检查 logits 分布
    import numpy as np
    logits_array = np.array(all_logits)
    logits_0 = logits_array[:, 0]
    logits_1 = logits_array[:, 1]
    
    print(f"\n📈 Logits 统计:")
    print(f"   类别 0 logits - 均值: {np.mean(logits_0):.3f}, 标准差: {np.std(logits_0):.3f}")
    print(f"   类别 1 logits - 均值: {np.mean(logits_1):.3f}, 标准差: {np.std(logits_1):.3f}")
    
    # 检查是否是标签反转
    reversed_correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == (1 - l))
    reversed_accuracy = reversed_correct / len(all_predictions)
    
    if reversed_accuracy > 0.7:
        print(f"\n⚠️  标签可能反转！如果反转标签，准确率为: {reversed_accuracy:.4f}")
        return "reversed_labels"
    
    # 检查是否随机猜测
    if 0.45 <= accuracy <= 0.55:
        # 进一步检查 logits 的差异
        logit_diffs = logits_1 - logits_0
        avg_diff = np.mean(np.abs(logit_diffs))
        
        print(f"\n🎲 Logits 差异均值: {avg_diff:.4f}")
        
        if avg_diff < 0.5:
            print("   → 差异很小，模型可能接近随机猜测")
            return "random"
        else:
            print("   → Logits 有明显差异，但准确率低可能是其他原因")
            return "other_issue"
    
    # 显示一些具体样本
    print(f"\n📋 前 10 个样本的详细信息:")
    print(f"{'索引':<6} {'真实':<6} {'预测':<6} {'Logit[0]':<10} {'Logit[1]':<10} {'结果':<6}")
    print("-" * 60)
    
    for i in range(min(10, len(all_predictions))):
        result = "✅" if all_predictions[i] == all_labels[i] else "❌"
        print(f"{i:<6} {all_labels[i]:<6} {all_predictions[i]:<6} "
              f"{all_logits[i][0]:<10.3f} {all_logits[i][1]:<10.3f} {result:<6}")
    
    print("=" * 80)
    
    return "normal"


def test_model_with_examples(model_dir, device):
    """
    用明确的测试样本测试模型
    """
    print("\n" + "=" * 80)
    print("🧪 手动样本测试")
    print("=" * 80)
    
    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # 测试样本 - 非常明确的情感
    test_samples = [
        ("This movie is absolutely wonderful and amazing!", 1),
        ("Terrible film, complete waste of time and money.", 0),
        ("Best movie ever! Loved every second of it!", 1),
        ("Boring, dull, and utterly disappointing.", 0),
        ("Fantastic performances and brilliant direction!", 1),
        ("Worst movie I have ever seen in my life.", 0),
        ("Incredible story, amazing acting, perfect!", 1),
        ("Awful, horrible, do not watch this garbage.", 0),
    ]
    
    correct = 0
    results = []
    
    for text, true_label in test_samples:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_label = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        
        results.append({
            'text': text,
            'true': true_label,
            'pred': pred_label,
            'correct': is_correct,
            'logits': logits.cpu().numpy()[0],
            'confidence': confidence
        })
    
    # 打印结果
    print(f"\n{'结果':<4} {'真实':<6} {'预测':<6} {'置信度':<10} 文本")
    print("-" * 80)
    
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"{status:<4} {r['true']:<6} {r['pred']:<6} {r['confidence']:<10.3f} {r['text'][:50]}")
    
    accuracy = correct / len(test_samples)
    print(f"\n总体准确率: {correct}/{len(test_samples)} = {accuracy:.2%}")
    
    # 分析结果
    if accuracy == 0.0:
        print("\n⚠️  完全错误！标签可能完全反转！")
    elif accuracy == 1.0:
        print("\n✅ 完美！模型工作正常！")
    elif 0.45 <= accuracy <= 0.55:
        print("\n🎲 接近随机猜测！可能的原因：")
        print("   1. 模型未训练或训练失败")
        print("   2. 加载了错误的检查点")
        print("   3. 数据预处理有问题")
    else:
        print(f"\n📊 准确率 {accuracy:.2%} - 模型有一定区分能力但不理想")
    
    # 检查 logits 模式
    print(f"\n📈 Logits 分析:")
    for i, r in enumerate(results[:5]):
        print(f"   样本 {i+1}: [负面: {r['logits'][0]:.3f}, 正面: {r['logits'][1]:.3f}]")
    
    print("=" * 80)
    
    return accuracy
    

# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)

    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    # ========== 添加诊断 ==========
    print("\n🔍 开始诊断模型...")
    diagnosis_result = diagnose_model(model, eval_dataloader, device, num_batches=10)
    print(f"诊断结果: {diagnosis_result}")
    
    # 手动样本测试
    test_accuracy = test_model_with_examples(output_dir, device)
    # ========== 诊断结束 ==========

    metric = evaluate.load("accuracy")
    out_file = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

        # write to output file
        for pred, label in zip(predictions, batch["labels"]):
                out_file.write(f"{pred.item()}\n")
                out_file.write(f"{label.item()}\n")
    
    out_file.close()
    score = metric.compute()
    
    # ========== 最终分析 ==========
    print("\n" + "=" * 80)
    print("🎯 最终评估结果")
    print("=" * 80)
    print(f"测试集准确率: {score['accuracy']:.4f}")
    print(f"手动样本准确率: {test_accuracy:.4f}")
    print(f"诊断结果: {diagnosis_result}")
    
    if score['accuracy'] < 0.55 and test_accuracy < 0.55:
        print("\n⚠️  结论：模型准确率接近随机猜测")
        print("\n可能的原因:")
        print("   1. 模型未正确训练（检查训练loss）")
        print("   2. 加载了未训练的模型")
        print("   3. 标签映射错误")
        print("   4. 数据预处理有问题")
    elif score['accuracy'] < 0.2:
        print("\n⚠️  结论：标签可能完全反转")
    
    print("=" * 80)
    # ========== 分析结束 ==========

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may find it helpful to see how the dataloader was created at other place in this code.

    original_train_dataset = dataset["train"]

    augmented_dataset = original_train_dataset.shuffle(seed=42).select(range(5000))

    augmented_dataset = augmented_dataset.map(custom_transform, load_from_cache_file=False)

    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets([original_train_dataset, augmented_dataset])

    combined_tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

    combined_tokenized_dataset = combined_tokenized_dataset.remove_columns(["text"])
    combined_tokenized_dataset = combined_tokenized_dataset.rename_column("label", "labels")
    combined_tokenized_dataset.set_format("torch")

    if args.debug_train:
        small_combined_dataset = combined_tokenized_dataset.shuffle(seed=42).select(range(4000))
        train_dataloader = DataLoader(small_combined_dataset, shuffle=True, batch_size=args.batch_size)
    else:
        train_dataloader = DataLoader(combined_tokenized_dataset, shuffle=True, batch_size=args.batch_size)
    
    print(f"Original training dataset size: {len(original_train_dataset)}")
    print(f"Augmented samples: 5000")
    print(f"Combined dataset size: {len(combined_tokenized_dataset)}")
    print(f"len(train_dataloader): {len(train_dataloader)}")

    ##### YOUR CODE ENDS HERE ######

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out")
        # Change eval dir
        args.model_dir = "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)

