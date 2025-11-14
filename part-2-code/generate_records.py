# 保存为 generate_gt_records.py
import os
from tqdm import tqdm
from utils import save_queries_and_records

def generate_ground_truth_records():
    """为 train 和 dev 生成 ground truth records"""
    os.makedirs('records', exist_ok=True)
    
    for split in ['train', 'dev']:
        sql_path = f'data/{split}.sql'
        record_path = f'records/{split}_gt_records.pkl'
        
        print(f"\n{'='*50}")
        print(f"Generating ground truth records for {split} set...")
        print(f"{'='*50}")
        
        # 读取 SQL 查询
        with open(sql_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines()]
        
        print(f"Found {len(queries)} queries in {sql_path}")
        print(f"This may take a few minutes...")
        
        # 生成并保存 records（utils.py 中的 compute_records 已经有 tqdm 进度条）
        save_queries_and_records(queries, sql_path, record_path)
        
        print(f"✅ Saved ground truth records to {record_path}\n")

if __name__ == '__main__':
    print("\n" + "🚀 "*25)
    print("Starting Ground Truth Records Generation")
    print("🚀 "*25 + "\n")
    
    generate_ground_truth_records()
    
    print("\n" + "✨ "*25)
    print("All Ground Truth Records Generated Successfully!")
    print("✨ "*25 + "\n")
    
    # 验证生成的文件
    print("Verification:")
    for split in ['train', 'dev']:
        record_path = f'records/{split}_gt_records.pkl'
        if os.path.exists(record_path):
            size_mb = os.path.getsize(record_path) / 1024 / 1024
            print(f"  ✅ {record_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {record_path} NOT FOUND!")