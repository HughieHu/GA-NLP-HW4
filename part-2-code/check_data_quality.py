# check_data_quality_detailed.py
import re

def check_sql_issues_detailed(sql_file):
    """详细检查 SQL 文件中的问题"""
    with open(sql_file, 'r') as f:
        sqls = [line.strip() for line in f]
    
    issues = []
    
    for i, sql in enumerate(sqls):
        # 检查缺少运算符的模式
        # 匹配：列名 + 空格 + 数字，但后面不是 AND/OR/右括号
        matches = list(re.finditer(r'(\w+)\s+(\d+)(?!\s*(?:AND|OR|\)|,))', sql))
        if matches:
            issues.append({
                'line': i,
                'sql': sql,
                'matches': [(m.group(0), m.span()) for m in matches]
            })
    
    print(f"\n{'='*80}")
    print(f"Detailed Analysis: {sql_file}")
    print(f"{'='*80}")
    print(f"Total queries: {len(sqls)}")
    print(f"Queries with missing operators: {len(issues)} ({len(issues)/len(sqls)*100:.1f}%)")
    
    if issues:
        print(f"\nShowing first 10 examples:\n")
        for idx, issue in enumerate(issues[:10]):
            print(f"{idx+1}. Line {issue['line']}:")
            print(f"   Full SQL: {issue['sql']}")
            print(f"   Problems found:")
            for match, span in issue['matches']:
                print(f"     - '{match}' at position {span}")
            print()
    
    return issues

# 检查所有数据集
for split in ['train', 'dev']:
    issues = check_sql_issues_detailed(f'data/{split}.sql')