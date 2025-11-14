# generate_latex_table.py (完整独立版本)
import json
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import os
import re

# ============================================================================
# ErrorAnalyzer Class (从 error_analysis.py 复制过来)
# ============================================================================

class ErrorAnalyzer:
    def __init__(self, 
                 pred_sql_path: str,
                 pred_records_path: str, 
                 gt_sql_path: str,
                 gt_records_path: str,
                 questions_path: str = None):
        """
        初始化错误分析器
        """
        self.pred_sql_path = pred_sql_path
        self.pred_records_path = pred_records_path
        self.gt_sql_path = gt_sql_path
        self.gt_records_path = gt_records_path
        self.questions_path = questions_path
        
        # 加载数据
        self.pred_sqls = self._load_sql(pred_sql_path)
        self.pred_records = self._load_records(pred_records_path)
        self.gt_sqls = self._load_sql(gt_sql_path)
        self.gt_records = self._load_records(gt_records_path)
        self.questions = self._load_questions(questions_path) if questions_path else None
        
        # 错误统计
        self.errors = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        print(f"✅ Loaded {len(self.pred_sqls)} predictions")
        print(f"✅ Loaded {len(self.gt_sqls)} ground truth examples")
    
    def _load_sql(self, path: str) -> List[str]:
        """加载SQL文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"SQL file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            sqls = [line.strip() for line in f]
        return sqls
    
    def _load_records(self, path: str):
        """加载pickle记录文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Records file not found: {path}")
        
        with open(path, 'rb') as f:
            records = pickle.load(f)
        
        if isinstance(records, tuple):
            print(f"⚠️  Converting records from tuple to list")
            records = list(records)
        
        return records
    
    def _load_questions(self, path: str) -> List[str]:
        """加载问题文件"""
        if not os.path.exists(path):
            print(f"⚠️  Questions file not found: {path}")
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]
        return questions
    
    def _normalize_sql(self, sql: str) -> str:
        """标准化SQL用于比较"""
        return ' '.join(sql.lower().split())
    
    def _extract_clause(self, sql: str, clause: str) -> str:
        """提取SQL子句"""
        sql_lower = sql.lower()
        if clause not in sql_lower:
            return ""
        
        start = sql_lower.find(clause)
        next_clauses = ['from', 'where', 'group by', 'having', 'order by', 'limit']
        
        end = len(sql)
        for next_clause in next_clauses:
            if next_clause == clause:
                continue
            pos = sql_lower.find(next_clause, start + len(clause))
            if pos != -1 and pos < end:
                end = pos
        
        return sql[start:end].strip()
    
    def _get_record(self, records, idx):
        """安全地获取record"""
        try:
            if isinstance(records, (list, tuple)):
                if idx < len(records):
                    return records[idx]
            elif isinstance(records, dict):
                return records.get(idx, None)
            return None
        except:
            return None
    
    def categorize_error(self, idx: int) -> List[str]:
        """对单个样本进行错误分类"""
        if idx >= len(self.pred_sqls) or idx >= len(self.gt_sqls):
            return ['index_out_of_range']
        
        pred_sql = self.pred_sqls[idx]
        gt_sql = self.gt_sqls[idx]
        
        pred_records = self._get_record(self.pred_records, idx)
        gt_records = self._get_record(self.gt_records, idx)
        
        errors = []
        
        pred_norm = self._normalize_sql(pred_sql)
        gt_norm = self._normalize_sql(gt_sql)
        
        # 1. 完全正确
        if pred_norm == gt_norm:
            return ['correct']
        
        # 2. 结果正确但SQL不同
        if pred_records == gt_records and pred_records is not None:
            errors.append('correct_result_different_sql')
            return errors
        
        # 3. 语法错误
        if pred_records is None:
            errors.append('syntax_error')
            
            pred_lower = pred_sql.lower().strip()
            
            if pred_sql.strip() == '':
                errors.append('syntax_empty_prediction')
                return errors
            
            if 'select' not in pred_lower:
                errors.append('syntax_missing_select_keyword')
                return errors
            
            if len(pred_sql.split()) < 3:
                errors.append('syntax_incomplete_sql_too_short')
                return errors
            
            if pred_sql.count('(') != pred_sql.count(')'):
                errors.append('syntax_unmatched_parentheses')
            
            single_quotes = pred_sql.count("'")
            if single_quotes % 2 != 0:
                errors.append('syntax_unmatched_single_quotes')
            
            if 'select' in pred_lower and 'from' not in pred_lower:
                errors.append('syntax_missing_from_clause')
            
            keywords = ['select', 'from', 'where', 'group by', 'order by', 'limit']
            for keyword in keywords:
                count = pred_lower.count(keyword)
                if count > 1 and keyword not in ['select']:
                    errors.append(f'syntax_duplicate_{keyword.replace(" ", "_")}_keyword')
            
            if 'join' in pred_lower and ' on ' not in pred_lower and ' using' not in pred_lower:
                errors.append('syntax_join_missing_on_condition')
            
            agg_funcs = ['count', 'sum', 'avg', 'max', 'min']
            for func in agg_funcs:
                if func in pred_lower:
                    func_idx = pred_lower.find(func)
                    remaining = pred_lower[func_idx + len(func):].strip()
                    if not remaining.startswith('('):
                        errors.append(f'syntax_aggregation_{func}_missing_parentheses')
            
            if pred_lower.endswith(('where', 'and', 'or', 'join', 'on', 'order by', 'group by')):
                errors.append('syntax_incomplete_ends_with_keyword')
            
            if len(errors) == 1:
                errors.append('syntax_error_unknown_cause')
            
            return errors
        
        # 4. 语义错误
        pred_lower = pred_sql.lower()
        gt_lower = gt_sql.lower()
        
        # SELECT子句分析
        pred_select = self._extract_clause(pred_sql, 'select')
        gt_select = self._extract_clause(gt_sql, 'select')
        
        if self._normalize_sql(pred_select) != self._normalize_sql(gt_select):
            if '*' in pred_select and '*' not in gt_select:
                errors.append('select_star_instead_of_specific_columns')
            elif '*' not in pred_select and '*' in gt_select:
                errors.append('select_specific_columns_instead_of_star')
            
            agg_funcs = ['count', 'sum', 'avg', 'max', 'min']
            for func in agg_funcs:
                if func in gt_select and func not in pred_select:
                    errors.append(f'missing_aggregation_{func}')
                elif func in pred_select and func not in gt_select:
                    errors.append(f'unnecessary_aggregation_{func}')
            
            if not any(err.startswith('select_') or 'aggregation' in err for err in errors):
                errors.append('wrong_columns_selected')
        
        # JOIN分析
        if 'join' in gt_lower and 'join' not in pred_lower:
            errors.append('missing_join')
        elif 'join' in pred_lower and 'join' not in gt_lower:
            errors.append('unnecessary_join')
        
        # WHERE子句分析
        if 'where' in gt_lower and 'where' not in pred_lower:
            errors.append('missing_where_clause')
        elif 'where' not in gt_lower and 'where' in pred_lower:
            errors.append('unnecessary_where_clause')
        elif 'where' in gt_lower and 'where' in pred_lower:
            pred_where = self._extract_clause(pred_sql, 'where')
            gt_where = self._extract_clause(gt_sql, 'where')
            
            if self._normalize_sql(pred_where) != self._normalize_sql(gt_where):
                errors.append('wrong_where_condition_value')
        
        # GROUP BY分析
        if 'group by' in gt_lower and 'group by' not in pred_lower:
            errors.append('missing_group_by')
        
        # ORDER BY分析
        if 'order by' in gt_lower and 'order by' not in pred_lower:
            errors.append('missing_order_by')
        
        # LIMIT分析
        if 'limit' in gt_lower and 'limit' not in pred_lower:
            errors.append('missing_limit')
        
        # DISTINCT分析
        if 'distinct' in gt_lower and 'distinct' not in pred_lower:
            errors.append('missing_distinct')
        
        if not errors:
            errors.append('other_semantic_error')
        
        return errors
    
    def analyze(self):
        """执行完整的错误分析"""
        total_examples = len(self.gt_sqls)
        correct_count = 0
        correct_result_count = 0
        
        for idx in range(total_examples):
            try:
                error_types = self.categorize_error(idx)
                
                if 'correct' in error_types:
                    correct_count += 1
                    continue
                
                if 'correct_result_different_sql' in error_types:
                    correct_result_count += 1
                
                question = self.questions[idx] if self.questions and idx < len(self.questions) else f"Example {idx}"
                pred_rec = self._get_record(self.pred_records, idx)
                gt_rec = self._get_record(self.gt_records, idx)
                
                for error_type in error_types:
                    self.error_counts[error_type] += 1
                    self.errors[error_type].append({
                        'index': idx,
                        'question': question,
                        'predicted_sql': self.pred_sqls[idx],
                        'ground_truth_sql': self.gt_sqls[idx],
                        'predicted_records': str(pred_rec)[:100] if pred_rec else None,
                        'ground_truth_records': str(gt_rec)[:100] if gt_rec else None
                    })
            except Exception as e:
                print(f"⚠️  Error analyzing example {idx}: {e}")
                continue
        
        return self.errors, self.error_counts


# ============================================================================
# LaTeXTableGenerator Class
# ============================================================================

class LaTeXTableGenerator:
    def __init__(self, baseline_analyzer: ErrorAnalyzer, finetuned_analyzer: ErrorAnalyzer):
        self.baseline_analyzer = baseline_analyzer
        self.finetuned_analyzer = finetuned_analyzer
        
        # Error descriptions
        self.error_descriptions = {
            'syntax_duplicate_from_keyword': 'The model generates duplicate FROM keywords, resulting in invalid syntax (e.g., "SELECT * FROM FROM table").',
            'syntax_unmatched_parentheses': 'Mismatched parentheses in the query, with opening parentheses not properly closed.',
            'syntax_error_unknown_cause': 'Syntax error preventing execution, but specific cause unclear from automated analysis.',
            'syntax_duplicate_where_keyword': 'Multiple WHERE keywords in a single query, violating SQL syntax.',
            'syntax_missing_from_clause': 'SELECT statement present but missing required FROM clause.',
            'wrong_columns_selected': 'Incorrect columns selected, leading to wrong results despite valid syntax.',
            'missing_where_clause': 'Missing WHERE clause needed for filtering results.',
            'missing_join': 'Query requires joining tables but JOIN is missing.',
            'missing_aggregation_count': 'Should use COUNT() aggregation but does not.',
            'missing_group_by': 'Uses aggregation functions without required GROUP BY clause.',
            'missing_order_by': 'Should sort results with ORDER BY but does not.',
            'wrong_table_reference': 'References incorrect table(s) in the FROM clause.',
        }
    
    def get_top_errors(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """获取最常见的错误类型"""
        all_errors = {}
        
        for error_type in set(self.baseline_analyzer.error_counts.keys()) | set(self.finetuned_analyzer.error_counts.keys()):
            if error_type in ['correct', 'correct_result_different_sql']:
                continue
            
            bl_count = self.baseline_analyzer.error_counts.get(error_type, 0)
            ft_count = self.finetuned_analyzer.error_counts.get(error_type, 0)
            
            all_errors[error_type] = {
                'baseline': bl_count,
                'finetuned': ft_count,
                'total': bl_count + ft_count
            }
        
        sorted_errors = sorted(all_errors.items(), key=lambda x: x[1]['total'], reverse=True)
        return sorted_errors[:n]
    
    def get_error_example(self, error_type: str, model_type: str = 'baseline') -> Dict:
        """获取特定错误类型的示例"""
        analyzer = self.baseline_analyzer if model_type == 'baseline' else self.finetuned_analyzer
        
        if error_type not in analyzer.errors or len(analyzer.errors[error_type]) == 0:
            return None
        
        return analyzer.errors[error_type][0]
    
    def format_sql_for_latex(self, sql: str, max_length: int = 45) -> str:
        """格式化SQL用于LaTeX"""
        # 转义特殊字符
        replacements = {
            '\\': '\\textbackslash{}',
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        
        for old, new in replacements.items():
            sql = sql.replace(old, new)
        
        if len(sql) > max_length:
            sql = sql[:max_length] + '...'
        
        return f"\\texttt{{{sql}}}"
    
    def generate_latex_row(self, error_type: str, error_info: Dict) -> str:
        """生成单行LaTeX表格"""
        description = self.error_descriptions.get(error_type, 'Error description not available.')
        
        bl_count = error_info['baseline']
        ft_count = error_info['finetuned']
        
        examples = []
        statistics = []
        
        # Baseline example
        if bl_count > 0:
            bl_example = self.get_error_example(error_type, 'baseline')
            if bl_example:
                pred_sql = self.format_sql_for_latex(bl_example['predicted_sql'])
                examples.append(f"\\textbf{{BL:}} {pred_sql}")
                statistics.append(f"\\textbf{{BL:}} {bl_count}/{len(self.baseline_analyzer.gt_sqls)}")
        
        # Fine-tuned example
        if ft_count > 0:
            ft_example = self.get_error_example(error_type, 'finetuned')
            if ft_example:
                pred_sql = self.format_sql_for_latex(ft_example['predicted_sql'])
                examples.append(f"\\textbf{{FT:}} {pred_sql}")
                statistics.append(f"\\textbf{{FT:}} {ft_count}/{len(self.finetuned_analyzer.gt_sqls)}")
        
        example_str = ' \\newline '.join(examples) if examples else 'N/A'
        stats_str = ' \\newline '.join(statistics) if statistics else 'N/A'
        
        error_name = error_type.replace('_', ' ').title()
        
        row = f"    {error_name} & {example_str} & {description} & {stats_str} \\\\\n    \\midrule"
        
        return row
    
    def generate_simplified_table(self, selected_errors: List[str], output_file: str = 'error_table.tex'):
        """生成简化表格"""
        latex_table = r"""\begin{table}[h]
  \centering
  \footnotesize
  \begin{tabular}{p{2.2cm}p{4.5cm}p{5cm}p{2cm}}
    \toprule
    \textbf{Error Type} & \textbf{Example} & \textbf{Description} & \textbf{Stats} \\
    \midrule
"""
        
        for error_type in selected_errors:
            bl_count = self.baseline_analyzer.error_counts.get(error_type, 0)
            ft_count = self.finetuned_analyzer.error_counts.get(error_type, 0)
            
            error_info = {
                'baseline': bl_count,
                'finetuned': ft_count,
                'total': bl_count + ft_count
            }
            
            row = self.generate_latex_row(error_type, error_info)
            latex_table += row + "\n"
        
        latex_table += r"""    \bottomrule
  \end{tabular}
  \caption{Qualitative error analysis comparing Baseline (BL: zero-shot T5) and Fine-tuned (FT) models on dev set.}
  \label{tab:qualitative}
\end{table}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"✅ LaTeX table saved to: {output_file}")
        return latex_table
    
    def generate_human_readable_report(self, top_n: int = 15, output_file: str = 'error_report.txt'):
        """生成人类可读报告"""
        top_errors = self.get_top_errors(top_n)
        
        report = "="*100 + "\n"
        report += "ERROR ANALYSIS REPORT - FOR LATEX TABLE\n"
        report += "="*100 + "\n\n"
        
        for i, (error_type, error_info) in enumerate(top_errors, 1):
            report += f"\n{'='*100}\n"
            report += f"ERROR #{i}: {error_type}\n"
            report += f"{'='*100}\n\n"
            
            description = self.error_descriptions.get(error_type, 'N/A')
            report += f"DESCRIPTION:\n{description}\n\n"
            
            bl_count = error_info['baseline']
            ft_count = error_info['finetuned']
            bl_total = len(self.baseline_analyzer.gt_sqls)
            ft_total = len(self.finetuned_analyzer.gt_sqls)
            
            report += "STATISTICS:\n"
            if bl_count > 0:
                report += f"  Baseline:    {bl_count}/{bl_total} ({bl_count/bl_total*100:.1f}%)\n"
            if ft_count > 0:
                report += f"  Fine-tuned:  {ft_count}/{ft_total} ({ft_count/ft_total*100:.1f}%)\n"
            report += "\n"
            
            report += "EXAMPLES:\n\n"
            
            if bl_count > 0:
                bl_example = self.get_error_example(error_type, 'baseline')
                if bl_example:
                    report += "  BASELINE:\n"
                    report += f"    Q: {bl_example.get('question', 'N/A')}\n"
                    report += f"    Pred: {bl_example['predicted_sql']}\n"
                    report += f"    GT:   {bl_example['ground_truth_sql']}\n\n"
            
            if ft_count > 0:
                ft_example = self.get_error_example(error_type, 'finetuned')
                if ft_example:
                    report += "  FINE-TUNED:\n"
                    report += f"    Q: {ft_example.get('question', 'N/A')}\n"
                    report += f"    Pred: {ft_example['predicted_sql']}\n"
                    report += f"    GT:   {ft_example['ground_truth_sql']}\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {output_file}")
        return report


# ============================================================================
# Main Function (修复版)
# ============================================================================

def main():
    print("\n" + "="*100)
    print("📊 GENERATING LATEX TABLE FOR ERROR ANALYSIS")
    print("="*100 + "\n")
    
    # 创建输出目录
    os.makedirs('baseline', exist_ok=True)
    print("✅ Created output directory: baseline/\n")
    
    # 文件路径
    baseline_pred_sql = 'part-2-code/baseline/dev_predictions.sql'
    baseline_pred_records = 'part-2-code/baseline/dev_predictions_records.pkl'
    
    finetuned_pred_sql = 'part-2-code/test_results/dev_predictions.sql'
    finetuned_pred_records = 'part-2-code/test_results/dev_predictions_records.pkl'
    
    gt_sql_path = 'part-2-code/data/dev.sql'
    gt_records_path = 'part-2-code/records/dev_gt_records.pkl'
    questions_path = 'part-2-code/data/dev.nl'
    
    # 检查文件是否存在
    required_files = [
        baseline_pred_sql,
        baseline_pred_records,
        finetuned_pred_sql,
        finetuned_pred_records,
        gt_sql_path,
        gt_records_path
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Please update the file paths in main() to match your directory structure.")
        return
    
    # 创建分析器
    print("📂 Loading baseline...")
    baseline_analyzer = ErrorAnalyzer(
        pred_sql_path=baseline_pred_sql,
        pred_records_path=baseline_pred_records,
        gt_sql_path=gt_sql_path,
        gt_records_path=gt_records_path,
        questions_path=questions_path
    )
    baseline_analyzer.analyze()
    
    print("\n📂 Loading fine-tuned...")
    finetuned_analyzer = ErrorAnalyzer(
        pred_sql_path=finetuned_pred_sql,
        pred_records_path=finetuned_pred_records,
        gt_sql_path=gt_sql_path,
        gt_records_path=gt_records_path,
        questions_path=questions_path
    )
    finetuned_analyzer.analyze()
    
    # 创建表格生成器
    generator = LaTeXTableGenerator(baseline_analyzer, finetuned_analyzer)
    
    # 生成报告
    print("\n📝 Generating report...")
    generator.generate_human_readable_report(top_n=15, output_file='baseline/error_report.txt')
    
    # 选择3-5个最具代表性的错误
    selected_errors = [
        'syntax_duplicate_from_keyword',  # Baseline特有的语法错误
        'wrong_columns_selected',         # 语义错误 - 列选择
        'missing_where_clause',           # 语义错误 - 过滤
    ]
    
    print("\n📝 Generating LaTeX table...")
    generator.generate_simplified_table(selected_errors, output_file='baseline/error_table.tex')
    
    print("\n" + "="*100)
    print("✨ COMPLETE!")
    print("="*100)
    print("\n📁 Files:")
    print("   LaTeX table:  baseline/error_table.tex")
    print("   Report:       baseline/error_report.txt")
    print("\n💡 Check the report to select the best errors for your table!")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()