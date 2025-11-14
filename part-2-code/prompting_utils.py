import os
import re

def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    if not os.path.exists(schema_path):
        print(f"Warning: Schema file not found at {schema_path}")
        return ""
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = f.read()
    
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    response = response.strip()
    
    sql_code_block = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_code_block:
        return sql_code_block.group(1).strip()
    
    code_block = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        content = code_block.group(1).strip()
        if any(keyword in content.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']):
            return content

    sql_prefix = re.search(r'(?:SQL|Query|Answer):\s*(.*?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
    if sql_prefix:
        return sql_prefix.group(1).strip()

    lines = response.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line_upper = line.strip().upper()

        if any(line_upper.startswith(keyword) for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']):
            in_sql = True
            sql_lines.append(line.strip())
        elif in_sql:

            if line.strip() == '':
                break
            elif line.strip().startswith(('#', '//', '--')):
                break
            else:
                sql_lines.append(line.strip())
    
    if sql_lines:
        return ' '.join(sql_lines)

    select_match = re.search(r'(SELECT\s+.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    print(f"Warning: Could not extract SQL from response: {response[:100]}...")
    return response.strip()


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    if not output_path:  
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\n")
        f.write(f"Record EM: {record_em}\n")
        f.write(f"Record F1: {record_f1}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Error Messages Summary:\n")
        f.write(f"{'='*80}\n")

        total_errors = sum(1 for msg in error_msgs if msg)
        f.write(f"Total errors: {total_errors} / {len(error_msgs)}\n\n")

        for i, msg in enumerate(error_msgs):
            if msg:  
                f.write(f"Query {i}: {msg}\n")
    
    print(f"Logs saved to {output_path}")
    
def clean_sql_query(sql_query):
    '''
    Clean and normalize SQL query
    '''
    sql_query = ' '.join(sql_query.split())

    sql_query = sql_query.rstrip(';')
    
    return sql_query.strip()