# fix_tabs.py
with open('batch_test_all.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace tabs with 4 spaces
content = content.replace('\t', '    ')

with open('batch_test_all.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed indentation!")