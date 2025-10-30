#!/bin/bash
# Repository Cleanup Script
# Run this to clean up old/redundant files

echo "🧹 Code Intelligence Assistant - Repository Cleanup"
echo "=================================================="
echo ""

# Files to remove (obsolete/replaced)
FILES_TO_REMOVE=(
    "scripts/07_endreview.py"
    "scripts/old_reviewer.py"
    "test_hybrid.json"
    ".vscode/settings.json"
)

# Directories to clean
DIRS_TO_CLEAN=(
    "cache"
    "__pycache__"
    "*.pyc"
)

echo "📂 Files marked for removal:"
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  ❌ $file"
        rm "$file"
    fi
done

echo ""
echo "📁 Cleaning directories:"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "  ✓ Cleaned Python cache files"

echo ""
echo "📁 Organizing structure:"

# Create necessary directories
mkdir -p tests/test_cases
mkdir -p data/embeddings
mkdir -p reviews
mkdir -p docs

echo "  ✓ Created directory structure"

# Move test files to tests/
if [ -d "test_cases" ] && [ ! -d "tests/test_cases" ]; then
    echo "  → Moving test_cases/ to tests/"
    mv test_cases tests/
fi

if [ -f "batch_test_all.py" ]; then
    echo "  → Moving batch_test_all.py to tests/"
    mv batch_test_all.py tests/
fi

if [ -f "comprehensive_test_cases.py" ]; then
    echo "  → Moving comprehensive_test_cases.py to tests/"
    mv comprehensive_test_cases.py tests/
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit cleanup: git add . && git commit -m 'chore: cleanup repository structure'"
echo "  3. Run tests: python tests/batch_test_all.py"