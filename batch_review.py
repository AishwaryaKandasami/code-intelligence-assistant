#!/usr/bin/env python3
"""
Batch Code Review - Review multiple files or directories

Usage:
    python batch_review.py src/
    python batch_review.py file1.py file2.py file3.py
    python batch_review.py src/ --output-dir reviews/
"""

import sys
import json
from pathlib import Path
import argparse
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from hybrid_reviewer import HybridReviewer


def find_python_files(paths: list) -> list:
    """Find all Python files in given paths."""
    files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.py':
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob('*.py'))
    return sorted(set(files))


def format_summary(results: list) -> str:
    """Generate summary report."""
    total = len(results)
    with_issues = sum(1 for r in results if r.get('issues_count', 0) > 0)
    critical = sum(1 for r in results 
                  if any(i.get('severity') == 'CRITICAL' 
                        for i in r.get('review', {}).get('issues', [])))
    high = sum(1 for r in results
              if any(i.get('severity') == 'HIGH'
                    for i in r.get('review', {}).get('issues', [])))
    
    total_time = sum(r.get('performance', {}).get('total_time', 0) for r in results)
    total_issues = sum(len(r.get('review', {}).get('issues', [])) for r in results)
    
    lines = [
        "",
        "="*70,
        "📊 BATCH REVIEW SUMMARY",
        "="*70,
        f"Files reviewed: {total}",
        f"Files with issues: {with_issues} ({with_issues/total*100:.1f}%)",
        f"Total issues found: {total_issues}",
        f"  🔴 Critical: {critical}",
        f"  🟠 High: {high}",
        f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)",
        f"Average: {total_time/total:.1f}s per file",
        "="*70,
    ]
    
    # List files with critical/high issues
    critical_files = [r for r in results
                     if any(i.get('severity') in ['CRITICAL', 'HIGH']
                           for i in r.get('review', {}).get('issues', []))]
    
    if critical_files:
        lines.append("\n🚨 FILES REQUIRING IMMEDIATE ATTENTION:\n")
        for result in critical_files:
            file = result.get('file', 'unknown')
            issues = [i for i in result.get('review', {}).get('issues', [])
                     if i.get('severity') in ['CRITICAL', 'HIGH']]
            lines.append(f"  {file}")
            for issue in issues:
                lines.append(f"    - [{issue['severity']}] {issue['title']}")
        lines.append("")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Batch Code Review - Review multiple Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_review.py src/
  python batch_review.py file1.py file2.py
  python batch_review.py src/ --output-dir reviews/ --format markdown
  python batch_review.py src/ --critical-only
        """
    )
    
    parser.add_argument('paths', nargs='+', help='Files or directories to review')
    parser.add_argument('--output-dir', '-o', help='Directory for review outputs')
    parser.add_argument('--format', '-f', choices=['json', 'markdown'], 
                       default='json', help='Output format')
    parser.add_argument('--critical-only', action='store_true',
                       help='Only show files with critical/high issues')
    parser.add_argument('--parallel', '-p', type=int, metavar='N',
                       help='Review N files in parallel (not implemented yet)')
    parser.add_argument('--no-ollama', action='store_true')
    parser.add_argument('--no-openrouter', action='store_true')
    
    args = parser.parse_args()
    
    # Find files
    print("🔍 Finding Python files...", file=sys.stderr)
    files = find_python_files(args.paths)
    
    if not files:
        print("❌ No Python files found", file=sys.stderr)
        return 1
    
    print(f"📁 Found {len(files)} Python files", file=sys.stderr)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Outputs will be saved to: {output_dir}", file=sys.stderr)
    
    # Initialize reviewer
    print("\n🤖 Initializing Code Intelligence Assistant...", file=sys.stderr)
    try:
        reviewer = HybridReviewer(
            use_ollama=not args.no_ollama,
            use_openrouter=not args.no_openrouter
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    # Review files
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Starting batch review of {len(files)} files...", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    
    results = []
    start_time = time.time()
    
    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Reviewing {file}...", file=sys.stderr)
        
        try:
            code = file.read_text(encoding='utf-8')
            result = reviewer.review_code(code, str(file))
            results.append(result)
            
            issues_count = len(result.get('review', {}).get('issues', []))
            duration = result.get('performance', {}).get('total_time', 0)
            
            print(f"  ✓ Complete: {issues_count} issues, {duration:.1f}s", file=sys.stderr)
            
            # Save individual review
            if args.output_dir:
                output_file = output_dir / f"{file.stem}_review.{args.format}"
                
                if args.format == 'json':
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                else:  # markdown
                    from review import format_markdown
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(format_markdown(result))
                
                print(f"  💾 Saved to {output_file}", file=sys.stderr)
            
        except Exception as e:
            print(f"  ❌ Error: {e}", file=sys.stderr)
            results.append({
                'file': str(file),
                'error': str(e),
                'review': {'issues': [], 'suggestions': []},
                'performance': {'total_time': 0}
            })
        
        print("", file=sys.stderr)
    
    total_time = time.time() - start_time
    
    # Filter if requested
    if args.critical_only:
        results = [r for r in results
                  if any(i.get('severity') in ['CRITICAL', 'HIGH']
                        for i in r.get('review', {}).get('issues', []))]
    
    # Print summary
    print(format_summary(results), file=sys.stderr)
    
    # Save summary
    if args.output_dir:
        summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_files': len(files),
                'reviewed': len(results),
                'total_time': total_time,
                'results': results
            }, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_file}", file=sys.stderr)
    
    # Exit code
    critical_count = sum(1 for r in results
                        if any(i.get('severity') == 'CRITICAL'
                              for i in r.get('review', {}).get('issues', [])))
    
    if critical_count > 0:
        return 2
    elif results:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())