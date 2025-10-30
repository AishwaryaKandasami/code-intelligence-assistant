#!/usr/bin/env python3
"""
Code Intelligence Assistant - Production CLI
Simple interface for code review

Usage:
    python review.py file.py
    python review.py file.py --output review.json
    python review.py file.py --format markdown
"""

import sys
import json
from pathlib import Path
import argparse

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from hybrid_reviewer import HybridReviewer


def format_markdown(review_data: dict) -> str:
    """Format review as markdown."""
    review = review_data.get('review', {})
    
    md = [
        f"# Code Review: {review_data.get('file', 'Unknown')}",
        "",
        f"**Strategy**: {review_data.get('strategy', 'unknown')}",
        f"**Time**: {review_data['performance']['total_time']:.2f}s",
        "",
        "## Summary",
        "",
        review.get('summary', 'No summary available'),
        "",
    ]
    
    issues = review.get('issues', [])
    if issues:
        md.append("## Issues Found\n")
        for i, issue in enumerate(issues, 1):
            md.append(f"### {i}. {issue.get('title', 'Untitled')}")
            md.append(f"**Category**: {issue.get('category', 'N/A')}")
            md.append(f"**Severity**: {issue.get('severity', 'N/A')}")
            md.append(f"**Location**: {issue.get('location', 'N/A')}")
            md.append(f"\n{issue.get('description', '')}")
            md.append(f"\n**Suggestion**: {issue.get('suggestion', '')}\n")
    else:
        md.append("## ✅ No Issues Found\n")
    
    suggestions = review.get('suggestions', [])
    if suggestions:
        md.append("## Suggestions\n")
        for sug in suggestions:
            md.append(f"- {sug}")
        md.append("")
    
    return '\n'.join(md)


def format_terminal(review_data: dict) -> str:
    """Format review for terminal output."""
    review = review_data.get('review', {})
    perf = review_data.get('performance', {})
    
    lines = [
        "="*70,
        f"📝 CODE REVIEW: {review_data.get('file', 'Unknown')}",
        "="*70,
        "",
        f"Strategy: {review_data.get('strategy', 'unknown')}",
        f"Time: {perf.get('total_time', 0):.2f}s",
        f"Model: {perf.get('model_used', 'unknown')}",
        "",
        "📋 SUMMARY",
        "-"*70,
        review.get('summary', 'No summary'),
        "",
    ]
    
    issues = review.get('issues', [])
    if issues:
        lines.append(f"🔍 ISSUES FOUND: {len(issues)}")
        lines.append("-"*70)
        for i, issue in enumerate(issues, 1):
            severity_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(issue.get('severity', ''), '⚪')
            
            lines.append(f"\n{i}. {severity_emoji} [{issue.get('severity', 'N/A')}] {issue.get('title', 'Untitled')}")
            lines.append(f"   Category: {issue.get('category', 'N/A')}")
            lines.append(f"   Location: {issue.get('location', 'N/A')}")
            lines.append(f"   Issue: {issue.get('description', '')[:200]}...")
            lines.append(f"   Fix: {issue.get('suggestion', '')[:200]}...")
    else:
        lines.append("✅ NO ISSUES FOUND")
    
    lines.append("")
    lines.append("="*70)
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Code Intelligence Assistant - AI-powered code review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python review.py mycode.py
  python review.py mycode.py --output review.json
  python review.py mycode.py --format markdown > review.md
  python review.py mycode.py --no-ollama  (use OpenRouter only)
        """
    )
    
    parser.add_argument('file', help='Python file to review')
    parser.add_argument('--output', '-o', help='Save JSON output to file')
    parser.add_argument('--format', '-f', choices=['json', 'markdown', 'terminal'], 
                       default='terminal', help='Output format (default: terminal)')
    parser.add_argument('--no-ollama', action='store_true', 
                       help='Disable Ollama (use OpenRouter only)')
    parser.add_argument('--no-openrouter', action='store_true',
                       help='Disable OpenRouter (use Ollama only)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (errors only)')
    
    args = parser.parse_args()
    
    # Validate file
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Error: File not found: {args.file}", file=sys.stderr)
        return 1
    
    if not file_path.suffix == '.py':
        print(f"⚠ Warning: File doesn't appear to be Python (.py)", file=sys.stderr)
    
    # Read code
    try:
        code = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        return 1
    
    # Initialize reviewer
    try:
        if not args.quiet:
            print("Initializing Code Intelligence Assistant...", file=sys.stderr)
        
        reviewer = HybridReviewer(
            use_ollama=not args.no_ollama,
            use_openrouter=not args.no_openrouter
        )
    except Exception as e:
        print(f"❌ Error initializing reviewer: {e}", file=sys.stderr)
        return 1
    
    # Run review
    try:
        if not args.quiet:
            print(f"\n🔍 Reviewing {file_path}...\n", file=sys.stderr)
        
        result = reviewer.review_code(code, str(file_path))
    except Exception as e:
        print(f"❌ Error during review: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Save JSON if requested
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            if not args.quiet:
                print(f"\n💾 JSON saved to: {output_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Warning: Could not save JSON: {e}", file=sys.stderr)
    
    # Format output
    if args.format == 'json':
        print(json.dumps(result, indent=2))
    elif args.format == 'markdown':
        print(format_markdown(result))
    else:  # terminal
        print(format_terminal(result))
    
    # Exit code based on issues found
    issues = result.get('review', {}).get('issues', [])
    critical_issues = [i for i in issues if i.get('severity') in ['CRITICAL', 'HIGH']]
    
    if critical_issues:
        return 2  # Critical issues found
    elif issues:
        return 1  # Issues found
    else:
        return 0  # No issues


if __name__ == '__main__':
    sys.exit(main())