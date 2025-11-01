"""
Streamlit UI for Code Intelligence Assistant
Wraps your existing hybrid_reviewer.py backend

Save as: streamlit_app.py
Run: streamlit run streamlit_app.py
"""

from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
import subprocess
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime
import sys

# Page config
st.set_page_config(
    page_title="Code Intelligence Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .issue-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .issue-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .issue-medium {
        background-color: #fff9c4;
        border-left: 4px solid #fbc02d;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .issue-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #1f77b4, #2196f3);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'review_history' not in st.session_state:
    st.session_state.review_history = []
if 'current_review' not in st.session_state:
    st.session_state.current_review = None

# Check if hybrid_reviewer.py exists
REVIEWER_SCRIPT = Path("scripts/hybrid_reviewer.py")
if not REVIEWER_SCRIPT.exists():
    st.error(f"❌ hybrid_reviewer.py not found at: {REVIEWER_SCRIPT}")
    st.info("Please ensure you're running from the project root directory")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Reviewer Settings")
    
    # Model options
    use_ollama = st.checkbox("Use Ollama (Local)", value=True, help="Fast local screening with Qwen2.5-Coder")
    use_openrouter = st.checkbox("Use OpenRouter (Cloud)", value=True, help="Detailed reviews with Llama 3.1")
    
    if not use_ollama and not use_openrouter:
        st.warning("⚠️ Enable at least one model!")
    
    st.markdown("---")
    
    # Strategy info
    st.markdown("### 🎯 Review Strategy")
    if use_ollama and use_openrouter:
        st.success("**Hybrid Mode** (Recommended)")
        st.caption("Fast Ollama screening → Detailed OpenRouter review for critical issues")
    elif use_ollama:
        st.info("**Ollama Only**")
        st.caption("Fast local reviews")
    elif use_openrouter:
        st.info("**OpenRouter Only**")
        st.caption("Detailed cloud reviews")
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Check Ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=3)
        if result.returncode == 0:
            st.success("✅ Ollama Ready")
        else:
            st.warning("⚠️ Ollama Not Ready")
    except:
        st.warning("⚠️ Ollama Not Detected")
    
    # Check OpenRouter
    import os
    if os.getenv('OPENROUTER_API_KEY'):
        st.success("✅ OpenRouter API Key Set")
    else:
        st.warning("⚠️ No OpenRouter API Key")
        st.caption("Set in .env file")
    
    st.markdown("---")
    
    # Training sources
    st.markdown("### 📚 Learning From")
    st.markdown("""
    - ✅ FastAPI (API design)
    - ✅ Requests (Error handling)
    - ✅ Pydantic (Validation)
    
    **~150-200 PR reviews**
    """)
    
    st.markdown("---")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.review_history = []
        st.session_state.current_review = None
        st.rerun()
    
    # Stats
    if st.session_state.review_history:
        st.markdown("### 📈 Session Stats")
        st.metric("Reviews", len(st.session_state.review_history))
        total_issues = sum(len(r.get('review', {}).get('issues', [])) for r in st.session_state.review_history)
        st.metric("Issues Found", total_issues)

# Main header
st.markdown('<div class="main-header">🔍 Code Intelligence Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered code review using Ollama + OpenRouter hybrid system</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Review Code", "📊 Results", "📜 History", "ℹ️ About"])

with tab1:
    st.markdown("### 💻 Submit Code for Review")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Paste Code", "Upload File"],
            horizontal=True
        )
        
        code_input = None
        filename = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Python file",
                type=['py'],
                help="Select a .py file to review"
            )
            if uploaded_file:
                code_input = uploaded_file.read().decode('utf-8')
                filename = uploaded_file.name
                st.success(f"✅ Loaded: {filename}")
        else:
            code_input = st.text_area(
                "Paste your Python code here",
                height=400,
                placeholder="""def example_function(data):
    # Your code here
    result = process_data(data)
    return result
    
class DataHandler:
    def __init__(self):
        self.data = []
""",
                help="Paste the Python code you want reviewed"
            )
            filename = "pasted_code.py"
        
        # Show code preview
        if code_input:
            with st.expander("📋 Code Preview", expanded=False):
                st.code(code_input, language='python', line_numbers=True)
    
    with col2:
        st.markdown("### 🚀 Actions")
        
        # Review button
        review_clicked = st.button("🔍 Start Review", type="primary", disabled=not code_input)
        
        if review_clicked:
            if not use_ollama and not use_openrouter:
                st.error("❌ Please enable at least one model in the sidebar!")
            else:
                # Create temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(code_input)
                    temp_input = f.name
                
                # Create temp output
                temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
                
                # Build command
                cmd = [
                    sys.executable,
                    str(REVIEWER_SCRIPT),
                    '--file', temp_input,
                    '--output', temp_output
                ]
                
                if not use_ollama:
                    cmd.append('--no-ollama')
                if not use_openrouter:
                    cmd.append('--no-openrouter')
                
                # Run review
                with st.spinner("🔄 Analyzing code... This may take 20-40 seconds"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Start process
                        status_text.text("⚡ Initializing hybrid reviewer...")
                        progress_bar.progress(10)
                        
                        start_time = time.time()
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            encoding='utf-8'
                        )
                        
                        progress_bar.progress(50)
                        status_text.text("📊 Processing results...")
                        
                        duration = time.time() - start_time
                        
                        if result.returncode == 0:
                            # Load results
                            with open(temp_output, 'r', encoding='utf-8') as f:
                                review_data = json.load(f)
                            
                            # Add metadata
                            review_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            review_data['filename'] = filename
                            review_data['code'] = code_input[:500] + "..." if len(code_input) > 500 else code_input
                            review_data['duration'] = duration
                            
                            # Save to history
                            st.session_state.review_history.append(review_data)
                            st.session_state.current_review = review_data
                            
                            progress_bar.progress(100)
                            status_text.empty()
                            st.success(f"✅ Review completed in {duration:.1f}s!")
                            
                            # Clean up
                            Path(temp_input).unlink(missing_ok=True)
                            Path(temp_output).unlink(missing_ok=True)
                            
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Review failed!")
                            st.error(f"Error: {result.stderr}")
                            
                    except subprocess.TimeoutExpired:
                        st.error("⏰ Review timed out after 120 seconds")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    finally:
                        # Cleanup
                        Path(temp_input).unlink(missing_ok=True)
                        Path(temp_output).unlink(missing_ok=True)
        
        # Tips
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.caption("""
        - **Hybrid mode** gives best results
        - **Critical issues** get detailed review
        - **Response time**: 20-40s
        - Review saved to history
        """)

with tab2:
    st.markdown("### 📊 Review Results")
    
    if st.session_state.current_review:
        review = st.session_state.current_review
        review_content = review.get('review', {})
        performance = review.get('performance', {})
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        issues = review_content.get('issues', [])
        
        with col1:
            st.metric("Issues Found", len(issues))
        with col2:
            st.metric("Strategy", review.get('strategy', 'unknown'))
        with col3:
            st.metric("Review Time", f"{performance.get('total_time', 0):.1f}s")
        with col4:
            quality = review_content.get('overall_quality', 'UNKNOWN')
            st.metric("Quality", quality)
        
        st.markdown("---")
        
        # Summary
        summary = review_content.get('summary', '')
        if summary:
            st.markdown("#### 📋 Summary")
            st.info(summary)
        
        # Issues by severity
        if issues:
            st.markdown("#### 🔍 Issues Detected")
            
            # Group by severity
            by_severity = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
            for issue in issues:
                severity = issue.get('severity', 'MEDIUM')
                by_severity[severity].append(issue)
            
            # Display by severity
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                severity_issues = by_severity[severity]
                if severity_issues:
                    emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
                    st.markdown(f"##### {emoji} {severity} ({len(severity_issues)})")
                    
                    for i, issue in enumerate(severity_issues, 1):
                        css_class = f"issue-{severity.lower()}"
                        
                        with st.expander(f"{i}. {issue.get('title', 'Issue')}", expanded=(severity in ['CRITICAL', 'HIGH'])):
                            st.markdown(f"**Category**: {issue.get('category', 'UNKNOWN')}")
                            st.markdown(f"**Location**: {issue.get('location', 'N/A')}")
                            st.markdown(f"**Description**: {issue.get('description', '')}")
                            
                            suggestion = issue.get('suggestion', '')
                            if suggestion:
                                st.markdown(f"**💡 Suggestion**: {suggestion}")
        else:
            st.success("✨ No issues detected! Code looks good.")
        
        # General suggestions
        suggestions = review_content.get('suggestions', [])
        if suggestions:
            st.markdown("#### 💡 General Suggestions")
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        
        # Performance details
        with st.expander("⚡ Performance Details"):
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Screening Time", f"{performance.get('screening_time', 0):.2f}s")
            with perf_col2:
                st.metric("Detailed Time", f"{performance.get('detailed_time', 0):.2f}s")
            with perf_col3:
                st.metric("Model", performance.get('model_used', 'N/A'))
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Results as JSON"):
            st.download_button(
                "Download JSON",
                data=json.dumps(review, indent=2),
                file_name=f"review_{review['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("👈 Submit code for review in the 'Review Code' tab to see results here")

with tab3:
    st.markdown("### 📜 Review History")
    
    if st.session_state.review_history:
        st.markdown(f"**Total Reviews**: {len(st.session_state.review_history)}")
        
        for i, review in enumerate(reversed(st.session_state.review_history), 1):
            idx = len(st.session_state.review_history) - i
            
            with st.expander(
                f"#{idx + 1} - {review.get('filename', 'unknown')} - {review.get('timestamp', 'N/A')}",
                expanded=(i == 1)
            ):
                col1, col2, col3 = st.columns(3)
                
                issues_count = len(review.get('review', {}).get('issues', []))
                strategy = review.get('strategy', 'unknown')
                duration = review.get('duration', 0)
                
                with col1:
                    st.metric("Issues", issues_count)
                with col2:
                    st.metric("Strategy", strategy)
                with col3:
                    st.metric("Time", f"{duration:.1f}s")
                
                # Code preview
                st.markdown("**Code Preview:**")
                st.code(review.get('code', 'N/A'), language='python')
                
                # Summary
                summary = review.get('review', {}).get('summary', '')
                if summary:
                    st.markdown(f"**Summary**: {summary}")
                
                # View full review button
                if st.button(f"View Full Review #{idx + 1}", key=f"view_{idx}"):
                    st.session_state.current_review = review
                    st.rerun()
    else:
        st.info("No review history yet. Start reviewing code to build your history!")

with tab4:
    st.markdown("### ℹ️ About Code Intelligence Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Purpose
        An AI-powered code review assistant that learns from high-quality open-source projects 
        (FastAPI, Requests, Pydantic) to provide context-aware feedback.
        
        #### 🔧 How It Works
        
        **Hybrid Strategy**:
        1. **Fast Screening** (Ollama/Qwen2.5-Coder)
           - Quick analysis of code
           - Identifies severity level
           - Decides if detailed review needed
        
        2. **Detailed Review** (OpenRouter/Llama 3.1)
           - Deep analysis for critical issues
           - Detailed suggestions
           - References to best practices
        
        #### 📊 Categories Detected
        - 🐛 Bugs & Logic Errors
        - 🔒 Security Vulnerabilities
        - ⚡ Performance Issues
        - 📝 Documentation Gaps
        - 🎨 Style & Conventions
        - 🔧 Maintainability
        """)
    
    with col2:
        st.markdown("""
        #### ✨ Key Features
        - ✅ Hybrid Ollama + OpenRouter
        - ✅ Fast local screening (<5s)
        - ✅ Detailed cloud analysis (20-40s)
        - ✅ Context-aware suggestions
        - ✅ Multi-severity detection
        - ✅ Export results as JSON
        
        #### 🎯 Target Users
        - Bootcamp graduates
        - Indie developers
        - Small startups (2-5 person teams)
        - Developers learning Python best practices
        
        #### 📈 Test Results
        - **85.7% pass rate** on test suite
        - **12/14 tests passed**
        - Avg response: 32.9s
        - Strong detection across all categories
        
        #### 🔗 Links
        - [GitHub Repository](https://github.com/AishwaryaKandasami/code-intelligence-assistant)
        - [Test Results Report](reviews/TEST_RESULTS_REPORT.md)
        - Report Issues on GitHub
        """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Local Model**: Ollama with Qwen2.5-Coder 7B
        - Fast screening
        - Low resource usage
        - Offline capability
        
        **Cloud Model**: OpenRouter with Llama 3.1 8B
        - Detailed analysis
        - Better reasoning
        - Pay-per-use
        
        **Strategy Logic**:
        ```python
        if severity in ['CRITICAL', 'HIGH']:
            use_detailed_review()  # OpenRouter
        else:
            use_fast_review()  # Ollama only
        ```
        
        **Requirements**:
        - Python 3.10+
        - Ollama installed locally
        - OpenRouter API key (in .env)
        - ~4GB RAM for Ollama
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made by Aishwarya Kandasami<br>
        Learning from FastAPI, Requests & Pydantic<br>
        <em>Using Hybrid Ollama + OpenRouter System</em>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("💡 Tip: Use hybrid mode for best results. Critical issues automatically get detailed review!")