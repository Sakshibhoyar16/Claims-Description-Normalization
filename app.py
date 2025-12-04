"""
Claims Description Normalizer - Streamlit UI
Professional interface for normalizing insurance claim descriptions.
"""

import streamlit as st
import pandas as pd
from claims_normalizer import ClaimsNormalizer, NormalizedClaim
import json
from datetime import datetime
import io
import time


# Page configuration
st.set_page_config(
    page_title="Claims Description Normalizer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- THEME / CSS ----------

BASE_CSS = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    </style>
"""

DARK_MODE_CSS = """
    <style>
    body, .main, .block-container {
        background-color: #0e1117 !important;
        color: #f5f5f5 !important;
    }
    .main-header {
        color: #61dafb !important;
    }
    .sub-header {
        color: #b3b3b3 !important;
    }
    .metric-card {
        background-color: #1c1f26 !important;
        border-left: 4px solid #61dafb !important;
    }
    .success-box {
        background-color: #1c261f !important;
        border-left: 4px solid #28a745 !important;
    }
    .warning-box {
        background-color: #262115 !important;
        border-left: 4px solid #ffc107 !important;
    }
    </style>
"""

# We’ll inject BASE_CSS now; dark CSS (if enabled) will be injected later
st.markdown(BASE_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_normalizer():
    """Load and cache the normalizer model"""
    return ClaimsNormalizer()


def display_result_card(result: NormalizedClaim):
    """Display normalized claim result in an organized card format"""
    
    # Loss Types
    st.markdown("### 🔍 Loss Types")
    if result.loss_types:
        cols = st.columns(len(result.loss_types))
        for i, loss_type in enumerate(result.loss_types):
            with cols[i]:
                st.info(f"**{loss_type}**")
    else:
        st.warning("No loss types detected")
    
    st.markdown("---")
    
    # Severity and Confidence
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Severity Level")
        severity_colors = {
            "Minor": "🟢",
            "Moderate": "🟡",
            "Major": "🟠",
            "Catastrophic": "🔴"
        }
        icon = severity_colors.get(result.severity, "⚪")
        st.markdown(
             f"""
            <div class='metric-card' style='color: black; font-weight: 600;'>
            <h2 style='color: black;'>{icon} {result.severity}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )  
    
    with col2:
        st.markdown("### 📊 Confidence Score")
        confidence_pct = result.confidence_score * 100 if result.confidence_score is not None else 0
        
        if confidence_pct >= 70:
            color = "#28a745"
        elif confidence_pct >= 50:
            color = "#ffc107"
        else:
            color = "#dc3545"
        
        st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: {color};'>{confidence_pct:.0f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Affected Assets
    st.markdown("### 🏠 Affected Assets")
    if result.affected_assets:
        asset_cols = st.columns(min(len(result.affected_assets), 4))
        for i, asset in enumerate(result.affected_assets):
            with asset_cols[i % 4]:
                st.success(f"✓ {asset}")
    else:
        st.info("No specific assets identified")
    
    st.markdown("---")
    
    # Extracted Entities
    st.markdown("### 📝 Extracted Entities")
    if result.extracted_entities:
        for entity_type, values in result.extracted_entities.items():
            with st.expander(f"{entity_type.replace('_', ' ').title()} ({len(values)})"):
                for value in values:
                    st.write(f"• {value}")
    else:
        st.info("No additional entities extracted")


def generate_explanation(result: NormalizedClaim) -> str:
    """Create a simple human-readable explanation for the prediction."""
    loss = ", ".join(result.loss_types) if result.loss_types else "an unspecified loss type"
    sev = result.severity or "unknown"
    conf = f"{result.confidence_score * 100:.0f}%" if result.confidence_score is not None else "N/A"
    
    explanation = (
        f"The claim has been classified as **{loss}** with a **{sev}** severity. "
        f"The model's confidence in this assessment is about **{conf}**. "
    )
    
    if result.affected_assets:
        explanation += (
            "The following assets appear to be affected: " +
            ", ".join(result.affected_assets) + ". "
        )
    else:
        explanation += "No specific assets could be clearly identified from the text. "
    
    explanation += (
        "This classification is based on keywords, entities, and patterns detected "
        "in the claim description."
    )
    return explanation


def create_downloadable_json(result: NormalizedClaim) -> str:
    """Create formatted JSON for download"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "loss_types": result.loss_types,
        "severity": result.severity,
        "affected_assets": result.affected_assets,
        "confidence_score": result.confidence_score,
        "extracted_entities": result.extracted_entities,
        "raw_text": result.raw_text
    }
    return json.dumps(data, indent=2)


def create_downloadable_csv(results: list) -> str:
    """Create CSV from batch results"""
    data = []
    for result in results:
        data.append({
            "Loss Types": ", ".join(result.loss_types),
            "Severity": result.severity,
            "Affected Assets": ", ".join(result.affected_assets),
            "Confidence Score": result.confidence_score,
            "Raw Text": result.raw_text
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def build_batch_dataframe(results: list) -> pd.DataFrame:
    """Utility to build a DataFrame from batch results for analytics."""
    rows = []
    for r in results:
        rows.append({
            "loss_types": ", ".join(r.loss_types) if r.loss_types else "Unknown",
            "severity": r.severity if r.severity else "Unknown",
            "confidence": r.confidence_score if r.confidence_score is not None else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    """Main application function"""
    
    # Load model early
    normalizer = load_normalizer()
    
    # Sidebar
    with st.sidebar:
        st.image("logo.jpg", use_container_width=True)
    
        st.markdown("## About")
        st.info("""
        This AI-powered system automatically extracts:
        
        ✓ Loss Type Classification  
        ✓ Severity Assessment  
        ✓ Affected Assets Detection  
        ✓ Key Entity Extraction
        
        **Features:**
        - Single & Batch Processing
        - JSON/CSV Export
        - Real-time Analysis
        """)
        
        # Dark mode toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=False)
        if dark_mode:
            st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
        
        st.markdown("## Processing Mode")
        mode = st.radio("Select Mode:", ["Single Claim", "Batch Processing"], label_visibility="collapsed")
    
    # Header
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image("claim.jpg", width=50)
    with col2:
        st.markdown(
            "<div class='main-header' style='margin-top: -10px;'>Claims Description Normalizer</div>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='sub-header'>Transform raw claim notes into structured, actionable data</div>", 
            unsafe_allow_html=True
        )
    
    # --------- SINGLE CLAIM MODE ----------
    if mode == "Single Claim":
        st.markdown("## 📝 Single Claim Processing")
        
        # Example claims
        with st.expander("💡 View Example Claims"):
            examples = [
                "Water leak in kitchen damaged the floor and cabinets. Estimate $3,500 in repairs needed.",
                "Car was rear-ended on highway. Severe damage to bumper and trunk. Vehicle totaled.",
                "House fire on 03/15/2024. Entire roof destroyed, smoke damage throughout.",
                "Minor scratch on door from shopping cart. Cosmetic damage only.",
                "Storm damage - tree fell on house, broke through roof into bedroom."
            ]
            for i, example in enumerate(examples, 1):
                st.code(f"{i}. {example}", language="text")
        
        # Input area (with session state)
        if "claim_text" not in st.session_state:
            st.session_state.claim_text = ""
        
        claim_text = st.text_area(
            "Enter Claim Description:",
            height=150,
            placeholder="Type or paste the raw claim description here...",
            help="Enter the free-text claim notes from adjusters or customers",
            key="claim_text",
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            process_btn = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.claim_text = ""
            st.rerun()
        
        if process_btn:
            if not claim_text.strip():
                st.error("⚠️ Please enter a claim description.")
            else:
                try:
                    with st.spinner("Analyzing claim..."):
                        result = normalizer.normalize(claim_text)
                except Exception as e:
                    st.error(f"❌ Error while analyzing the claim: {e}")
                    return
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Display results
                display_result_card(result)
                
                # Explanation
                st.markdown("### 💬 AI Explanation")
                st.markdown(generate_explanation(result))
                
                # Download section
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    json_data = create_downloadable_json(result)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"claim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Show JSON in expander
                    with st.expander("👁️ View JSON"):
                        st.json(json.loads(json_data))
    
    # --------- BATCH MODE ----------
    else:
        st.markdown("## 📦 Batch Processing")
        
        st.info("💡 Enter multiple claims separated by blank lines or upload a text file")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload Claims File (TXT)", 
            type=['txt'],
            help="Upload a text file with one claim per line or claims separated by blank lines"
        )
        
        # Text area for manual entry (session state)
        if "batch_text" not in st.session_state:
            st.session_state.batch_text = ""
        
        batch_text = st.text_area(
            "Or Enter Claims Manually:",
            height=200,
            placeholder="Claim 1: Water damage in basement...\n\nClaim 2: Vehicle collision on Main St...\n\nClaim 3: Fire damage to kitchen...",
            help="Separate claims with blank lines",
            key="batch_text",
        )
        
        process_batch_btn = st.button("🔍 Process Batch", type="primary")
        
        if process_batch_btn:
            claims = []
            
            # Process uploaded file
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    claims = [c.strip() for c in content.split('\n\n') if c.strip()]
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
                    return
            # Process manual entry
            elif batch_text.strip():
                claims = [c.strip() for c in batch_text.split('\n\n') if c.strip()]
            
            if not claims:
                st.error("⚠️ Please provide claims to process.")
            else:
                try:
                    with st.spinner(f"Processing {len(claims)} claims..."):
                        results = []
                        progress = st.progress(0)
                        total = len(claims)
                        
                        for i, c in enumerate(claims, 1):
                            # per-claim processing for real progress bar
                            res = normalizer.normalize(c)
                            results.append(res)
                            progress.progress(i / total)
                            # (optional) tiny delay so progress bar is visible
                            time.sleep(0.01)
                except Exception as e:
                    st.error(f"❌ Error while processing batch: {e}")
                    return
                
                st.success(f"✅ Successfully processed {len(results)} claims!")
                st.markdown("---")
                
                # Build DataFrame for analytics
                df_results = build_batch_dataframe(results)
                
                # Display summary statistics
                st.markdown("### 📊 Batch Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Claims", len(results))
                
                with col2:
                    avg_confidence = df_results["confidence"].mean() if not df_results.empty else 0
                    st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
                
                with col3:
                    loss_types = set()
                    for r in results:
                        loss_types.update(r.loss_types)
                    st.metric("Unique Loss Types", len(loss_types))
                
                with col4:
                    severities = df_results["severity"].tolist()
                    major_count = sum(1 for s in severities if s in ["Major", "Catastrophic"])
                    st.metric("Major+ Claims", major_count)
                
                # Simple analytics chart
                st.markdown("### 📈 Severity Distribution")
                if not df_results.empty:
                    severity_counts = df_results["severity"].value_counts().rename_axis("Severity").reset_index(name="Count")
                    severity_counts = severity_counts.set_index("Severity")
                    st.bar_chart(severity_counts)
                else:
                    st.info("Not enough data to display charts.")
                
                st.markdown("---")
                
                # Display individual results
                st.markdown("### 📋 Individual Results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Claim {i}: {result.raw_text[:60]}..."):
                        display_result_card(result)
                        st.markdown("#### 💬 Explanation")
                        st.markdown(generate_explanation(result))
                
                # Download options
                st.markdown("---")
                st.markdown("### 💾 Export Batch Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = create_downloadable_csv(results)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"batch_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_batch = json.dumps(
                        [json.loads(create_downloadable_json(r)) for r in results],
                        indent=2
                    )
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_batch,
                        file_name=f"batch_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>Claims Description Normalizer v1.1 | Powered by AI & NLP</p>
            <p style='font-size: 0.8rem;'>Developed for efficient claims processing and analysis</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
