import streamlit as st
import requests
import json
import os

# Configure page
st.set_page_config(
    page_title="AI Market Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 AI Market Analyst Agent")
st.markdown("""
Welcome to the AI Market Analyst Agent! This intelligent system can:
- **Answer Questions** about market research documents
- **Generate Summaries** of key findings
- **Extract Structured Data** in JSON format
""")

# API endpoint configuration
API_BASE = os.getenv("API_BASE", "http://api:8000")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Tool selection
mode = st.sidebar.radio(
    "Select Mode",
    ["Autonomous Routing", "Q&A", "Summarize", "Extract"]
)

# Query input
st.header("Enter Your Query")
query = st.text_area(
    "Query",
    height=100,
    placeholder="E.g., What is Innovate Inc's market share?"
)

# Process button
if st.button("Process Query", type="primary"):
    if not query:
        st.error("Please enter a query")
    else:
        with st.spinner("Processing..."):
            try:
                # Prepare request
                if mode == "Autonomous Routing":
                    endpoint = f"{API_BASE}/api/query"
                    payload = {
                        "query": query,
                        "use_autonomous_routing": True
                    }
                elif mode == "Q&A":
                    endpoint = f"{API_BASE}/api/qa"
                    payload = {"query": query}
                elif mode == "Summarize":
                    endpoint = f"{API_BASE}/api/summarize"
                    payload = {"query": query}
                else:  # Extract
                    endpoint = f"{API_BASE}/api/extract"
                    payload = {"query": query}
                
                # Make request
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Display results
                st.success("Query processed successfully!")
                
                # Show metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Tool Used",
                        result.get("tool_used", "N/A")
                    )
                with col2:
                    st.metric(
                        "Processing Time",
                        f"{result.get('processing_time', 0):.3f}s"
                    )
                
                # Show response
                if result.get("response"):
                    st.header("Response")
                    st.write(result["response"])
                
                # Show extracted data
                if result.get("extracted_data"):
                    st.header("Extracted Data")
                    st.json(result["extracted_data"])
                
                # Show sources
                if result.get("sources"):
                    with st.expander("View Source Documents"):
                        for i, source in enumerate(result["sources"], 1):
                            st.subheader(f"Source {i}")
                            st.text(source["content"])
                            st.json(source["metadata"])
            
            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Example queries
st.sidebar.header("Example Queries")
st.sidebar.markdown("""
**Q&A:**
- What is Innovate Inc's market share?
- Who are the main competitors?
**Summarize:**
- Summarize the SWOT analysis
- Overview of market size and growth
**Extract:**
- Extract all company market shares
- List all strengths and weaknesses
""")
