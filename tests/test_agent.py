import pytest
from src.agent import create_agent

def test_qa_tool():
    agent = create_agent()
    result = agent.process_query(
        "What is Innovate Inc's market share?",
        explicit_tool="qa_tool"
    )
    assert "12%" in result["response"]

def test_autonomous_routing():
    agent = create_agent()
    result = agent.process_query(
        "Summarize the competitive landscape",
        use_autonomous_routing=True
    )
    assert result["tool_used"] == "summarize_tool"
