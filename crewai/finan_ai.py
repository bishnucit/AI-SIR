#using local llm ollama with crewai and duckduckgo

# -------------------------------
# CrewAI Trading Agents - Ollama + DuckDuckGo
# -------------------------------

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from duckduckgo_search import DDGS

# -------------------------------
# LLM (Ollama)
# -------------------------------
# Use string reference for Ollama Llama 3
llm = "ollama/mistral:7b"
#llm = "ollama/qwen2.5-coder:latest"
#llm = "ollama/llama3:latest"

# -------------------------------
# Tool: DuckDuckGo Search
# -------------------------------
@tool
def search_tool(query: str) -> str:
    """Search the web for information and return top results."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["body"])
    return "\n".join(results)

# -------------------------------
# Agents
# -------------------------------
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="Expert in financial markets using statistical modeling and machine learning to provide actionable insights.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=True
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and refine trading strategies based on insights from the Data Analyst and user risk tolerance.",
    backstory="Specialist in quantitative analysis and strategy design for financial markets.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=True
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies based on approved trading strategies.",
    backstory="Expert in timing, pricing, and execution logistics for maximizing trade efficiency.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=True
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="Specialist in risk assessment and mitigation strategies in financial markets.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=True
)

# -------------------------------
# Tasks
# -------------------------------
data_analysis_task = Task(
    description="Analyze market data for {stock_selection} and predict trends.",
    expected_output="Provide market insights and alerts highlighting opportunities or risks.",
    agent=data_analyst_agent
)

strategy_development_task = Task(
    description="Develop trading strategies for {stock_selection} based on analysis and {risk_tolerance} risk tolerance.",
    expected_output="Provide detailed trading strategies aligned with user risk preferences.",
    agent=trading_strategy_agent
)

execution_planning_task = Task(
    description="Plan execution strategy for {stock_selection} based on approved trading strategies.",
    expected_output="Provide step-by-step execution plan with timing, pricing, and logistics.",
    agent=execution_agent
)

risk_assessment_task = Task(
    description="Assess risks for {stock_selection} and propose mitigation strategies.",
    expected_output="Provide risk analysis report with potential exposures and recommended safeguards.",
    agent=risk_management_agent
)

# -------------------------------
# Crew
# -------------------------------
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task
    ],
    process=Process.sequential,  # safer and more stable for local models
    manager_llm=llm,             # Ollama manages the crew
    verbose=True
)

# -------------------------------
# Run
# -------------------------------
inputs = {
    'stock_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

result = financial_trading_crew.kickoff(inputs=inputs)

print(result)
