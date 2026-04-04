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
    'stock_selection': 'TICKEROFSTOCK',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

result = financial_trading_crew.kickoff(inputs=inputs)

print(result)


"""

This code effectively simulates a financial trading team:

Analyzes stock trends
Designs trading strategies
Suggests trade execution
Assesses and mitigates risks

All locally with Ollama LLM + DuckDuckGo for live information.


Sample output:


Title: Risk Analysis Report for TICKER Stocks

Executive Summary:
In this report, we provide an in-depth analysis of the potential risks associated with investing in TICKER stocks and propose appropriate mitigation strategies based on our trading strategy. We follow a balanced approach that seeks opportunities for gains while managing associated risks effectively by leveraging machine learning models.

Methodology:
Our analysis includes the following steps:
1. Data Collection
2. Cleaning & Preparation of data
3. Feature Engineering
4. Model Training & Evaluation
5. Alert Generation & Insights
6. Real-time analysis and monitoring of TICKER stock market movements.

Potential Risks:
1. Market volatility: Stock prices can witness significant fluctuations due to various factors such as geopolitical events, economic indicators, and company-specific news that are challenging to predict accurately.
2. Overfitting & underfitting of machine learning models: During the training process for our models, it is essential to balance generalization ability with accuracy to avoid both overfitting (where the model performs exceptionally well on training data but poorly on unseen data) and underfitting (where the model fails to capture essential patterns in data). By utilizing techniques like cross-validation and hyperparameter tuning, we limit these potential risks.
3. Data quality: Reliable data is critical for accurate predictions, so it's crucial to gather information from trustworthy sources, ensure minimal missing or inconsistent data, and process data appropriately through cleaning and pre-processing steps.
4. False alerts & missed opportunities: Generating alerts based on predefined conditions might occasionally produce false positives or miss actual trading opportunities. This can be managed by setting reasonable threshold levels for technical indicators and continuously improving the models to minimize these risks.

Mitigation Strategies:
1. Monitoring stock market movements in real-time: Continuously track TICKER stocks and generate actionable alerts based on our machine learning models' predictions, keeping stakeholders informed about potential opportunities or risks.
2. Updating, cleaning, and preparing data: Regularly collect & clean the required data for accurate analysis results to maintain high data quality that better supports informed trading decisions.
3. Periodic feature engineering: Continuously evaluate feature sets on a monthly basis to introduce new ones when needed as it helps identify emerging trends or patterns in financial time series data.
4. Quarterly evaluation of machine learning models: Regularly re-evaluate the models to ensure we use the most optimal model for predicting stock trends and improving our ability to make accurate predictions.
5. Continuous improvement: Iteratively improve our trading strategy by refining models, setting reasonable threshold levels, and optimizing overall performance through data pre-processing techniques.

Conclusion:
With this comprehensive risk analysis report and the proposed mitigation strategies, we aim to establish a well-rounded approach for assessing risks associated with TICKER stocks and making informed trading decisions while managing associated risks effectively. By leveraging machine learning models, we can identify trends, generate actionable alerts, and share insights with stakeholders to help drive successful outcomes in the rapidly evolving stock market landscape of TICKER
.
"""
