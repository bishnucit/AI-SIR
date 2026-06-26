#in this example we are using a tool called research agent that calls duckduck go api and then send it to llm and then send response to users

#also we are using the same prompt without any tool to differentiate what is the difference in output when we use a tool and without a tool.

# this helps in those scenarios where the llm is not trained on current events a custom tool can help with output for such llms where the search returns 
#are fed to llm and then it formats and gives the output.



from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

@tool
def research_tool(query: str) -> str:
    """
    Web research tool that returns structured, concise notes.

    Use this for:
    - LangGraph / LangChain / AI frameworks
    - Documentation lookup
    - Technical comparisons

    Args:
        query (str): The topic to research

    Returns:
        str: Clean structured research notes
    """
  
    results = search.run(query)
    print(results)

    return f"""
TOPIC: {query}

SUMMARY OF FINDINGS:
{results}

TASK FOR MODEL:
- Extract only relevant facts
- Identify key concepts
- Ignore irrelevant noise
- Produce a structured summary
"""

llm = ChatOllama(
    model="llama3.1:8b",  # adjust to your installed model
    temperature=0,
)

agent = create_deep_agent(
    model=llm,
    tools=[], #research_tool
    system_prompt="You are a research assistant.",
)

result = agent.invoke({
    "messages": [
        ("user", "Research LangGraph and write a summary")
    ]
})




agent2 = create_deep_agent(
    model=llm,
    tools=[research_tool], 
    system_prompt="You are a research assistant.",
)

result2 = agent2.invoke({
    "messages": [
        ("user", "Research LangGraph and write a summary")
    ]
})

without_tool = (result["messages"][-1].content)


with_tool = (result2["messages"][-1].content)

print("=== Agent without tools ===")
print(without_tool)

print("\n=== Agent with research tool (duckduckgo api) ===")
print(with_tool)
