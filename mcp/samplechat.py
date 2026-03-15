# Requires ollama installed and model mistral:7b downloaded locally

import anthropic
import json

# Initialize client
client = anthropic.Anthropic(
    base_url="http://localhost:11434",
    api_key="ollama",
)

# Define the tool
tools = [
    {
        "name": "calculator",
        "description": "Performs simple math calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., 1+1"
                }
            },
            "required": ["expression"]
        }
    }
]

# User message
user_message = "What is 1 + 1?"

# Call the model
message = client.messages.create(
    model="mistral:7b",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": user_message}]
)

# Check if the model wants to use a tool
for block in message.content:
    if block.type == "tool_use":
        # Parse the expression
        expr = block.input.get("expression")
        try:
            result = eval(expr)  # Evaluate the math safely
        except Exception as e:
            result = f"Error: {e}"
        print(f"Tool: {block.name}")
        print(f"{expr} = {result}")
    else:
        # Sometimes the model just outputs text
        print("Model response:", block.text)
