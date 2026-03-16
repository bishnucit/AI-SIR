import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from ollama_sampler import sample_llama


server = Server("demo-mcp-server")


async def handle_messages(request):
    """
    Handle incoming message request
    """

    prompt = request["messages"][0]["content"]["text"]

    output = sample_llama(prompt)

    return {
        "content": [
            {
                "type": "text",
                "text": output
            }
        ]
    }


async def main():

    initialization_options = {}

    # Register handler
    server.request_handlers["messages"] = handle_messages

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, initialization_options)


if __name__ == "__main__":
    asyncio.run(main())
