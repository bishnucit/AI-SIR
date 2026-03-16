import asyncio

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():

    # Tell MCP how to launch the server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (reader, writer):

        session = ClientSession(reader, writer)

        await session.initialize()

        request = {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Explain black holes simply"
                    }
                }
            ]
        }

        response = await session.request("messages", request)

        print("\nLLM Response:\n")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())


#make sure -
# step 1 - ollama serve from cmd prompt
# step 2 - python client.py 

#this runs the server.py that calls the ollama sampler file that calls the llm then the response is shared back 
