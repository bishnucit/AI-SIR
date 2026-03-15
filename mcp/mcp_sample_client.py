#make sure mcp_sample.py is running via mcp dev mcp_sample.py

#then run below file - python mcp_sample_client.py

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_sample.py"], 
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "read_doc_contents",
                {"doc_id": "plan.md"}
            )

            print("Document text:", result.content[0].text)


asyncio.run(main())


# prints - Tools: ['read_doc_contents', 'edit_document']
