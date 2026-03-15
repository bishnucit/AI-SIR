# sample mcp server that will run 
# mcp dev filename.py
# this will open the mcp inspector
# u can also use py file to test it if inspector not work

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Create instance
mcp = FastMCP("SampleMCP")

# Define a tool
#@mcp.tool()
#def add(a: int, b: int) -> int:
#    return a + b



docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

# Write a tool to read a doc
@mcp.tool(
    name = "read_doc_contents",
    description = "Read the contents of document and return it as a string."
)
def read_document(
    doc_id: str = Field(description = "Id of the document to read")
):
    if doc_id not in docs:
        raise ValueError(f"doc {doc_id} not found")
    
    return docs[doc_id]

# TODO: Write a tool to edit a doc

@mcp.tool(
    name = "edit_document",
    description = "Edit a document by replacing a string in the documents content with a new string"
)
def edit_document(
    doc_id: str = Field(description = "Id of the document to edit"),
    old_str: str =  Field(description = "text to replace"),
    new_str: str =  Field(description = "text to replace with"),
):
    if doc_id not in docs:
        raise ValueError(f"doc {doc_id} not found")
    
    docs[doc_id] = docs[doc_id].replace(old_str, new_str)


# Run if executed directly
if __name__ == "__main__":
    mcp.run()
