from mcp_sample import read_document, edit_document

# Read a doc
print(read_document("plan.md"))

# Edit a doc
edit_document(
    doc_id="plan.md",
    old_str="project's implementation",
    new_str="project implementation plan"
)

# Read again to verify
print(read_document("plan.md"))


#this is a sample python file that can be run to test the sample mcp functions read and edit doc
