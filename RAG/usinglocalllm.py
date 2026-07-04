import fitz  # PyMuPDF
import ollama
import os


PDF_PATH = "documents/test1.pdf" #path to pdf
OUTPUT_PATH = "results/test1.md" #path to result


# -------------------------
# 1. Read PDF page by page
# -------------------------
def get_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i + 1, text))

    return pages


# -------------------------
# 2. Ask LLM for observations
# -------------------------
def analyze_page(page_num, text):
    prompt = f"""
You are analyzing a document page.

Task:
- Read the content carefully
- Explain it in simple point-wise observations
- Do NOT repeat raw text
- Focus on meaning, not formatting

Page {page_num} content:
{text}

Return:
- Bullet point observations only
"""

    response = ollama.chat(
        model="llama3.1:8b", # change this if you are using some different model
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# -------------------------
# 3. Save markdown report
# -------------------------
def save_report(observations):
    os.makedirs("Results", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# PDF Analysis Report\n\n")

        for page_num, summary in observations:
            f.write(f"## Page {page_num}\n\n")
            f.write(summary)
            f.write("\n\n---\n\n")


# -------------------------
# 4. Main pipeline
# -------------------------
def main():
    print("Reading PDF...")

    pages = get_pages(PDF_PATH)

    print(f"Total pages found: {len(pages)}")

    results = []

    for page_num, text in pages:
        print(f"Analyzing page {page_num}...")

        summary = analyze_page(page_num, text)
        results.append((page_num, summary))

    print("Saving report...")
    save_report(results)

    print(f"Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
