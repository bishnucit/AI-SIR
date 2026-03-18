import arxiv
import os

DOWNLOAD_DIR = "./papers"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

client = arxiv.Client()

def search_arxiv(query, max_results=3):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id
        })

    return results


def download_pdf(paper):
    file_path = os.path.join(DOWNLOAD_DIR, f"{paper['title']}.pdf")
    # Avoid overwriting
    file_path = file_path.replace("/", "_")
    if not os.path.exists(file_path):
        paper.download_pdf(dirpath=DOWNLOAD_DIR)
    return file_path
