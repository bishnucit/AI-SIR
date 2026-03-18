from archiv_search import search_arxiv, download_pdf
from agent import summarize_text

def main():
    print("📚 Local Research Agent (mistral:7b + arXiv)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Search topic: ")
        if query.lower() == "exit":
            break

        papers = search_arxiv(query)

        print(f"\nFound {len(papers)} papers. Summarizing...\n")

        for paper in papers:
            print(f"📄 {paper['title']}")
            print(f"🔗 {paper['entry_id']}")

            summary = summarize_text(paper['summary'])
            print(f"\n📝 Summary:\n{summary}\n{'-'*40}\n")


if __name__ == "__main__":
    main()
