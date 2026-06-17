# app.py — EU AI Act RAG (standalone local version)
import os, time, sys
import httpx
from bs4 import BeautifulSoup
import chromadb
from google import genai
from google.genai import types
from fasthtml.common import *
from monsterui.all import *
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.environ["GEMINI_KEY"]


if "--prep" in sys.argv:
    # --- 1. Fetch & parse EU AI Act ---
    r_api = httpx.get(
        "https://publications.europa.eu/resource/cellar/dc8116a1-3fe6-11ef-865a-01aa75ed71a1",
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "en", "User-Agent": "Mozilla/5.0"},
        follow_redirects=True
    )
    soup = BeautifulSoup(r_api.text, "lxml-xml")
    top_art = soup.find_all("div", id=lambda x: x and x.startswith("art_") and "." not in x)
    anx_divs = soup.find_all("div", id=lambda x: x and "anx" in x.lower())

    def parse_article(div):
        a = div.find_all('p', class_="oj-ti-art")[0].get_text(separator=" ", strip=True)
        t = div.find_all('div', class_="eli-title")[0].get_text(separator=" ", strip=True)
        b = " ".join(p.get_text(separator=" ", strip=True) for p in div.find_all('p', class_="oj-normal"))
        return {'article_number': int(a.split()[-1]), 'title': t.replace('`', ''), 'text': b.replace('\xa0', ' ')}

    def parse_annex(div):
        title = div.find('p', class_='oj-doc-ti').get_text(strip=True).replace('\xa0', ' ')
        text = div.get_text(separator=' ', strip=True).replace('\xa0', ' ')
        return {'annex_id': div['id'], 'title': title, 'text': text}

    chunks = [parse_article(a) for a in top_art]

    # --- 2. Embeddings & ChromaDB ---
    gemini = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="eu_ai_act")

    existing_ids = set(collection.get()['ids'])
    new_chunks = [c for c in chunks if f"art_{c['article_number']}" not in existing_ids]

    if new_chunks:
        new_content = [f"title: Article {c['article_number']} - {c['title']} | text: {c['text']}" for c in new_chunks]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in new_content]
        chunk_embs = []
        for i in range(0, len(new_chunks), 90):
            chunk_embs.append(gemini.models.embed_content(model="gemini-embedding-2", contents=contents[i:i+90]))
            if i + 90 < len(new_chunks): time.sleep(100)
        all_embeddings = [emb for resp in chunk_embs for emb in resp.embeddings]
        for chunk, emb in zip(new_chunks, all_embeddings):
            collection.add(
                ids=[f"art_{chunk['article_number']}"],
                embeddings=[emb.values],
                documents=[chunk['text']],
                metadatas=[{"Article Number": chunk['article_number'], "Title": chunk['title']}]
            )

    print(f"Collection has {collection.count()} articles")

    annex_chunks = [parse_annex(d) for d in anx_divs]

    existing_ids = set(collection.get()['ids'])
    new_annexes = [a for a in annex_chunks if a['annex_id'] not in existing_ids]

    if new_annexes:
        new_content = [f"title: {a['title']} | text: {a['text']}" for a in new_annexes]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in new_content]
        response = gemini.models.embed_content(model="gemini-embedding-2", contents=contents)
        for annex, emb in zip(new_annexes, response.embeddings):
            collection.add(
                ids=[annex['annex_id']],
                embeddings=[emb.values],
                documents=[annex['text']],
                metadatas=[{"Title": annex['title'], "Type": "Annex"}]
            )
        print(f"Added {len(new_annexes)} annexes")

    meta_text = """
    The EU AI Act is Regulation (EU) 2024/1689 of the European Parliament and of the Council, 
    adopted on 13 June 2024, published in the Official Journal of the EU on 12 July 2024.
    It entered into force on 1 August 2024.
    It becomes fully applicable on 2 August 2026, with the following phased dates:
    - 2 February 2025: Chapter I (General Provisions) and Chapter II (Prohibited AI Practices) apply.
    - 2 August 2025: Chapter III Section 4 (Notifying Authorities), Chapter V (General-Purpose AI Models), Chapter VII (Governance), Chapter XII (Penalties), and Article 78 apply.
    - 2 August 2026: Full application (most obligations including high-risk AI rules).
    - 2 August 2027: Article 6(1) and corresponding obligations for high-risk AI systems under Annex I apply.
    The Act contains 113 Articles, 13 Annexes, and 180 Recitals.
    It is structured into 13 Chapters: I General Provisions, II Prohibited AI Practices, 
    III High-Risk AI Systems, IV Transparency Obligations, V General-Purpose AI Models, 
    VI Innovation Support, VII Governance, VIII EU Database, IX Post-Market Monitoring, 
    X Codes of Conduct, XI Delegation of Power, XII Penalties, XIII Final Provisions.
    Geographic scope: applies across all EU Member States.
    Enforcement is overseen by the European AI Office and national competent authorities.
    It repeals no prior regulation but complements existing EU law including GDPR.
    Official text: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
    """.strip()

    meta_chunk = {"id": "meta_overview", "title": "EU AI Act — Document Overview & Metadata", "text": meta_text}

    if meta_chunk["id"] not in set(collection.get()['ids']):
        response = gemini.models.embed_content(model="gemini-embedding-2", contents=meta_text)
        collection.add(
            ids=[meta_chunk["id"]],
            embeddings=[response.embeddings[0].values],
            documents=[meta_chunk["text"]],
            metadatas=[{"Title": meta_chunk["title"], "Type": "Metadata"}]
        )
        print("Added metadata chunk")

    print(f"Collection now has {collection.count()} items")
else:
    # --- 3. RAG functions ---
    DISTANCE_THRESHOLD = 0.7
    gemini = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="eu_ai_act")

    def _make_label(chunk_id, meta):
        """Create a human-readable label from the chunk id and metadata."""
        if chunk_id.startswith("art_"):
            return f"Article {meta.get('Article Number', chunk_id)} — {meta['Title']}"
        if chunk_id.startswith("anx_"):
            return meta["Title"]
        return meta.get("Title", chunk_id)

    def retrieve(question, n_results=5):
        results = collection.query(
            query_embeddings=[gemini.models.embed_content(
                model="gemini-embedding-2",
                contents=f"task: search result | query: {question}"
            ).embeddings[0].values],
            n_results=n_results
        )
        return [
            {"id": cid, "label": _make_label(cid, m), "title": m["Title"], "text": d, "distance": dist}
            for cid, m, d, dist in zip(
                results["ids"][0], results["metadatas"][0],
                results["documents"][0], results["distances"][0]
            )
            if dist < DISTANCE_THRESHOLD
        ]

    def generate(question, chunks):
        if not chunks:
            return "I couldn't find relevant context in the EU AI Act to answer this question confidently."
        context = "\n\n".join(f"[{c['label']}]: {c['text']}" for c in chunks)
        response = gemini.models.generate_content(
            model="gemini-3.1-flash-lite",
            config={"system_instruction": (
                "You are a legal assistant for the EU AI Act. "
                "Answer ONLY based on the context provided below. "
                "Do NOT use knowledge from your training data. "
                "If a document or annex is referenced but not included in the context, say so — do not describe its contents. "
                "Always cite the article or annex number your answer is based on. "
                "If the context doesn't support a confident answer, say: "
                "'The provided context does not contain enough information to answer this question.'"
            )},
            contents=f"Question: {question}\n\nContext:\n{context}"
        )
        return response.text

    def ask(question): return generate(question, retrieve(question))

    # --- 4. FastHTML app ---
    app, rt = fast_app(
        hdrs=Theme.blue.headers(daisy=True) + [
            Style("""
                body { background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); min-height: 100vh; }
                .answer-card { white-space: pre-wrap; }
                .example-btn { cursor: pointer; }
                .htmx-indicator { display: none; }
                .htmx-request .htmx-indicator { display: inline-block; }
                .htmx-request.htmx-indicator { display: inline-block; }
            """),
            Script("""
                function fillQuestion(text) {
                    document.querySelector('textarea[name="question"]').value = text;
                }
            """),
        ],
    )

    EXAMPLES = [
        "What AI practices are prohibited under the EU AI Act?",
        "What are the obligations for high-risk AI systems?",
        "When does the EU AI Act become fully applicable?",
        "What are the penalties for non-compliance?",
    ]

    def chunk_card(c):
        return Card(
            P(Strong(f"📄 {c['label']}"), cls="mb-1"),
            P(f"Distance: {c['distance']:.3f}", cls="text-muted-foreground text-sm"),
            cls="border-l-4 border-blue-500",
        )

    def answer_ui(question):
        chunks = retrieve(question)
        answer = generate(question, chunks)
        source_cards = [chunk_card(c) for c in chunks] if chunks else [P("No relevant sources found.", cls="text-muted-foreground text-sm")]
        return Div(
            Card(
                H4("Answer", cls="mb-3"),
                Div(render_md(answer), cls="answer-card prose"),
                cls="mb-4",
            ),
            H4("Sources", cls="mb-3"),
            *source_cards,
        )

    @rt("/")
    def get():
        return Container(
            Div(
                H1("🇪🇺 EU AI Act Q&A", cls="text-center mt-6 mb-2"),
                P("Ask questions about the EU Artificial Intelligence Act (Regulation 2024/1689)",
                  cls="text-center text-muted-foreground text-lg"),
                cls="mb-6",
            ),
            Card(
                Form(
                    Div(
                        Textarea(placeholder="Ask a question about the EU AI Act...", name="question", rows=3,
                                 cls="mb-3", style="max-width: 900px; margin: 0 auto; display: block;"),
                        Div(
                            Button("Ask", type="submit", cls=ButtonT.primary, style="padding: 20px 80px; font-size: 1.1rem;"),
                            Span("⏳ Searching...", cls="htmx-indicator ml-3 text-muted-foreground text-sm"),
                            cls="flex items-center justify-center gap-3 mt-3",
                        ),
                        cls="text-center",
                    ),
                    hx_post="/ask", hx_target="#result", hx_swap="innerHTML",
                    hx_indicator="#spinner",
                ),
                cls="mb-4",
                style="max-width: 940px; margin: 0 auto;",
            ),
            Div(
                P("Try an example:", cls="text-muted-foreground text-sm mb-2"),
                DivLAligned(
                    *[Button(q, onclick=f"fillQuestion('{q}')", cls=("example-btn", ButtonT.secondary, "text-xs"))
                      for q in EXAMPLES],
                    cls="flex-wrap gap-2",
                ),
                cls="mb-6",
            ),
            Span(id="spinner", cls="htmx-indicator"),
            Div(id="result"),
            Footer(
                P("⚠️ Answers are AI-generated. Always verify against the ",
                  A("official text", href="https://eur-lex.europa.eu/eli/reg/2024/1689/oj", target="_blank"),
                  ".", cls="text-center text-muted-foreground text-sm"),
                cls="mt-8 mb-6 pt-4 border-t",
            ),
        )

    @rt("/ask")
    def post(question: str): return answer_ui(question)

    serve()

