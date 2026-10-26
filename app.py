
import os
import glob
import re
import gradio as gr
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCE_DIR = "source_material"
PDF_FILES = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))

def load_knowledge():
    chunks = []

    for path in PDF_FILES:
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            # Split extracted text into smaller, overlapping passages.
            words = text.split()
            for start in range(0, len(words), 160):
                part = " ".join(words[start:start + 200])

                if part.strip():
                    chunks.append({
                        "text": part,
                        "source": os.path.basename(path),
                        "page": page_num
                    })

    return chunks

CHUNKS = load_knowledge()

if not CHUNKS:
    raise RuntimeError(
        "No extractable PDF text found in source_material."
    )

VECTORIZER = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)
MATRIX = VECTORIZER.fit_transform(
    [chunk["text"] for chunk in CHUNKS]
)

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

def answer_question(question):
    if not question or not question.strip():
        return "Please enter a question.", ""

    query = VECTORIZER.transform([question.strip()])
    scores = cosine_similarity(query, MATRIX).ravel()
    best = scores.argsort()[::-1][:3]

    if len(best) == 0 or scores[best[0]] < 0.08:
        return (
            "I couldn't find sufficient evidence in the supplied "
            "lecture PDF. Try using the terminology from the lecture.",
            ""
        )

    selected = [
        CHUNKS[i] for i in best if scores[i] > 0
    ]

    # Rank individual sentences by their relevance to the question.
    candidates = []
    for chunk in selected:
        for sentence in split_sentences(chunk["text"]):
            sentence = sentence.strip()
            if len(sentence.split()) < 5:
                continue

            sentence_vector = VECTORIZER.transform([sentence])
            relevance = cosine_similarity(
                query, sentence_vector
            )[0][0]

            if relevance > 0:
                candidates.append({
                    "text": sentence,
                    "score": relevance,
                    "source": chunk["source"],
                    "page": chunk["page"]
                })

    # Keep the most relevant sentences in their original wording.
    candidates.sort(key=lambda item: item["score"], reverse=True)

    chosen = []
    seen = set()
    for item in candidates:
        normalized = item["text"].lower()
        if normalized not in seen:
            chosen.append(item)
            seen.add(normalized)
        if len(chosen) == 4:
            break

    if not chosen:
        return (
            "Relevant passages were found, but no clear answer "
            "could be extracted. Review the cited source passages below.",
            "\n".join(
                f"{c['source']} — page {c['page']}"
                for c in selected
            )
        )

    answer = "### Answer from the lecture\n\n"
    answer += "\n\n".join(
        f"- {item['text']}" for item in chosen
    )
    answer += (
        "\n\n*These are relevant sentences extracted from the "
        "source, not a newly generated explanation.*"
    )

    citations = "\n".join(dict.fromkeys(
        f"{item['source']} — page {item['page']}"
        for item in chosen
    ))

    return answer, citations

with gr.Blocks(title="MIT DSA RAG Tutor") as demo:
    gr.Markdown(
        "# MIT DSA RAG Tutor\n"
        "### Randomized Algorithms and Quick Sort\n"
        "Ask questions grounded in the supplied MIT lecture PDF. "
        "The system retrieves relevant sentences and provides page citations."
    )

    question = gr.Textbox(
        label="Ask a DSA question",
        placeholder="Why does pivot selection matter?",
        lines=2
    )

    submit = gr.Button("Ask the lecture")
    answer = gr.Markdown()
    sources = gr.Textbox(
        label="Source citations",
        lines=3
    )

    submit.click(
        answer_question,
        inputs=question,
        outputs=[answer, sources]
    )

    question.submit(
        answer_question,
        inputs=question,
        outputs=[answer, sources]
    )

if __name__ == "__main__":
   demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)