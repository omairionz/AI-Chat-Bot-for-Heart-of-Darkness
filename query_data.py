import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a literary analysis assistant with deep knowledge of Joseph Conrad's *Heart of Darkness*.
Use the following text excerpts from the novel to answer the question accurately.

If the answer is not explicitly stated, make a subtle inference — 
but avoid inventing details that contradict the text.

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # CLI input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text.strip()

    # Load Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Retrieve most relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=7)
    if len(results) == 0 or results[0][1] < 0.02:  # Threshold for good matches
        print("Unable to find matching results.")
        return

    # Build context from top results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt sent to model:\n", prompt)

    # Deterministic text generation (less hallucination)
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        max_new_tokens=256,
        do_sample=False,       # Ensures consistent output
        temperature=0.1         # Keeps tone factual and stable
    )

    # Generate answer
    response = generator(prompt)[0]['generated_text']

    # Display result with sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\nResponse:\n{response}\n\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
