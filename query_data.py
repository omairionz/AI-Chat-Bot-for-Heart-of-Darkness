import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """

You are a literary assistant. Answer the question ONLY using the context below. 
Do NOT make up any information not found in the context. 


{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # CLI to input query
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load local Chroma DB with HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB for relevant chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=7)
    if len(results) == 0 or results[0][1] > 0.45:  # threashold
        print("Unable to find matching results.")
        return

    # Combine top results into context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt sent to model:\n", prompt)

    # Use HuggingFace local pipeline (CPU-friendly)
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",  # Replace with smaller model if needed
        max_new_tokens=256
       # do_sample=False,
       # temperature=0.1

    )

    # Generate answer
    response_text = generator(prompt, do_sample=True)[0]['generated_text']

    # Show response with sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response:\n{response_text}\n\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
