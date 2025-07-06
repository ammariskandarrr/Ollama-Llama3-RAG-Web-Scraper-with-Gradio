import gradio as gr
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configuration
DEFAULT_URL = "https://en.wikipedia.org/wiki/Lionel_Messi"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

def initialize_rag(url: str):
    """Initialize RAG pipeline with web content"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vectorstore.as_retriever()

def query_llm(question: str, context: str) -> str:
    """Query Ollama LLM with formatted prompt"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            'role': 'user',
            'content': f"Question: {question}\n\nContext: {context}"
        }]
    )
    return response['message']['content']

def rag_chain(question: str, retriever) -> str:
    """Execute full RAG pipeline"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return query_llm(question, context)

# Initialize components
retriever = initialize_rag(DEFAULT_URL)

# Gradio Interface
with gr.Blocks(title="Web RAG Assistant") as app:
    gr.Markdown("# 🦙 Web Content Q&A with Llama3")
    gr.Markdown("Ask questions about the webpage content")
    
    with gr.Row():
        question = gr.Textbox(
            label="Your Question",
            placeholder="What would you like to know?"
        )
        output = gr.Textbox(label="AI Answer")
    
    submit = gr.Button("Ask")
    submit.click(
        fn=lambda q: rag_chain(q, retriever),
        inputs=question,
        outputs=output
    )

if __name__ == "__main__":
    app.launch()
