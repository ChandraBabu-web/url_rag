import os
import re
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables from .env
load_dotenv()

# Function to clean text - Removes all characters except alphabets and spaces
def clean_text(text):
    return re.sub(r'[^A-Za-z\s]', '', text)  # Keeps only letters and spaces

# Function to reset the FAISS index directory
def reset_faiss_index(path="./faiss_index"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# URL processing function
def process_input(urls, question):
    # Initialize Ollama model
    model_local = Ollama(model="mistral")

    # Convert string of URLs into a list
    urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
    docs = []

    # Get FireCrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("FIRECRAWL_API_KEY environment variable not set")
        return "Missing API Key."

    print(f"URLs to scrape: {urls_list}")
    print(f"Using FireCrawl API Key: {api_key}")

    for url in urls_list:
        try:
            print(f"Scraping {url}...")
            loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {url}")

            # Clean text content
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)

            docs.extend(loaded_docs)
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")
            print(f"Error loading {url}: {e}")

    if not docs:
        print("No valid documents found.")
        return "No valid documents found."

    # Display document metadata
    print(f"Total documents loaded: {len(docs)}")
    for i, doc in enumerate(docs[:3]):  # Show a sample of 3 docs
        print(f"Doc {i+1} content (first 500 chars): {doc.page_content[:500]}")

    # Split the text into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"Total document chunks after splitting: {len(doc_splits)}")

    # Filter out complex metadata
    doc_splits = filter_complex_metadata(doc_splits)

    # Reset the FAISS index directory
    reset_faiss_index()

    # Initialize Hugging Face embeddings
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Convert text chunks into embeddings and store in the FAISS vector database
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=huggingface_embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})  # Adjust 'k' for more relevant results

    # Define the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.

    {context}

    Question: {question}

    Helpful Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize RetrievalQA chain
    retrievalQA = RetrievalQA.from_chain_type(
        llm=model_local,
        chain_type="stuff",
        retriever=retriever,
       # return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Query the RetrievalQA chain
    result = retrievalQA.invoke({"query": question})
    print(f"Response: {result}")

    return result

# Streamlit UI
st.title("Website Query with Ollama (RAG)")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Process input on button click
if st.button('Query URLs'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
