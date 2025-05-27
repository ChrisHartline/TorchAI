import os
import pathlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
import gradio as gr
from langchain_core.retrievers import BaseRetriever
import re
import PyPDF2

# Load environment variables and constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Document Loader
class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    supported_files = {
        "pdf": PyPDFLoader,
        "txt": TextLoader,
    }

def load_documents(file_path: str) -> list[Document]:
    """Load documents from file path"""
    ext = pathlib.Path(file_path).suffix.lower().lstrip('.')
    loader_class = DocumentLoader.supported_files.get(ext)
    if not loader_class:
        raise DocumentLoaderException(f"Unsupported file type: {ext}. Please provide a .txt or .pdf file")
    
    loader = loader_class(file_path)
    docs = loader.load()
    return docs

# Embeddings and vector storage
def configure_retriever(docs: list[Document]) -> BaseRetriever:
    """Configure retriever for document search"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k":20})
    return retriever

# Chatbot
def configure_chatbot(retriever: BaseRetriever) -> Chain:
    """Configure the conversational chatbot"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=2, 
        streaming=True,
        max_tokens=15000
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=model, 
        retriever=retriever, 
        memory=memory, 
        verbose=True
    )

# Gradio app functions
def process_files(files):
    """Process uploaded files and create chatbot"""
    if not files:
        return None
    
    docs = []
    for file in files:
        if os.path.exists(file.name):
            docs.extend(load_documents(file.name))
    
    if not docs:
        raise DocumentLoaderException("No documents were successfully loaded")
    
    retriever = configure_retriever(docs)
    return configure_chatbot(retriever)

def respond(message, chat_history, qa_chain):
    """Handle chat responses"""
    if not qa_chain:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Please upload documents first."})
        return "", chat_history
    
    try:
        response = qa_chain.invoke({"question": message})
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response["answer"]})
        return "", chat_history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_message})
        return "", chat_history

def process_files_with_status(files):
    """Process files and return status"""
    if not files:
        return None, "Please upload at least one document."
    try:
        result = process_files(files)
        return result, "Documents processed successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove lines that are just numbers or very short
    text = '\n'.join(line for line in text.split('\n') 
                    if len(line.strip()) > 3 and not line.strip().isdigit())
    # Remove common metadata patterns
    text = re.sub(r'File size.*?MB', '', text)
    text = re.sub(r'Format:.*?Edition', '', text)
    text = re.sub(r'\d+\.\d+\s+out of \d+ stars', '', text)
    text = re.sub(r'\d+\s+ratings', '', text)
    # Remove "Read more" and similar phrases
    text = re.sub(r'Read more.*$', '', text)
    # Remove empty lines again
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def process_pdf(pdf_file):
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    # Clean the text immediately after extraction
                    cleaned_page = clean_text(page_text)
                    if cleaned_page:  # Only add non-empty pages
                        text += cleaned_page + "\n"
            except Exception as e:
                print(f"Warning: Error extracting text from page: {str(e)}")
                continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Split into chunks
        chunks = split_into_chunks(text)
        
        return chunks
    except Exception as e:
        print(f"Error in process_pdf: {str(e)}")
        raise

def split_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text (str): The text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        if start > 0:
            start = start - chunk_overlap
        
        if end >= text_length:
            chunks.append(text[start:])
            break
            
        if end < text_length:
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1:
                end = paragraph_break
            else:
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1:
                    end = sentence_break + 1
        
        chunks.append(text[start:end].strip())
        start = end
    
    return chunks

# Gradio Interface
with gr.Blocks(title="TorchAIassist") as demo:
    gr.Markdown("# TorchAIassist")
    gr.Markdown("A chatbot for your documents")
    
    with gr.Row():
        file_output = gr.File(
            label="Upload your documents",
            file_count="multiple",
            file_types=[".pdf", ".txt"]
        )
        status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(height=600, type="messages")
    msg = gr.Textbox(
        label="Ask a question about your documents",
        placeholder="Let me know what you want to know about your documents"
    )
    clear = gr.Button("Clear")
    
    qa_chain = gr.State(None)
    
    # Event handlers
    file_output.change(
        fn=process_files_with_status,
        inputs=[file_output],
        outputs=[qa_chain, status]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, qa_chain],
        outputs=[msg, chatbot]
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()