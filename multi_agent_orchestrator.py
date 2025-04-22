import os
import json
import re
import uuid
import base64
from io import BytesIO
from typing import List, Dict, Annotated, Literal, TypedDict, Optional, Any, Union
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from operator import itemgetter

# For parsing uploaded diagrams and BRs
import docx2txt
import fitz  # PyMuPDF
import pandas as pd

import streamlit as st
import datetime
import hashlib

# New imports for URL handling and web scraping
import requests
from bs4 import BeautifulSoup, Comment
import validators
import urllib.parse

# Get API key from Streamlit secrets instead of environment variables
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize session state for uploaded documents and chat history if using Streamlit
def initialize_state():
    # Generate a unique ID for this session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize uploaded documents dictionary
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}
    
    # Initialize chat history list
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize all chat history for sidebar display
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = []
        
    # Initialize current chat ID
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    # Initialize id mappings dict
    if "id_mappings" not in st.session_state:
        st.session_state.id_mappings = {}
    
    # Initialize FAISS vector store for uploaded documents
    if "document_store" not in st.session_state:
        st.session_state.document_store = None
        
    # Initialize flag for new file upload
    if "new_file_uploaded" not in st.session_state:
        st.session_state.new_file_uploaded = False

# Function to clear current chat
def clear_chat():
    if st.session_state.chat_history:
        # Save current chat history to all chats before clearing
        chat_data = {
            "id": st.session_state.current_chat_id,
            "timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
            "messages": st.session_state.chat_history,
            "title": st.session_state.chat_history[0]["content"][:30] + "..." if st.session_state.chat_history else "Empty chat"
        }
        st.session_state.all_chats.append(chat_data)
        
    # Create a new chat ID
    st.session_state.current_chat_id = str(uuid.uuid4())
    # Clear current chat history
    st.session_state.chat_history = []

# Function to handle new file upload
def handle_file_upload():
    # Set the flag to indicate a new file was uploaded
    st.session_state.new_file_uploaded = True
    
    # Save current chat if it has content
    if st.session_state.chat_history:
        chat_data = {
            "id": st.session_state.current_chat_id,
            "timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
            "messages": st.session_state.chat_history,
            "title": st.session_state.chat_history[0]["content"][:30] + "..." if st.session_state.chat_history else "Empty chat"
        }
        st.session_state.all_chats.append(chat_data)
    
    # Create a new chat ID but don't clear existing chat immediately
    # It will be cleared when the user starts a new interaction
    st.session_state.current_chat_id = str(uuid.uuid4())

# Function to extract text from various file types
def extract_text_from_file(file):
    filename = file.name
    content = file.read()
    
    if filename.endswith('.pdf'):
        # Extract text from PDF
        with fitz.open(stream=content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    elif filename.endswith('.docx'):
        # Extract text from DOCX
        text = docx2txt.process(BytesIO(content))
        return text
    elif filename.endswith('.txt'):
        # Extract text from TXT
        return content.decode('utf-8')
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        # For images, attempt OCR to extract text
        try:
            # Use pytesseract for OCR
            import pytesseract
            from PIL import Image
            
            # Open the image using PIL
            image = Image.open(BytesIO(content))
            
            # Apply OCR using pytesseract
            extracted_text = pytesseract.image_to_string(image)
            
            # If text was extracted successfully, return it
            if extracted_text and len(extracted_text.strip()) > 0:
                return f"Text extracted from image using OCR:\n\n{extracted_text}"
            else:
                # Fallback if no text was extracted
                return f"Image file: {filename}\nNo text could be extracted through OCR. Please provide a clearer image or manually transcribe the content."
        except:
            # Fallback in case of error or if pytesseract is not installed
            return f"Image file: {filename}\nCould not perform OCR. Please ensure pytesseract is installed or provide a text version of the content."
    elif filename.endswith('.csv'):
        # Extract data from CSV
        df = pd.read_csv(BytesIO(content))
        return df.to_string()
    elif filename.endswith('.xlsx'):
        # Extract data from Excel
        df = pd.read_excel(BytesIO(content))
        return df.to_string()
    else:
        return f"Unsupported file type: {filename}"

# Function to parse BR IDs and content from uploaded documents
def parse_br_ids(text):
    """
    Parse business requirement IDs and system architecture IDs from the provided text.
    
    Args:
        text: The document text to parse
        
    Returns:
        dict: A dictionary mapping IDs to their content
    """
    # Debug the input
    print(f"Parsing BR IDs from text (first 100 chars): {text[:100]}...")
    
    br_dict = {}
    
    # First try to find BR IDs with standard format (BR-XXX)
    # This pattern is more flexible to catch various BR ID formats
    standard_br_pattern = r'(?:^|\s|\(|-)(?P<br_id>BR-\d+)(?:\)|:|\s)'
    standard_matches = re.findall(standard_br_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    print(f"Found {len(standard_matches)} standard BR ID matches: {standard_matches[:5]}")
    
    # Process standard BR-XXX references first
    for br_id in standard_matches:
        br_id = br_id.upper()  # Normalize to uppercase
        # Skip if we already processed this ID
        if br_id in br_dict:
            continue
            
        # Pattern to find content after this BR ID
        # Look for content until the next BR ID or end of text
        content_pattern = fr'(?:^|\s|\(|-)({re.escape(br_id)})(?:\)|:|\s)(.+?)(?=(?:^|\s|\(|-)BR-\d+(?:\)|:|\s)|$)'
        content_match = re.search(content_pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        if content_match:
            # Get the content and clean it up
            content = content_match.group(2).strip()
            br_dict[br_id] = content
    
    # If no IDs found with standard pattern, try looser patterns for BR IDs
    if not br_dict:
        # Look for "Business Requirement" headings followed by numbers
        heading_pattern = r'(?:Business Requirement|Business Rule|BR)[\s:-]+(\d+)'
        heading_matches = re.findall(heading_pattern, text, re.IGNORECASE)
        
        print(f"Found {len(heading_matches)} heading BR matches: {heading_matches[:5]}")
        
        for br_num in heading_matches:
            br_id = f"BR-{br_num}"
            # Pattern to find content after this heading
            heading_content_pattern = fr'(?:Business Requirement|Business Rule|BR)[\s:-]+{br_num}(.+?)(?=(?:Business Requirement|Business Rule|BR)[\s:-]+\d+|$)'
            heading_content_match = re.search(heading_content_pattern, text, re.IGNORECASE | re.DOTALL)
            
            if heading_content_match:
                # Get the content and clean it up
                content = heading_content_match.group(1).strip()
                br_dict[br_id] = content
    
    # If still no IDs found, try to look for bullet points or sections that might be requirements
    if not br_dict:
        # Look for numbered sections or bullet points that might be requirements
        bullet_pattern = r'(?:^|\n)(?:\d+\.|\*|\-)\s*(.+?)(?=(?:^|\n)(?:\d+\.|\*|\-)|$)'
        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)
        
        print(f"Found {len(bullet_matches)} bullet point matches")
        
        # Create artificial BR IDs for these sections
        for i, content in enumerate(bullet_matches, 1):
            br_id = f"BR-{i:03d}"  # Format as BR-001, BR-002, etc.
            br_dict[br_id] = content.strip()
    
    # If still no BRs and the document's type hint suggests it's a BR document, 
    # break it into sections using paragraphs
    if not br_dict and "business requirement" in text.lower()[:1000]:
        paragraphs = re.split(r'\n\s*\n', text)
        filtered_paragraphs = [p for p in paragraphs if len(p.strip()) > 50]  # Only paragraphs with substantial content
        
        print(f"Creating {len(filtered_paragraphs)} BR IDs from paragraphs")
        
        for i, para in enumerate(filtered_paragraphs, 1):
            if para.strip():
                br_id = f"BR-{i:03d}"  # Format as BR-001, BR-002, etc.
                br_dict[br_id] = para.strip()
    
    # Handle system architecture IDs
    sys_arch_pattern = r'(?:\(|\s|^)(SYS-ARCH-\d+)(?:\)|\s|:)'
    sys_matches = re.findall(sys_arch_pattern, text, re.DOTALL)
    
    print(f"Found {len(sys_matches)} system architecture ID matches: {sys_matches[:5]}")
    
    for sys_id in sys_matches:
        sys_id = sys_id.strip()
        # Find the content following this SYS-ARCH ID until the next ID or end of text
        content_pattern = f'{sys_id}(?:\\)|\\s|:)\\s*(.*?)(?=(?:\\(|\\s)(?:SYS-ARCH|BR)-\\d+(?:\\)|\\s|:)|\\Z)'
        content_matches = re.search(content_pattern, text, re.DOTALL)
        if content_matches:
            br_dict[sys_id] = content_matches.group(1).strip()
    
    print(f"Final extracted IDs: {list(br_dict.keys())}")
    return br_dict

class OrchestratorState(TypedDict):
    query: str
    message: Annotated[list, add_messages]
    documents: list
    decision: str
    reason: str
    response: str
    agent: str
    selected_agent: str
    agent_reasoning: str
    confidence_scores: Dict[str, float]
    query_classification: Dict[str, Any]
    agent_selection_history: List[Dict[str, Any]]
    uploaded_documents: Dict[str, Any]  # To store uploaded BR and architecture diagrams
    id_mappings: Dict[str, str]  # To store BR ID mappings
    chat_history: List[Dict[str, Any]]  # To store chat history

OrchestratorGraph = StateGraph(OrchestratorState)

# Try to initialize the LLM with the specified model, but fallback to a different model if needed
try:
    llm = ChatGroq(groq_api_key=groq_api_key,
                  model_name="llama-3.3-70b-versatile",
                  max_tokens=2500)  # Increased from 500 to 2500 to handle longer outputs
    print("Successfully initialized llama-3.3-70b-versatile model")
except Exception as e:
    print(f"Error initializing llama-3.3-70b-versatile: {e}")
    print("Falling back to mixtral-8x7b-32768 model")
    # Skip trying Claude-3 Opus since it requires an Anthropic API key
    # Try the fallback model
    try:
        llm = ChatGroq(groq_api_key=groq_api_key,
                      model_name="mixtral-8x7b-32768",
                      max_tokens=2500)
        print("Successfully initialized mixtral-8x7b-32768 model")
    except Exception as e:
        print(f"Error initializing mixtral-8x7b-32768 model: {e}")
        print("Trying one more fallback to llama3-8b-8192")
        # Try one more fallback
        llm = ChatGroq(groq_api_key=groq_api_key,
                      model_name="llama3-8b-8192",
                      max_tokens=2000)
        print("Successfully initialized llama3-8b-8192 model")


def vector_embedding():
 
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
      
        data_path = os.path.join(current_dir, "data")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists("faiss_index"):
            try:
                faiss_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                return faiss_db
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                # Continue to create a new one
                
        if os.path.exists("USPro/dataset.json"):
            with open("USPro/dataset.json", "r") as file:
                data = json.load(file)
            
            final_fiass_docs = []
            for req in data['functional_requirements']:
                func_req_desc = req['description']
                for story in req['user_stories']:
                    user_story = story['story']
                    final_fiass_docs.append({
                        'type': 'user_story',
                        'functional requirment': func_req_desc,
                        'content': user_story,
                    })
                for test in story['test_cases']:
                    test_desc = test['scenario']
                    final_fiass_docs.append({
                        'type': 'test_case',
                        'functional requirment': func_req_desc,
                        'content': test_desc,
                    })
            
            documents = [Document(page_content=item['content'], metadata={
                'type': item['type'],
                'functional requirment': item['functional requirment'],
                }) for item in final_fiass_docs]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            faiss_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
            print(f"Vector store created and saved to faiss_index")
            FAISS.save_local(faiss_db, "faiss_index")
            return faiss_db
        else:
            # Create a simple temporary vector store with the sample data
            # This allows the system to work without the dataset.json file
            print("USPro/dataset.json not found, creating a temporary vector store...")
            sample_data = [
                Document(page_content="User should be able to log in securely", metadata={'type': 'user_story', 'functional requirment': 'User Authentication'}),
                Document(page_content="System should validate user credentials", metadata={'type': 'user_story', 'functional requirment': 'Login Validation'}),
                Document(page_content="Test that valid credentials allow login", metadata={'type': 'test_case', 'functional requirment': 'Login Testing'})
            ]
            faiss_db = FAISS.from_documents(documents=sample_data, embedding=embeddings)
            return faiss_db
            
    except Exception as e:
        print(f"Error in vector_embedding: {e}")
        # Create an emergency fallback with minimal functionality
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            emergency_docs = [
                Document(page_content="Emergency fallback document for testing", metadata={'type': 'test', 'functional requirment': 'Emergency'})
            ]
            faiss_db = FAISS.from_documents(documents=emergency_docs, embedding=embeddings)
            return faiss_db
        except:
            # Last resort - return None and let the calling function handle it
            return None

def update_document_store(text, document_type, filename):
    """
    Update the FAISS document store with new text.
    
    Args:
        text: The text content to add
        document_type: Type of the document (e.g., "BR", "Architecture", "User Story")
        filename: Name of the uploaded file
    """
    try:
        # Parse IDs based on document type
        extracted_ids = {}
        
        # Skip BR ID parsing for web content
        if document_type.lower() == "web content":
            # For web content, just treat as a single document without BR parsing
            # Create a generic ID for this document
            doc_id = f"WEB-{str(uuid.uuid4())[:8]}"
            extracted_ids[doc_id] = text
            
            # Add to ID mappings
            st.session_state.id_mappings[doc_id] = {
                "type": document_type,
                "filename": filename,
                "content_snippet": text[:100] + "..." if len(text) > 100 else text,
                "full_content": text
            }
        elif document_type.lower() in ["br", "business requirement", "requirements"]:
            # For business requirements, use BR-XXX format
            # Only parse BR IDs for explicitly uploaded BR documents
            extracted_ids = parse_br_ids(text)
            
            # Update the ID mappings
            for br_id, content in extracted_ids.items():
                st.session_state.id_mappings[br_id] = {
                    "type": "Business Requirement",
                    "filename": filename,
                    "content_snippet": content[:100] + "..." if len(content) > 100 else content,
                    "full_content": content
                }
        elif document_type.lower() in ["system architecture", "architecture", "sys arch"]:
            # For system architecture, use SYS-ARCH-XXX format
            # First try to find existing SYS-ARCH IDs
            sys_arch_pattern = r'(?:\(|\s|^)(SYS-ARCH-\d+)(?:\)|\s|:)'
            sys_matches = re.findall(sys_arch_pattern, text, re.DOTALL)
            
            # If SYS-ARCH IDs found, extract them
            if sys_matches:
                for sys_id in sys_matches:
                    sys_id = sys_id.strip()
                    # Find the content following this SYS-ARCH ID until the next ID or end of text
                    content_pattern = f'{sys_id}(?:\\)|\\s|:)\\s*(.*?)(?=(?:\\(|\\s)(?:SYS-ARCH|BR)-\\d+(?:\\)|\\s|:)|\\Z)'
                    content_matches = re.search(content_pattern, text, re.DOTALL)
                    if content_matches:
                        extracted_ids[sys_id] = content_matches.group(1).strip()
            else:
                # If no SYS-ARCH IDs found, create sections by headers or paragraphs
                # Try to split by markdown headers
                header_pattern = r'(#+\s+.+?)\n'
                headers = re.findall(header_pattern, text)
                
                if headers:
                    # Split by headers
                    sections = re.split(header_pattern, text)
                    # First item is any text before the first header, skip if empty
                    sections = sections[1:] if sections[0].strip() == "" else sections
                    
                    # Pair headers with their content
                    for i in range(0, len(sections) - 1, 2):
                        header = sections[i].strip()
                        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
                        # Create a SYS-ARCH ID for this section
                        sys_id = f"SYS-ARCH-{i//2 + 1:03d}"
                        # Add header to the content
                        full_content = f"{header}\n{content}"
                        extracted_ids[sys_id] = full_content
                else:
                    # Fall back to paragraph-based splitting
                    paragraphs = re.split(r'\n\s*\n', text)
                    for i, para in enumerate(paragraphs):
                        if para.strip():
                            sys_id = f"SYS-ARCH-{i+1:03d}"
                            extracted_ids[sys_id] = para.strip()
            
            # Update ID mappings for system architecture
            for sys_id, content in extracted_ids.items():
                st.session_state.id_mappings[sys_id] = {
                    "type": "System Architecture",
                    "filename": filename,
                    "content_snippet": content[:100] + "..." if len(content) > 100 else content,
                    "full_content": content
                }
        else:
            # For other document types, just treat as a single document
            # Create a generic ID for this document
            doc_id = f"DOC-{str(uuid.uuid4())[:8]}"
            extracted_ids[doc_id] = text
            
            # Add to ID mappings
            st.session_state.id_mappings[doc_id] = {
                "type": document_type,
                "filename": filename,
                "content_snippet": text[:100] + "..." if len(text) > 100 else text,
                "full_content": text
            }
        
        # Create embedding for the document
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create document chunks
        if extracted_ids:
            # Create a document for each extracted ID
            documents = []
            for id_key, content in extracted_ids.items():
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "type": document_type,
                        "filename": filename,
                        "id": id_key
                    }
                ))
        else:
            # Otherwise, create a single document
            documents = [Document(
                page_content=text,
                metadata={
                    "type": document_type,
                    "filename": filename
                }
            )]
        
        # Split text into chunks with higher overlap for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Add document info to uploaded_documents tracking
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = {}
        
        # Create a unique document ID
        doc_unique_id = f"{document_type}-{str(uuid.uuid4())[:8]}"
        st.session_state.uploaded_documents[doc_unique_id] = {
            "type": document_type,
            "filename": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "extracted_ids": list(extracted_ids.keys()),
            "content_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
        # Check if we already have a document store
        if st.session_state.document_store:
            # Add to existing document store
            st.session_state.document_store.add_documents(chunks)
        else:
            # Create new document store
            st.session_state.document_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        return True, extracted_ids
    except Exception as e:
        print(f"Error updating document store: {e}")
        return False, {}

def get_vector_store():
    """
    Get the vector store, considering both the uploaded documents and the default documents.
    """
    try:
        # First check if we have uploaded documents
        if st.session_state.document_store:
            return st.session_state.document_store
        
        # If not, fall back to the default vector store
        faiss_db = vector_embedding()
        if faiss_db is None:
            print("WARNING: Vector store could not be created. Using mock data instead.")
            # Create a mock object with similarity_search method that returns basic documents
            class MockDB:
                def similarity_search(self, query):
                    return [Document(page_content=f"Mock document related to: {query}", metadata={'type': 'mock'})]
            return MockDB()
        return faiss_db
    except Exception as e:
        print(f"Error getting vector store: {e}")
        # Return a mock DB as fallback
        class MockDB:
            def similarity_search(self, query):
                return [Document(page_content=f"Mock document related to: {query}", metadata={'type': 'mock'})]
        return MockDB()

def format_docs(docs):
    try:
        if isinstance(docs, Document):
            return docs.page_content
            
        if isinstance(docs, list):
            try:
                val = "\n".join(doc["page_content"] for doc in docs)
            except:
                val = "\n".join(doc.page_content for doc in docs)
            return val
            
        # If it's a string
        if isinstance(docs, str):
            return docs
            
        print(f"Unexpected type: {type(docs)}")
        return str(docs)
            
    except Exception as e:
        print(f"Error in format_docs: {type(docs)} {docs[:3]}")
        raise

# Add this function after the format_docs function (around line 390)
def process_chat_history(chat_history_list):
    """
    Process the full chat history to extract relevant context while 
    managing token length for effective LLM input.
    
    Args:
        chat_history_list: List of chat messages with 'role' and 'content' keys
        
    Returns:
        str: Processed chat history string
    """
    if not chat_history_list:
        return ""
    
    # Initialize the result string
    processed_history = ""
    
    # Track BR IDs and other important entities mentioned in the conversation
    mentioned_br_ids = set()
    
    # Add all interactions to the history
    for entry in chat_history_list:
        if entry.get('role') == 'user':
            # Add user queries in full
            query = entry.get('content', '')
            processed_history += f"User: {query}\n"
            
            # Extract BR IDs from the query
            br_ids = re.findall(r'BR-\d+', query)
            mentioned_br_ids.update(br_ids)
            
        elif entry.get('role') == 'assistant':
            # Create an abbreviated version of the assistant's response
            response = entry.get('content', '')
            
            # Keep track of agent used
            agent = entry.get('agent', 'unknown')
            
            # Extract BR IDs from the response
            br_ids = re.findall(r'BR-\d+', response)
            mentioned_br_ids.update(br_ids)
            
            # Create abbreviated response
            if len(response) > 300:
                shortened = response[:300] + "..."
                processed_history += f"Assistant ({agent}): {shortened}\n"
            else:
                processed_history += f"Assistant ({agent}): {response}\n"
    
    # Add a summary of the BR IDs mentioned throughout the conversation
    if mentioned_br_ids:
        processed_history += f"\nImportant Business Requirements referenced in conversation: {', '.join(sorted(mentioned_br_ids))}\n"
    
    return processed_history

class QueryClassification(BaseModel):
    query_type: str = Field(description="The type of query (test case, user story, functional requirement, or other)")
    key_terms: List[str] = Field(description="Key terms identified in the query")
    intent: str = Field(description="The inferred intent of the user")
    complexity: str = Field(description="Estimated complexity (low, medium, high)")
    domain: str = Field(description="The domain or subject area of the query")


class AgentSelection(BaseModel):
    selected_agent: str = Field(description="The name of the selected agent")
    reasoning: str = Field(description="Detailed reasoning for why this agent was selected")
    confidence_scores: Dict[str, float] = Field(
        description="Confidence scores for each agent (0.0 to 1.0)",
        default_factory=dict
    )

# Query classifier - Analyzes the query to extract key information
def query_classifier(state: OrchestratorState) -> OrchestratorState:
    query = state['query']
    
    # Step 1: Process URLs properly if present
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w./%?=&#+]*)?'
    url_matches = re.findall(url_pattern, query)
    
    if url_matches:
        print(f"Detected URLs in query: {url_matches}")
        state['url_detected'] = True
        state['urls'] = url_matches
        
        # Process each URL to extract content
        url_contents = []
        for url in url_matches:
            try:
                # Make sure we're using the full URL including path
                full_url = url
                print(f"Processing complete URL: {full_url}")
                
                # Send request to the URL
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(full_url, headers=headers, timeout=15)
                
                # Check if request was successful
                if response.status_code != 200:
                    print(f"Failed to fetch URL: {full_url} (status code: {response.status_code})")
                    continue
                
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract the title
                title = soup.title.string if soup.title else "Untitled webpage"
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get the main content - more specific targeting for better content extraction
                main_content = None
                # Try to find common content containers
                for container in ['main', 'article', 'div[role="main"]', '#content', '.content', 'div.main', 'div.body']:
                    content = soup.select(container)
                    if content and len(content[0].get_text(strip=True)) > 200:  # Ensure there's meaningful content
                        main_content = content[0]
                        break
                
                # Fallback to body if no specific container found
                if not main_content:
                    main_content = soup.find('body')
                
                if main_content:
                    # Get text and clean it
                    text = main_content.get_text(separator='\n')
                    # Clean text: remove multiple newlines and whitespace
                    text = re.sub(r'\n+', '\n', text)
                    text = re.sub(r'\s+', ' ', text)
                    content = text.strip()
                else:
                    # Fallback to entire page content
                    text = soup.get_text(separator='\n')
                    text = re.sub(r'\n+', '\n', text)
                    text = re.sub(r'\s+', ' ', text)
                    content = text.strip()
                
                if content:
                    # Generate a document ID for the URL
                    url_doc_id = f"URL-{hashlib.md5(full_url.encode()).hexdigest()[:8]}"
                    
                    # Add to session state ID mappings
                    if hasattr(st, 'session_state'):
                        st.session_state.id_mappings[url_doc_id] = {
                            "type": "Web Content",
                            "filename": full_url,
                            "content_snippet": content[:100] + "..." if len(content) > 100 else content,
                            "full_content": content,
                            "title": title,
                            "url": full_url
                        }
                    
                    # Store the content for immediate use
                    url_contents.append({
                        "url": full_url,
                        "title": title,
                        "content": content,
                        "doc_id": url_doc_id
                    })
                    
                    print(f"Successfully extracted content from URL: {full_url} (title: {title})")
                    
                    # Update the document store with the URL content
                    update_document_store(
                        text=content,
                        document_type="Web Content",
                        filename=f"{title} ({full_url})"
                    )
            
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
        
        # Store extracted URL contents in state
        state['url_contents'] = url_contents
        
        if url_contents:
            # Flag that we have URL content to use as context
            state['use_url_context'] = True
            
            # Determine the correct agent based on the query content
            if 'test case' in query.lower() or 'testcase' in query.lower():
                state['selected_agent'] = 'testcase_agent'
                print("Pre-selecting testcase_agent based on URL content and query mentioning test cases")
            elif 'test data' in query.lower() or 'testdata' in query.lower():
                state['selected_agent'] = 'testdata_agent'
                print("Pre-selecting testdata_agent based on URL content and query mentioning test data")
            elif 'user stor' in query.lower():
                state['selected_agent'] = 'story_agent'
                print("Pre-selecting story_agent based on URL content and query mentioning user stories")
            elif 'requirement' in query.lower():
                state['selected_agent'] = 'functional_req_agent'
                print("Pre-selecting functional_req_agent based on URL content and query mentioning requirements")
            
            # Create a summary of the URLs for the query
            url_summary = ""
            for i, url_data in enumerate(url_contents):
                url_summary += f"URL {i+1}: {url_data['title']} ({url_data['url']})\n"
            
            print(f"URL content extracted and ready to use as context:\n{url_summary}")
        else:
            state['use_url_context'] = False
    else:
        state['url_detected'] = False
        state['use_url_context'] = False
    
    # Original query classification logic
    prompt = """
    You are a query classifier for a multi-agent software development assistant. Your task is to analyze the user query to determine its type, key terms, intent, complexity, and domain.
    
    For query_type, use one of these values:
    - "test case" if the query is about generating test scenarios or validation steps
    - "test data" if the query is about generating sample data for testing or validation
    - "user story" if the query is about creating user stories or capturing user requirements
    - "functional requirement" if the query is about defining system capabilities or technical specifications
    - "other" if the query doesn't fit into the categories above
    
    Be especially careful to distinguish between "test case" (which focuses on test steps/procedures) and "test data" (which focuses on sample values/inputs for testing).
    
    User Query: {query}
    
    YOUR RESPONSE MUST BE VALID JSON ONLY (no explanations, no markdown, just the JSON):
    
    {{
        "query_type": "test case|test data|user story|functional requirement|other",
        "key_terms": ["term1", "term2", ...],
        "intent": "brief description of user intent",
        "complexity": "low|medium|high",
        "domain": "domain area, e.g., software development, banking, etc.",
        "confidence": 0.0-1.0
    }}
    """
    
    # Add confidence field to the classification
    classification_template = ChatPromptTemplate.from_template(prompt)
    chain = classification_template | llm | StrOutputParser()
    
    try:
        raw_response = chain.invoke({"query": state['query']})
        
        # Clean up the response to ensure it's valid JSON
        # Remove any non-JSON content that might be present
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, raw_response)
        if match:
            json_str = match.group(1)
            try:
                classification = json.loads(json_str)
            except json.JSONDecodeError:
                # If still having trouble, try with a more aggressive cleanup
                cleaned_json = re.sub(r'```json|```|\n|\r', '', json_str)
                classification = json.loads(cleaned_json)
        else:
            raise ValueError(f"Could not extract JSON from response: {raw_response}")
        
        # For queries that specifically mention "test data", ensure query_type is "test data"
        lower_query = query.lower()
        if ("test data" in lower_query or "sample data" in lower_query or "test dataset" in lower_query) and "test case" not in lower_query:
            classification["query_type"] = "test data"
        
        # Add a default confidence score if not provided
        if "confidence" not in classification:
            classification["confidence"] = 0.8
            
        state['query_classification'] = classification
        
        # Step 2: Document relevance check - prioritize URL content if present
        if state.get('use_url_context', False):
            # When URLs are present, prioritize them over other documents
            state['document_relevance'] = {
                "relevant": True,
                "confidence": 0.95,
                "reasoning": f"Query includes URLs which have been processed for content. Using URL content as primary context.",
                "use_url_context": True
            }
            print("Document relevance: PRIORITIZING URL CONTENT over other documents")
        elif hasattr(st, 'session_state') and st.session_state.id_mappings:
            # If no URLs but we have other documents, check for BR IDs
            mentioned_br_ids = re.findall(r'BR-\d+', state['query'])
            
            # If BR IDs are mentioned, the query is definitely relevant to documents
            if mentioned_br_ids:
                state['document_relevance'] = {
                    "relevant": True,
                    "confidence": 0.95,
                    "reasoning": f"Query directly references {', '.join(mentioned_br_ids)} which are in the uploaded documents.",
                    "use_url_context": False
                }
                print(f"Document relevance: RELEVANT (confidence 0.95) - Query mentions {', '.join(mentioned_br_ids)}")
            else:
                # Without proper semantic checking, default to considering documents relevant
                # This is safer than potentially missing relevant document context
                state['document_relevance'] = {
                    "relevant": True,
                    "confidence": 0.6,
                    "reasoning": "No direct document references found, but defaulting to using documents.",
                    "use_url_context": False
                }
                print("Document relevance: RELEVANT (confidence 0.6) - No explicit references but using documents by default")
        else:
            state['document_relevance'] = {
                "relevant": False,
                "confidence": 1.0,
                "reasoning": "No documents are uploaded, so the query cannot be relevant to any documents.",
                "use_url_context": False
            }
            print("Document relevance: NOT RELEVANT (confidence 1.0) - No documents uploaded")
            
    except Exception as e:
        print(f"Error in query classification: {e}")
        state['query_classification'] = {
            "query_type": "unknown",
            "key_terms": [],
            "intent": "unknown",
            "complexity": "unknown",
            "domain": "unknown",
            "confidence": 0.0
        }
        state['document_relevance'] = {
            "relevant": True,  # Default to True to be safe
            "confidence": 0.5,
            "reasoning": f"Error during classification: {str(e)}. Defaulting to using documents.",
            "use_url_context": False
        }
    
    return state

# Agent Selector - This is the reasoning component that decides which agent to use
def agent_selector(state: OrchestratorState) -> OrchestratorState:

    if 'agent_selection_history' not in state.keys():
        state['agent_selection_history'] = []
    
    if 'agent_reasoning' not in state.keys():
        state['agent_reasoning'] = "Default reasoning: No specific agent reasoning provided."
    
    prompt = """
    You are an intelligent workflow orchestrator that decides which specialized agent should handle a user query.
    You have four specialized agents available:
    
    1. Test Case Generator: Specializes in creating structured test cases for software features. Good for queries about testing methodologies, validation scenarios, or quality assurance plans.
    
    2. Story Generator: Specializes in creating user stories that capture user needs and requirements. Good for queries about user experiences, scenarios, or feature descriptions.
    
    3. Functional Requirement Generator: Specializes in creating formal functional requirements. Good for queries about system capabilities, technical specifications, or formal documentation.
    
    4. Test Data Generator: Specializes in creating realistic test datasets to validate test cases. Good for queries about test inputs, sample data, edge cases, or data validation scenarios.
    
    Based on the user query and its classification, you need to:
    1. Analyze what the user is asking for
    2. Determine which agent would be best suited to handle this query
    3. Provide your reasoning for this selection
    4. Assign confidence scores to each agent (0.0 to 1.0)
    
    User Query: {query}
    
    Query Classification:
    {classification}
    
    YOUR RESPONSE MUST BE VALID JSON ONLY in the following format (no explanations, no markdown, just the JSON):
    
    {{
        "selected_agent": "name of the selected agent",
        "reasoning": "your detailed reasoning for selecting this agent",
        "confidence_scores": {{
            "test_case_generator": 0.0 to 1.0,
            "story_generator": 0.0 to 1.0,
            "functional_requirement_generator": 0.0 to 1.0,
            "test_data_generator": 0.0 to 1.0
        }}
    }}
    """
    
    query = state['query']
    
    # Get query classification if not already in state
    if 'query_classification' not in state.keys():
        try:
            classification_result = query_classifier(state)
            classification = classification_result['query_classification']
            state['query_classification'] = classification
        except Exception as e:
            print(f"Error in query classification: {e}")
            classification = {
                "query_type": "unknown",
                "key_terms": [],
                "intent": "unknown",
                "complexity": "unknown",
                "domain": "unknown"
            }
            state['query_classification'] = classification
    else:
        classification = state['query_classification']
    
    try:
        classification_json = json.dumps(classification, indent=2)
        
        # Create prompt template
        agent_selection_template = ChatPromptTemplate.from_template(prompt)
        
        # Create the chain
        agent_selection_chain = (
            {"query": itemgetter("query"), "classification": itemgetter("classification")}
            | agent_selection_template
            | llm
            | StrOutputParser()
        )
        
        # Execute the chain
        result = agent_selection_chain.invoke({
            "query": query,
            "classification": classification_json
        })
        
        # Parse the JSON response
        parsed_response = json.loads(result)
        
        # Validate the response
        selected_node = parsed_response["selected_agent"]
        reasoning = parsed_response["reasoning"]
        confidence_scores = parsed_response["confidence_scores"]
        
        # Map the agent name to node name
        agent_mapping = {
            "test case generator": "testcase_agent",
            "story generator": "story_agent",
            "functional requirement generator": "functional_req_agent",
            "test data generator": "testdata_agent"
        }
        
        # Convert to the node name if a known agent name is used
        for agent_name, node_name in agent_mapping.items():
            if selected_node.lower() == agent_name.lower():
                selected_node = node_name
                break
                
        # Update state with agent selection
        state['selected_agent'] = selected_node
        state['agent_reasoning'] = reasoning
        state['confidence_scores'] = confidence_scores
        
        # Add to selection history
        state['agent_selection_history'].append({
            "query": query,
            "classification": classification,
            "selected_agent": selected_node,
            "reasoning": reasoning,
            "confidence_scores": confidence_scores
        })
        
        print(f"Query: {query}")
        print(f"Selected Agent: {selected_node}")
        print(f"Reasoning: {reasoning}")
        print(f"Confidence Scores: {confidence_scores}")
        
    except Exception as e:
        print(f"Error in agent selection: {e}")
        # Fallback to a default agent if parsing fails
        state['selected_agent'] = "story_agent"
        state['agent_reasoning'] = f"Defaulting to story agent due to error: {str(e)}"
        state['confidence_scores'] = {
            "test_case_generator": 0.0,
            "story_generator": 1.0,
            "functional_requirement_generator": 0.0,
            "test_data_generator": 0.0
        }
        
        # Add to selection history
        state['agent_selection_history'].append({
            "query": query,
            "classification": classification,
            "selected_agent": "story_agent",
            "reasoning": f"Defaulting to story agent due to error: {str(e)}",
            "confidence_scores": {
                "test_case_generator": 0.0,
                "story_generator": 1.0,
                "functional_requirement_generator": 0.0,
                "test_data_generator": 0.0
            },
            "error": str(e)
        })
    
    return state

# Confidence threshold checker - Determines if confidence is high enough or if we need more analysis
def check_confidence(state: OrchestratorState) -> Literal["proceed", "analyze_further", "out_of_scope"]:
    """
    Check confidence scores to determine how to proceed.
    
    Returns:
        - "proceed" if we have high confidence in the agent selection
        - "analyze_further" if additional analysis is needed
        - "out_of_scope" if the query is likely not within the system's domain
    """
    # Get confidence scores and document relevance
    confidence_scores = state.get('confidence_scores', {})
    classification_confidence = state.get('query_classification', {}).get('confidence', 0.0)
    document_relevance = state.get('document_relevance', {})
    
    # If the query explicitly mentions BR IDs or is clearly relevant to uploaded documents,
    # never treat it as out of scope
    mentioned_br_ids = re.findall(r'BR-\d+', state.get('query', ''))
    is_doc_relevant = document_relevance.get('relevant', False) and document_relevance.get('confidence', 0) > 0.7
    
    # If specific BR IDs are mentioned or documents are clearly relevant, the query is in scope
    if mentioned_br_ids or is_doc_relevant:
        print("Query is considered in-scope due to document relevance or BR ID mentions")
        # Check regular confidence levels to decide between proceed and analyze_further
        if state.get('selected_agent') in ('testcase_agent', 'functional_req_agent', 'story_agent', 'testdata_agent', 'combined_agent'):
            selected_agent_confidence = confidence_scores.get(state['selected_agent'], 0.0)
            if selected_agent_confidence > 0.5:
                return "proceed"
            else:
                return "analyze_further"
        else:
            return "proceed"  # Default to proceeding
    
    # Only check for out-of-scope if the query isn't clearly relevant to documents
    # Check for very low confidence scores across all agents
    if classification_confidence < 0.4:
        print(f"Query might be out-of-scope: classification confidence is low ({classification_confidence})")
        return "out_of_scope"
        
    # If the highest agent confidence is very low, it might be out of scope
    if confidence_scores:
        highest_confidence = max(confidence_scores.values())
        if highest_confidence < 0.3:  # Very low confidence threshold
            print(f"Query might be out-of-scope: highest agent confidence is low ({highest_confidence})")
            return "out_of_scope"
    
    # Regular confidence check logic for in-scope queries
    if state.get('selected_agent') in ('testcase_agent', 'functional_req_agent', 'story_agent', 'testdata_agent', 'combined_agent'):
        # Get the confidence score for the selected agent
        selected_agent_confidence = confidence_scores.get(state['selected_agent'], 0.0)
        
        # If confidence is high enough, proceed
        if selected_agent_confidence > 0.5:
            return "proceed"
        # Otherwise, analyze further
        else:
            return "analyze_further"
    else:
        # Default to proceeding if no selected agent, or unknown agent
        return "proceed"

# Further analysis node - Performs deeper analysis when confidence is low
def analyze_further(state: OrchestratorState) -> OrchestratorState:
    prompt = """
    You are an expert query analyzer for a software development assistant system.
    The initial agent selection process resulted in low confidence scores.
    
    User Query: {query}
    
    Initial Classification:
    {classification}
    
    Initial Agent Selection:
    Selected Agent: {selected_agent}
    Reasoning: {reasoning}
    Confidence Scores: {confidence_scores}
    
    Please perform a deeper analysis of this query to determine the most appropriate agent.
    Consider the following:
    1. Are there any ambiguities in the query that could be causing low confidence?
    2. Are there multiple valid interpretations of what the user is asking for?
    3. Is there a clear primary intent that should take precedence?
    
    Based on your analysis, provide a revised agent selection in the following JSON format:
    {{
        "selected_agent": "name of the selected agent",
        "reasoning": "your detailed reasoning for selecting this agent",
        "confidence_scores": {{
            "test_case_generator": 0.0 to 1.0,
            "story_generator": 0.0 to 1.0,
            "functional_requirement_generator": 0.0 to 1.0,
            "test_data_generator": 0.0 to 1.0
        }}
    }}
    
    Your response should be valid JSON only.
    """
    
    query = state['query']
    classification = state.get('query_classification', {})
    selected_agent = state['selected_agent']
    reasoning = state['agent_reasoning']
    confidence_scores = state['confidence_scores']
    
    analyzer_template = ChatPromptTemplate.from_template(prompt)
    
    # Use JsonOutputParser to ensure structured output
    parser = JsonOutputParser(pydantic_object=AgentSelection)
    
    llm_chain = (
        {
            "query": itemgetter("query"),
            "classification": itemgetter("query_classification"),
            "selected_agent": itemgetter("selected_agent"),
            "reasoning": itemgetter("agent_reasoning"),
            "confidence_scores": itemgetter("confidence_scores")
        } 
        | analyzer_template 
        | llm 
        | parser
    )
    
    try:
        result = llm_chain.invoke({
            "query": query,
            "query_classification": classification,
            "selected_agent": selected_agent,
            "agent_reasoning": reasoning,
            "confidence_scores": confidence_scores
        })
        
        # Map the selected agent to the actual node name
        agent_mapping = {
            "test case generator": "testcase_agent",
            "story generator": "story_agent",
            "functional requirement generator": "functional_req_agent",
            "test data generator": "testdata_agent"
        }
        
        # Normalize the selected agent name for mapping
        selected_agent_lower = result.selected_agent.lower()
        for key in agent_mapping:
            if key in selected_agent_lower:
                selected_node = agent_mapping[key]
                break
        else:
            # Default to story_agent if no match
            selected_node = "story_agent"
        
        # Update state with revised selection results
        state['selected_agent'] = selected_node
        state['agent_reasoning'] = f"[After further analysis] {result.reasoning}"
        state['confidence_scores'] = result.confidence_scores
        
        # Add to selection history
        state['agent_selection_history'].append({
            "query": query,
            "classification": classification,
            "selected_agent": selected_node,
            "reasoning": f"[After further analysis] {result.reasoning}",
            "confidence_scores": result.confidence_scores,
            "analysis_type": "deeper_analysis"
        })
        
        # Log the revised decision
        print(f"Revised Agent Selection: {selected_node}")
        print(f"Revised Reasoning: {result.reasoning}")
        print(f"Revised Confidence Scores: {result.confidence_scores}")
        
    except Exception as e:
        print(f"Error in further analysis: {e}")
        # Keep the original selection if analysis fails
        # No need to update state as we're keeping the original selection
        
        # Add to selection history
        state['agent_selection_history'].append({
            "query": query,
            "classification": classification,
            "selected_agent": selected_agent,
            "reasoning": reasoning,
            "confidence_scores": confidence_scores,
            "analysis_type": "deeper_analysis_failed",
            "error": str(e)
        })
    
    return state


def Story_Agent(state: OrchestratorState) -> OrchestratorState:
    # First, generate the functional requirements that would be created by the functional_req_agent
    # This ensures consistency between functional requirements and user stories
    fr_state = state.copy()
    
    # Extract the BR ID from the query to create a consistent FR generation request
    mentioned_br_ids = re.findall(r'BR-\d+', state['query'])
    functional_reqs = ""
    
    # Check if we should prioritize URL content over BR content
    if state.get('use_url_context', False) and 'url_contents' in state and state['url_contents']:
        # If using URL content, don't automatically generate functional requirements from BRs
        # instead, we'll use the URL content directly
        print("Using URL content as context for user story generation instead of BR-based functional requirements")
        functional_reqs = "Based on the URL content provided in the context below instead of formal functional requirements."
    elif mentioned_br_ids:
        # Create a FR generation query based on the same BR
        br_id = mentioned_br_ids[0]
        fr_query = f"Generate functional requirements for {br_id}"
        fr_state['query'] = fr_query
        
        # Check if we have cached functional requirements for this BR
        fr_cache_key = f"fr_cache_{fr_query.strip().lower()}"
        
        # If we don't have cached functional requirements, generate them using functional_req_agent
        if not hasattr(st, 'session_state') or fr_cache_key not in st.session_state:
            # Call functional_req_agent to generate consistent functional requirements
            fr_output = functional_req_agent(fr_state)
            if 'response' in fr_output:
                # Extract just the functional requirements content without the query analysis
                fr_content = fr_output['response']
                if "Output ID:" in fr_content:
                    fr_content = fr_content.split("Output ID:")[1].split("\n\n", 1)[1]
                
                # Cache these functional requirements for future use
                if hasattr(st, 'session_state'):
                    st.session_state[fr_cache_key] = fr_content
                
                functional_reqs = fr_content
        else:
            # Use cached functional requirements
            functional_reqs = st.session_state[fr_cache_key]
            print(f"Using cached functional requirements for BR: {br_id}")
    
    prompt = """
    You are an expert user story writer following industry-standard Agile methodologies. Your task is to analyze the provided business requirements and functional requirements to create professionally formatted user stories.
    
    FORMAT YOUR USER STORIES USING THIS EXACT STRUCTURE:
    
    ## User Stories
    
    ### US-001: [Short Title]
    **As a** [type of user],
    **I want** [an action or feature],
    **So that** [benefit/value/reasoning]
    
    **Acceptance Criteria:**
    1. [Criterion 1]
    2. [Criterion 2]
    3. [Criterion 3]
    
    **Priority:** [High/Medium/Low]
    **Story Points:** [Fibonacci number: 1, 2, 3, 5, 8, 13]
    **Related BR IDs:** [List of related BR IDs like BR-001, BR-002]
    **Related FR IDs:** [List of related FR IDs like FR-001, FR-002]
    
    ### US-002: [Short Title]
    [Follow same structure...]
    
    INSTRUCTIONS:
    1. Create well-structured user stories with unique IDs (US-001, US-002, etc.)
    2. Include clear acceptance criteria for each story
    3. Assign appropriate priority and story point estimates
    4. Ensure stories align with the business requirements and functional requirements
    5. Make stories testable and specific
    6. Break down complex requirements into multiple stories if needed
    7. Include all functional and non-functional requirements
    8. Reference the BR IDs when creating stories to maintain traceability
    9. Reference the FR IDs that each user story implements
    10. If the query mentions updates to previously discussed requirements or BR IDs, make sure to maintain consistency with previous context
    11. Follow the exact syntax and structure from previous outputs
    12. EXTREMELY IMPORTANT: ONLY generate user stories for the specific BR ID mentioned in the query (e.g., if BR-004 is mentioned, only generate user stories for BR-004)
    13. Always maintain consistency in output - your response should be exactly the same each time for the same BR ID
    14. CRITICAL: Only use information from the provided business requirements documents and functional requirements. Do not add any user stories that are not based on the provided documents.
    15. If no relevant business requirements are found in the provided documents, state that clearly instead of inventing user stories.

    Here are the functional requirements to base your user stories on:
    {functional_reqs}
    
    Here is the user query:
    {query}

    Here is the context from uploaded business requirements:
    {context}
    
    Here is the reasoning for why you were selected to handle this query:
    {reasoning}
    
    Query Classification:
    {classification}
    
    Business Requirement IDs:
    {br_ids}
    
    Previous Chat History (for context):
    {chat_history}
    """
    query = state['query']
    
    # Ensure agent_reasoning is in the state
    if 'agent_reasoning' not in state:
        state['agent_reasoning'] = "Default reasoning: No specific agent reasoning provided."
    
    reasoning = state.get('agent_reasoning', "No specific reasoning provided")
    classification = state.get('query_classification', {})
    
    # Get BR IDs if available in session state
    br_ids = ""
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
        br_ids = "The following Business Requirement IDs are available:\n"
        for br_id, details in st.session_state.id_mappings.items():
            if br_id.startswith("BR-"):
                br_ids += f"- {br_id}: {details['content_snippet']}\n"
    
    # Emphasize that only supplied BRs should be used
    if not br_ids or br_ids == "The following Business Requirement IDs are available:\n":
        br_ids = "No business requirements have been uploaded. Please only generate user stories if explicitly mentioned in the query, and clearly state that no BR documents were available."
    
    # Get FR IDs if available in session state
    fr_ids = ""
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
        for id_key, details in st.session_state.id_mappings.items():
            if details.get('type') == 'Functional Requirement':
                fr_ids += f"Functional Requirements: {details['content_snippet']}\n\n"
    
    # Get chat history for context retention
    chat_history = ""
    if hasattr(st, 'session_state') and 'chat_history' in st.session_state:
        chat_history = process_chat_history(st.session_state.chat_history)
    
    # Check if using URL content instead of standard document retrieval
    if state.get('use_url_context', False) and 'url_contents' in state and state['url_contents']:
        # Format the URL content as the context instead of retrieved documents
        url_context = "URL CONTENT:\n\n"
        for i, url_data in enumerate(state['url_contents']):
            url_context += f"--- URL {i+1}: {url_data['title']} ({url_data['url']}) ---\n\n"
            # Include substantial content but avoid making it too large
            max_content_length = 5000  # Limit content length
            content = url_data['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "...[content truncated for size]"
            url_context += f"{content}\n\n"
        
        formatted_docs = url_context
        print(f"Using URL content as context for user story generation")
        
        # Add information to the prompt about URL-based context
        prompt += "\n\nIMPORTANT: Your primary context is the content from the URLs provided in the query. Generate user stories based on this content rather than pre-existing business requirements."
    else:
        # Regular document retrieval for standard cases
        # Check if document retrieval should be skipped based on relevance check
        should_retrieve_docs = True
        formatted_docs = ""
        
        if 'document_relevance' in state:
            # If query is not relevant to documents with high confidence, skip retrieval
            if not state['document_relevance']['relevant'] and state['document_relevance']['confidence'] > 0.7:
                should_retrieve_docs = False
                formatted_docs = "No document context used as query appears unrelated to uploaded documents."
                print("Skipping document retrieval as query is not relevant to documents")
        
        # Only retrieve documents if needed
        if should_retrieve_docs:
            try:
                faiss_db = get_vector_store()
                docs = faiss_db.similarity_search(query)
                formatted_docs = format_docs(docs)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                formatted_docs = f"No context available due to error: {str(e)}"
    
    state['documents'] = formatted_docs
    
    # Extract specific BR ID mentioned in the query to ensure we only generate for this BR
    mentioned_br_ids = re.findall(r'BR-\d+', query)
    
    # Only apply BR-specific instructions if we're not using URL content
    if not state.get('use_url_context', False):
        # If we have mentioned BR IDs, add special instruction to only generate for these BRs
        if mentioned_br_ids:
            br_instruction = f"\n\nIMPORTANT: ONLY generate user stories related to {', '.join(mentioned_br_ids)}. Do not include user stories for any other BR IDs."
            prompt += br_instruction
            
            # Check if the mentioned BR exists in the uploaded documents
            br_exists = False
            if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
                for br_id in mentioned_br_ids:
                    if br_id in st.session_state.id_mappings:
                        br_exists = True
                        break
            
            if not br_exists:
                br_instruction += f"\n\nWARNING: The requested business requirement(s) {', '.join(mentioned_br_ids)} were not found in the uploaded documents. Only generate user stories if you can find relevant content in the provided context."
                prompt += br_instruction
    
    # If we don't have functional requirements, inform the model to generate user stories but mention the issue
    if not functional_reqs:
        functional_reqs = "No specific functional requirements available. Please generate appropriate user stories based on the provided context."
    
    userstory_template = ChatPromptTemplate.from_template(prompt)
    
    # Set a fixed temperature and random seed for consistency
    llm_with_seed = llm.with_config(configurable={"temperature": 0.0, "seed": 42})
    
    llm_chain = (
        {"query": itemgetter("query"), 
         "context": itemgetter("documents"),
         "reasoning": itemgetter("agent_reasoning"),
         "classification": itemgetter("query_classification"),
         "br_ids": lambda _: br_ids if 'br_ids' in locals() or 'br_ids' in globals() else mentioned_br_ids if mentioned_br_ids else [],
         "chat_history": lambda _: chat_history,
         "functional_reqs": lambda _: functional_reqs if 'functional_reqs' in locals() or 'functional_reqs' in globals() else []} 
        | userstory_template 
        | llm_with_seed 
        | StrOutputParser()
    )
    
    try:
        # Check if there are existing user stories for this BR ID
        output_id = None
        user_stories = None
        
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and mentioned_br_ids:
            # Look for existing user stories for this BR
            for id_key, details in st.session_state.id_mappings.items():
                if (details.get('type') == 'User Story' and 
                    any(br_id in details.get('referenced_br_ids', []) for br_id in mentioned_br_ids) and
                    'full_content' in details):
                    print(f"Found existing user stories for {', '.join(mentioned_br_ids)}, reusing")
                    user_stories = details.get('full_content', '')
                    output_id = id_key
                    break
        
        # If not found in existing mappings, check cache or generate new
        if not user_stories:
            # Check if this exact query has been processed before and is in session state
            cache_key = f"us_cache_{query.strip().lower()}"
            if hasattr(st, 'session_state') and cache_key in st.session_state:
                user_stories = st.session_state[cache_key]
                print(f"Using cached user stories for query: {query[:50]}...")
            else:
                user_stories = llm_chain.invoke({
                    "query": query, 
                    "documents": formatted_docs,
                    "agent_reasoning": reasoning,
                    "query_classification": classification,
                    "functional_reqs": functional_reqs
                })
                
                # Cache the result for future use
                if hasattr(st, 'session_state'):
                    st.session_state[cache_key] = user_stories
            
            # Generate unique IDs for this output
            output_id = f"STORIES-{str(uuid.uuid4())[:8]}"
        
        # Extract user story IDs from the output
        user_story_ids = re.findall(r'US-\d+', user_stories)
        
        # Extract BR IDs referenced in output
        br_ids_in_output = re.findall(r'BR-\d+', user_stories)
        
        # Extract FR IDs referenced in output
        fr_ids_in_output = re.findall(r'FR-\d+', user_stories)
        
        # Create a mapping entry for this output if it's new
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and output_id not in st.session_state.id_mappings:
            st.session_state.id_mappings[output_id] = {
                "type": "User Story",
                "query": query,
                "user_story_ids": user_story_ids,
                "referenced_br_ids": br_ids_in_output,
                "referenced_fr_ids": fr_ids_in_output,
                "content_snippet": user_stories[:100] + "..." if len(user_stories) > 100 else user_stories,
                "full_content": user_stories
            }
        
        # Add query classification and agent details to output
        query_analysis = f"""
## Query Analysis
| Category | Details |
|----------|---------|
| Query Type | {classification.get('query_type', 'Not specified')} |
| Intent | {classification.get('intent', 'Not specified')} |
| Complexity | {classification.get('complexity', 'Not specified')} |
| Key Terms | {', '.join(classification.get('key_terms', ['None']))} |
| Domain | {classification.get('domain', 'Not specified')} |

## Agent Selection
| Agent | Confidence Score |
|-------|-----------------|
| Test Case Generator | {state['confidence_scores'].get('test_case_generator', 0):.2f} |
| User Story Generator | {state['confidence_scores'].get('story_generator', 0):.2f} |
| Functional Requirement Generator | {state['confidence_scores'].get('functional_requirement_generator', 0):.2f} |
| Test Data Generator | {state['confidence_scores'].get('test_data_generator', 0):.2f} |

**Selected Agent**: User Story Generator  
**Reasoning**: {reasoning}

## User Story Output
Output ID: {output_id}

"""
        
        # Final response with analysis and generated stories
        state['response'] = query_analysis + user_stories
    except Exception as e:
        print(f"Error in story agent: {e}")
        state['response'] = f"Could not generate user stories due to an error: {str(e)}"
    
    state['agent'] = "story_agent"
    return state

# Test Case Agent (reused from testcase_bot.py)
def testcase_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Test Case Generator Agent - Creates test scenarios and validation steps.
    
    This agent specializes in generating detailed test cases for validating software functionality.
    """
    prompt = """
    You are an expert Test Case Engineer specializing in creating comprehensive test cases for software applications. 
    Your task is to analyze the provided requirements and generate detailed, executable test cases
    that validate the functionality described in the requirements.
    
    FORMAT YOUR TEST CASES USING THIS EXACT STRUCTURE:
    
    ## Test Cases for [Feature/Requirement]
    
    ### TC-001: [Short Test Case Title]
    **Description:** Brief description of what this test case verifies
    **Priority:** [High/Medium/Low]
    **Prerequisites:** 
    - List any prerequisites needed before executing this test
    
    **Test Steps:**
    1. [First step]
    2. [Second step]
    3. [Third step]
    ...
    
    **Expected Results:**
    1. [Expected result for step 1]
    2. [Expected result for step 2]
    3. [Expected result for step 3]
    ...
    
    **Related Requirements:** [List of related requirement IDs or references]
    
    ### TC-002: [Short Test Case Title]
    [Follow same structure...]
    
    INSTRUCTIONS:
    1. Create well-structured test cases with unique IDs (TC-001, TC-002, etc.)
    2. Include detailed, step-by-step instructions that anyone could follow
    3. Each step must have a clear, verifiable expected result
    4. Cover both positive and negative test scenarios (e.g., valid and invalid inputs)
    5. Include edge cases and boundary testing where appropriate
    6. Test all functional aspects described in the requirements
    7. Ensure test cases are traceable back to business requirements
    8. If the user mentions a specific BR ID, focus ONLY on test cases for that BR
    9. If the user mentions specific functionality, focus test cases on that functionality
    10. For each Business Requirement, create a MINIMUM of 3 test cases
    11. If the requirements involve a user interface, include UI validation tests
    12. If the requirements involve APIs, include API testing scenarios
    13. If the requirements involve data flows, include data validation tests
    14. For healthcare IT systems, follow HIPAA compliance requirements and healthcare interoperability standards
    15. For HL7 FHIR APIs, include specific tests for Bundle submissions, resource validation, and conformance to profiles
    16. IMPORTANT: Only use information from the provided requirements, do not invent functionality not described
    17. If no relevant information is found in the provided documents, state that clearly instead of inventing test cases
    
    User Query: {query}
    
    Context from Requirements:
    {context}
    
    Reasoning for why this agent was selected:
    {reasoning}
    
    Query Classification:
    {classification}
    
    Business Requirement IDs:
    {br_ids}
    
    Previous Chat History (for context):
    {chat_history}
    """
    
    query = state['query']
    
    # Enhance prompt for specialized domains based on URL content or query
    if 'healthcare' in query.lower() or 'health' in query.lower() or 'medical' in query.lower() or 'hipaa' in query.lower() or 'fhir' in query.lower() or 'hl7' in query.lower():
        prompt += """
        
        HEALTHCARE IT SPECIFIC INSTRUCTIONS:
        1. Follow HIPAA compliance requirements in all test cases
        2. Include test cases for patient data security and privacy
        3. Add test cases for audit logging of PHI access
        4. For HL7 FHIR APIs, test for:
           - Valid resource formats according to FHIR specifications
           - Proper handling of FHIR bundles
           - Correct implementation of FHIR search parameters
           - Appropriate HTTP status codes for FHIR operations
           - OAuth2 authentication where applicable
        5. Include interoperability test cases with other healthcare systems
        6. Test for compliance with specific implementation guides mentioned in the requirements
        7. Verify that clinical terminologies (SNOMED CT, LOINC, RxNorm) are properly implemented
        8. Test cases should validate proper consent management
        9. Include scenarios for handling different patient matching algorithms
        10. Test both REST and SOAP interfaces if mentioned in requirements
        """
    
    # Check URL content for specialized domain detection
    if state.get('use_url_context', False) and 'url_contents' in state:
        url_content_text = ""
        for url_data in state['url_contents']:
            url_content_text += url_data.get('content', '')
            
        # Healthcare domain detection from URL content
        healthcare_terms = ['FHIR', 'HL7', 'healthcare', 'patient', 'clinical', 'medical', 'HIPAA', 
                           'provider', 'encounter', 'diagnosis', 'procedure', 'EHR', 'EMR']
        
        if any(term.lower() in url_content_text.lower() for term in healthcare_terms):
            prompt += """
            
            HEALTHCARE IT SPECIFIC INSTRUCTIONS (detected from URL content):
            1. Follow HIPAA compliance requirements in all test cases
            2. Include test cases for patient data security and privacy
            3. Specifically test FHIR resource validation against implementation guides
            4. Test for proper handling of patient identifiers and demographics
            5. Include test cases for FHIR Bundle submissions and responses
            6. Test FHIR Search operations with various parameters
            7. Verify proper implementation of SMART on FHIR if applicable
            8. Ensure authentication and authorization tests are included
            9. Test RESTful operations (GET, POST, PUT, DELETE) against FHIR endpoints
            10. Include negative test cases for invalid resources and error handling
            11. Verify correct implementation of FHIR profiles and extensions
            """
    
    # Ensure agent_reasoning is in the state
    if 'agent_reasoning' not in state:
        state['agent_reasoning'] = "Default reasoning: No specific agent reasoning provided."
    
    reasoning = state.get('agent_reasoning', "No specific reasoning provided")
    classification = state.get('query_classification', {})
    
    # Get BR IDs if available in session state
    br_ids = ""
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
        br_ids = "The following Business Requirement IDs are available:\n"
        for br_id, details in st.session_state.id_mappings.items():
            if br_id.startswith("BR-"):
                br_ids += f"- {br_id}: {details['content_snippet']}\n"
    
    # Emphasize that only supplied BRs should be used
    if not br_ids or br_ids == "The following Business Requirement IDs are available:\n":
        br_ids = "No business requirements have been uploaded. Please only generate test cases if explicitly mentioned in the query, and clearly state that no BR documents were available."
    
    # Get chat history for context retention
    chat_history = ""
    if hasattr(st, 'session_state') and 'chat_history' in st.session_state:
        chat_history = process_chat_history(st.session_state.chat_history)
    
    # Initialize user_stories to an empty dict to prevent access errors
    user_stories = {}
    
    # Check if using URL content instead of standard document retrieval
    if state.get('use_url_context', False) and 'url_contents' in state and state['url_contents']:
        # Format the URL content as the context instead of retrieved documents
        url_context = "URL CONTENT:\n\n"
        for i, url_data in enumerate(state['url_contents']):
            url_context += f"--- URL {i+1}: {url_data['title']} ({url_data['url']}) ---\n\n"
            # Include full content for URLs
            content = url_data['content']
            url_context += f"{content}\n\n"
        
        formatted_docs = url_context
        print(f"Using URL content as context for test case generation")
        
        # Add information to the prompt about URL-based context
        prompt += "\n\nIMPORTANT: Your primary context is the content from the URLs provided in the query. Generate test cases based directly on this content rather than pre-existing business requirements. Focus on the functionality, APIs, or interfaces described in the URL content."
        
        # Add additional prompt for ensuring tabular output format for URL content
        prompt += "\n\nCRITICAL: Always maintain IT standard tabular output format even when working with URL content. Use markdown tables when presenting structured data. Ensure sections are clearly delineated with proper headings and formatting."
    else:
        # Check if document retrieval should be skipped based on relevance check
        should_retrieve_docs = True
        formatted_docs = ""
        
        if 'document_relevance' in state:
            # If query is not relevant to documents with high confidence, skip retrieval
            if not state['document_relevance']['relevant'] and state['document_relevance']['confidence'] > 0.7:
                should_retrieve_docs = False
                formatted_docs = "No document context used as query appears unrelated to uploaded documents."
                print("Skipping document retrieval as query is not relevant to documents")
        
        # Only retrieve documents if needed
        if should_retrieve_docs:
            try:
                faiss_db = get_vector_store()
                docs = faiss_db.similarity_search(query)
                formatted_docs = format_docs(docs)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                formatted_docs = f"No context available due to error: {str(e)}"
    
    # Store the documents in the state
    state['documents'] = formatted_docs
    
    # Extract specific BR ID mentioned in the query to ensure we only generate for this BR
    mentioned_br_ids = re.findall(r'BR-\d+', query)
    
    # Only apply BR-specific instructions if we're not using URL content
    if not state.get('use_url_context', False):
        # If we have mentioned BR IDs, add special instruction to only generate for these BRs
        if mentioned_br_ids:
            br_instruction = f"\n\nIMPORTANT: ONLY generate test cases related to {', '.join(mentioned_br_ids)}. Do not include test cases for any other BR IDs."
            prompt += br_instruction
            
        # Check if we already have user stories for this BR to help with test case generation
        story_cache_key = f"user_stories_for_{'_'.join(mentioned_br_ids)}" if mentioned_br_ids else None
        
        if hasattr(st, 'session_state') and story_cache_key and story_cache_key in st.session_state:
            user_stories = st.session_state[story_cache_key]
            print(f"Found existing user stories for {', '.join(mentioned_br_ids)}, reusing")
            
            # Add user stories to the prompt for better context
            prompt += "\n\nRELATED USER STORIES:\n"
            for us_id, us_content in user_stories.items():
                prompt += f"\n{us_id}:\n{us_content}\n"
    
    # Let's build the prompt with the available context
    prompt_template = ChatPromptTemplate.from_template(prompt)
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        # Process the generation
        testcases = chain.invoke({
            "query": query,
            "context": formatted_docs,
            "reasoning": reasoning,
            "classification": json.dumps(classification, indent=2),
            "br_ids": br_ids,
            "chat_history": chat_history
        })
        
        # Store the response
        state['response'] = testcases
        state['agent'] = "testcase_agent"
    except Exception as e:
        print(f"Error in test case agent: {e}")
        state['response'] = f"Could not generate test cases due to an error: {str(e)}"
        state['agent'] = "testcase_agent"
    
    return state

# Functional Requirement Agent (new agent)
def functional_req_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Functional Requirement Generator Agent - Creates functional requirements from business requirements.
    
    This agent analyzes business requirements and generates clear, structured functional requirements.
    """
    prompt = """
    You are an expert Functional Requirements Analyst specializing in breaking down business requirements into 
    detailed functional requirements. Your task is to analyze the provided business requirements and generate
    precise, well-structured functional requirements.

    FORMAT YOUR FUNCTIONAL REQUIREMENTS RESPONSE USING THIS EXACT STRUCTURE:
    
    ## Functional Requirements for [Business Requirement ID/Title]
    
    ### Functional Requirement 1: [Short Title for FR-1]
    **ID**: FR-[BR-ID]-001
    **Description**: [Clear, concise description of what the system should do]
    **Dependencies**: [Any dependencies on other requirements or systems, if applicable]
    **Acceptance Criteria**:
    1. [Specific, measurable criteria 1]
    2. [Specific, measurable criteria 2]
    3. [Specific, measurable criteria 3]
    
    ### Functional Requirement 2: [Short Title for FR-2]
    **ID**: FR-[BR-ID]-002
    **Description**: [Clear, concise description of what the system should do]
    **Dependencies**: [Any dependencies on other requirements or systems, if applicable]
    **Acceptance Criteria**:
    1. [Specific, measurable criteria 1]
    2. [Specific, measurable criteria 2]
    
    [Continue with more functional requirements as needed]
    
    INSTRUCTIONS:
    1. Each business requirement should be broken down into 3-7 functional requirements
    2. Focus on WHAT the system should do, not HOW it should be implemented
    3. Ensure each functional requirement is specific, measurable, achievable, relevant, and time-bound
    4. Use consistent IDs that reference the original business requirement ID
    5. Include clear acceptance criteria for each requirement
    6. If a business requirement includes multiple capabilities, create separate functional requirements for each
    7. Ensure requirements are atomic - each should specify exactly one capability
    8. Use simple, clear language that can be understood by non-technical stakeholders
    9. If the user mentions specific business requirements (BR IDs), focus on those requirements
    10. If no specific BR IDs are mentioned, focus on the most important requirements based on the context
    
    User Query: {query}
    
    Context from Requirements and Documents:
    {context}
    
    Reasoning for why this agent was selected:
    {reasoning}
    
    Query Classification:
    {classification}
    
    Previous Chat History (for context):
    {chat_history}
    
    BR IDs mentioned or relevant:
    {br_ids}
    """
    query = state['query']
    
    # Ensure agent_reasoning is in the state
    if 'agent_reasoning' not in state:
        state['agent_reasoning'] = "Default reasoning: No specific agent reasoning provided."
    
    reasoning = state.get('agent_reasoning', "No specific reasoning provided")
    classification = state.get('query_classification', {})
    
    # Get BR IDs if available in session state
    br_ids = ""
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
        br_ids = "The following Business Requirement IDs are available:\n"
        for br_id, details in st.session_state.id_mappings.items():
            if br_id.startswith("BR-"):
                br_ids += f"- {br_id}: {details['content_snippet']}\n"
    
    # Emphasize that only supplied BRs should be used
    if not br_ids or br_ids == "The following Business Requirement IDs are available:\n":
        br_ids = "No business requirements have been uploaded. Please only generate functional requirements if explicitly mentioned in the query, and clearly state that no BR documents were available."
    
    # Get chat history for context retention
    chat_history = ""
    if hasattr(st, 'session_state') and 'chat_history' in st.session_state:
        chat_history = process_chat_history(st.session_state.chat_history)
    
    try:
        faiss_db = get_vector_store()
        docs = faiss_db.similarity_search(query)
        formatted_docs = format_docs(docs)
        state['documents'] = formatted_docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        formatted_docs = f"No context available due to error: {str(e)}"
        state['documents'] = formatted_docs
    
    # Extract specific BR ID mentioned in the query to ensure we only generate for this BR
    mentioned_br_ids = re.findall(r'BR-\d+', query)
    
    # If we have mentioned BR IDs, add special instruction to only generate for these BRs
    if mentioned_br_ids:
        br_instruction = f"\n\nIMPORTANT: ONLY generate functional requirements related to {', '.join(mentioned_br_ids)}. Do not include functional requirements for any other BR IDs."
        prompt += br_instruction
        
        # Check if the mentioned BR exists in the uploaded documents
        br_exists = False
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
            for br_id in mentioned_br_ids:
                if br_id in st.session_state.id_mappings:
                    br_exists = True
                    break
        
        if not br_exists:
            br_instruction += f"\n\nWARNING: The requested business requirement(s) {', '.join(mentioned_br_ids)} were not found in the uploaded documents. Only generate functional requirements if you can find relevant content in the provided context."
            prompt += br_instruction
    
    functional_req_template = ChatPromptTemplate.from_template(prompt)
    
    # Set a fixed temperature and random seed for consistency
    llm_with_seed = llm.with_config(configurable={"temperature": 0.0, "seed": 42})
    
    llm_chain = (
        {"query": itemgetter("query"), 
         "context": itemgetter("documents"),
         "reasoning": itemgetter("agent_reasoning"),
         "classification": itemgetter("query_classification"),
         "br_ids": lambda _: br_ids if 'br_ids' in locals() or 'br_ids' in globals() else mentioned_br_ids if mentioned_br_ids else [],
         "chat_history": lambda _: chat_history} 
        | functional_req_template 
        | llm_with_seed 
        | StrOutputParser()
    )
    
    try:
        # Check if this exact query has been processed before and is in session state or if there are existing FRs for the mentioned BR
        cache_key = f"fr_cache_{query.strip().lower()}"
        output_id = None
        
        # Check if there are existing functional requirements for this BR
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and mentioned_br_ids:
            for id_key, details in st.session_state.id_mappings.items():
                if (details.get('type') == 'Functional Requirement' and 
                    any(br_id in details.get('referenced_br_ids', []) for br_id in mentioned_br_ids)):
                    print(f"Found existing functional requirements for {', '.join(mentioned_br_ids)}, reusing")
                    functional_reqs = details.get('full_content', '')
                    output_id = id_key
                    break
        
        # If not found in existing mappings, check cache or generate new
        if not output_id:
            if hasattr(st, 'session_state') and cache_key in st.session_state:
                functional_reqs = st.session_state[cache_key]
                print(f"Using cached functional requirements for query: {query[:50]}...")
            else:
                functional_reqs = llm_chain.invoke({
                    "query": query, 
                    "documents": formatted_docs,
                    "agent_reasoning": reasoning,
                    "query_classification": classification
                })
                
                # Cache the result for future use
                if hasattr(st, 'session_state'):
                    st.session_state[cache_key] = functional_reqs
            
            # Generate unique ID only if we're creating new content
            output_id = f"FREQS-{str(uuid.uuid4())[:8]}"
        
        # Extract FR IDs from the output
        fr_ids = re.findall(r'FR-\d+', functional_reqs)
        
        # Extract BR IDs referenced in output
        br_ids_in_output = re.findall(r'BR-\d+', functional_reqs)
        
        # Create a mapping entry for this output if it's new
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and output_id not in st.session_state.id_mappings:
            st.session_state.id_mappings[output_id] = {
                "type": "Functional Requirement",
                "query": query,
                "fr_ids": fr_ids,
                "referenced_br_ids": br_ids_in_output,
                "content_snippet": functional_reqs[:100] + "..." if len(functional_reqs) > 100 else functional_reqs,
                "full_content": functional_reqs
            }
        
        # Add query classification and agent details to output
        query_analysis = f"""
## Query Analysis
| Category | Details |
|----------|---------|
| Query Type | {classification.get('query_type', 'Not specified')} |
| Intent | {classification.get('intent', 'Not specified')} |
| Complexity | {classification.get('complexity', 'Not specified')} |
| Key Terms | {', '.join(classification.get('key_terms', ['None']))} |
| Domain | {classification.get('domain', 'Not specified')} |

## Agent Selection
| Agent | Confidence Score |
|-------|-----------------|
| Test Case Generator | {state['confidence_scores'].get('test_case_generator', 0):.2f} |
| User Story Generator | {state['confidence_scores'].get('story_generator', 0):.2f} |
| Functional Requirement Generator | {state['confidence_scores'].get('functional_requirement_generator', 0):.2f} |
| Test Data Generator | {state['confidence_scores'].get('test_data_generator', 0):.2f} |

**Selected Agent**: Functional Requirement Generator  
**Reasoning**: {reasoning}

## Functional Requirements Output
Output ID: {output_id}

"""
        
        # Final response with analysis and generated requirements
        state['response'] = query_analysis + functional_reqs
    except Exception as e:
        print(f"Error in functional requirements agent: {e}")
        state['response'] = f"Could not generate functional requirements due to an error: {str(e)}"
    
    state['agent'] = "functional_req_agent"
    return state

# Add TestData agent for storing test results
def testdata_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Test Data Generator Agent - Creates test datasets for software testing.
    
    This agent specializes in generating comprehensive test datasets based on requirements.
    """
    prompt = """
    You are an expert Test Data Engineer specializing in creating comprehensive test datasets for software testing.
    Your task is to analyze the provided requirements and generate realistic, well-structured test data samples
    that can be used for testing the functionality described in the requirements.
    
    FORMAT YOUR TEST DATA RESPONSE USING THIS EXACT STRUCTURE:
    
    ## Test Data for [Feature/Requirement]
    
    ### Valid/Typical Data
    **Description:** Brief description of this data category
    **Purpose:** What these test data samples aim to validate
    **Test Cases:** List of test case IDs these data samples apply to (if applicable)
    ```json
    [
      {{
        // Well-formatted JSON data sample 1
      }},
      {{
        // Well-formatted JSON data sample 2
      }}
    ]
    ```
    
    ### Edge Case/Boundary Data
    **Description:** Brief description of this data category
    **Purpose:** What these edge cases aim to validate
    **Test Cases:** List of test case IDs these data samples apply to (if applicable)
    ```json
    [
      {{
        // Well-formatted JSON edge case data sample 1
      }},
      {{
        // Well-formatted JSON edge case data sample 2
      }}
    ]
    ```
    
    ### Invalid/Error Data
    **Description:** Brief description of this error data category
    **Purpose:** What error handling these samples aim to validate
    **Test Cases:** List of test case IDs these data samples apply to (if applicable)
    ```json
    [
      {{
        // Well-formatted JSON invalid data sample 1
      }},
      {{
        // Well-formatted JSON invalid data sample 2
      }}
    ]
    ```
    
    INSTRUCTIONS:
    1. Generate realistic test data based on the requirements
    2. Create data for positive scenarios, edge cases, and negative/error scenarios
    3. Ensure all data is well-formatted in proper JSON
    4. Use realistic values that match domain requirements (e.g., proper formats for dates, IDs, etc.)
    5. If dates/times are required, use ISO format (YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ)
    6. Include comments in the JSON samples to explain specific fields when helpful
    7. If the user mentions specific test cases (TC IDs), focus on generating data for those cases
    8. If the user mentions specific business requirements (BR IDs), focus on those requirements
    9. Only use information from provided test cases and business requirements
    10. If no relevant test cases or requirements are found, state that clearly
    11. Include both required and optional fields in your data samples when applicable
    12. For complex systems, provide data that tests integrations between components
    
    User Query: {query}
    
    Context from Requirements:
    {context}
    
    Reasoning for why this agent was selected:
    {reasoning}
    
    Query Classification:
    {classification}
    
    Previous Chat History (for context):
    {chat_history}
    """
    
    query = state['query']
    
    # Enhance prompt for specialized domains based on URL content or query
    if 'healthcare' in query.lower() or 'health' in query.lower() or 'medical' in query.lower() or 'hipaa' in query.lower() or 'fhir' in query.lower() or 'hl7' in query.lower():
        prompt += """
        
        HEALTHCARE IT SPECIFIC INSTRUCTIONS:
        1. For patient demographics, use realistic but fictional patient identifiers and information
        2. Include proper healthcare codes (ICD-10, CPT, LOINC, SNOMED CT) where applicable
        3. For HL7 FHIR resources, follow the exact structure from the FHIR specification
        4. Test data should include proper FHIR Bundle structures for submission scenarios
        5. Include appropriate resource references and identifiers according to FHIR conventions
        6. Generate Patient, Practitioner, Organization, and Encounter resources as needed
        7. For FHIR search parameters, include test data that tests different combinations
        8. Include metadata required by the US Core Implementation Guide for relevant resources
        9. Use realistic NPI numbers for providers and realistic but fictional patient identifiers
        10. For error cases, include invalid resource structures, invalid references, and validation errors
        """
    
    # Check URL content for specialized domain detection
    if state.get('use_url_context', False) and 'url_contents' in state:
        url_content_text = ""
        for url_data in state['url_contents']:
            url_content_text += url_data.get('content', '')
            
        # Healthcare domain detection from URL content
        healthcare_terms = ['FHIR', 'HL7', 'healthcare', 'patient', 'clinical', 'medical', 'HIPAA', 
                           'provider', 'encounter', 'diagnosis', 'procedure', 'EHR', 'EMR']
        
        if any(term.lower() in url_content_text.lower() for term in healthcare_terms):
            prompt += """
            
            HEALTHCARE IT SPECIFIC INSTRUCTIONS (detected from URL content):
            1. For FHIR resources, strictly follow the structure and required elements from the content provided in the URL
            2. If implementation guides or profiles are mentioned in the URL content, ensure test data conforms to those profiles
            3. Include extensions defined in the URL content with proper values
            4. For FHIR Bundle submissions, follow the exact Bundle structure documented in the URL
            5. Use example values from the URL content when available
            6. Include realistic but fictional values for all required fields in FHIR resources
            7. Test data should demonstrate proper use of FHIR data types (CodeableConcept, Reference, Identifier, etc.)
            8. For each FHIR resource type mentioned in the URL, include at least one valid and one invalid example
            9. Include examples of search parameters mentioned in the URL content
            10. If multiple versions of FHIR are mentioned, clearly indicate which version each test data sample targets
            11. Generate test data for both synchronous and asynchronous FHIR transactions if mentioned
            """
            
            # Add FHIR Bundle specific instructions if detected in URL content
            if 'bundle' in url_content_text.lower():
                prompt += """
                
                FHIR BUNDLE SPECIFIC INSTRUCTIONS:
                1. Create complete Bundle resources with proper bundle type (transaction, batch, collection, etc.)
                2. Include example Bundles with proper entry.request elements for transaction/batch types
                3. For transaction Bundles, include a mix of POST, PUT, and GET operations
                4. Demonstrate proper use of fullUrl in Bundle entries
                5. Show examples of Bundle.entry.resource for various resource types
                6. Include examples of conditional references within Bundles
                7. Demonstrate proper ordering of Bundle entries to handle dependencies
                8. Include both valid Bundles and Bundles with specific validation errors
                9. For batch requests, include examples with mixed success and error outcomes
                10. Show examples of Bundles with contained resources
                """
    
    # Ensure agent_reasoning is in the state
    if 'agent_reasoning' not in state:
        state['agent_reasoning'] = "Default reasoning: No specific agent reasoning provided."
    
    reasoning = state.get('agent_reasoning', "No specific reasoning provided")
    classification = state.get('query_classification', {})
    
    # Extract BR IDs and TC IDs from the query
    mentioned_br_ids = re.findall(r'BR-\d+', query)
    mentioned_tc_ids = re.findall(r'TC-\d+|US-\d+-TC-\d+', query)
    
    # Get chat history for context retention
    chat_history = ""
    if hasattr(st, 'session_state') and 'chat_history' in st.session_state:
        chat_history = process_chat_history(st.session_state.chat_history)
    
    # Check if using URL content instead of standard document retrieval
    if state.get('use_url_context', False) and 'url_contents' in state and state['url_contents']:
        # Format the URL content as the context instead of retrieved documents
        url_context = "URL CONTENT:\n\n"
        for i, url_data in enumerate(state['url_contents']):
            url_context += f"--- URL {i+1}: {url_data['title']} ({url_data['url']}) ---\n\n"
            # Include full content for URLs
            content = url_data['content']
            url_context += f"{content}\n\n"
        
        formatted_docs = url_context
        print(f"Using URL content as context for test data generation")
        
        # Add information to the prompt about URL-based context
        prompt += "\n\nIMPORTANT: Your primary context is the content from the URLs provided in the query. Generate test data based directly on this content rather than pre-existing business requirements or test cases. Focus on the data structures, APIs, or interfaces described in the URL content."
        
        # Add additional prompt for ensuring tabular output format for URL content
        prompt += "\n\nCRITICAL: Always maintain IT standard tabular output format even when working with URL content. Use markdown tables when presenting structured data. Ensure sections are clearly delineated with proper headings and formatting."
    else:
        # Check if document retrieval should be skipped based on relevance check
        should_retrieve_docs = True
        formatted_docs = ""
        
        if 'document_relevance' in state:
            # If query is not relevant to documents with high confidence, skip retrieval
            if not state['document_relevance']['relevant'] and state['document_relevance']['confidence'] > 0.7:
                should_retrieve_docs = False
                formatted_docs = "No document context used as query appears unrelated to uploaded documents."
                print("Skipping document retrieval as query is not relevant to documents")
        
        # Only retrieve documents if needed
        if should_retrieve_docs:
            try:
                faiss_db = get_vector_store()
                docs = faiss_db.similarity_search(query)
                formatted_docs = format_docs(docs)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                formatted_docs = f"No context available due to error: {str(e)}"
    
    # Store the documents in the state
    state['documents'] = formatted_docs
    
    # Define variables that will be needed later
    test_cases = []
    br_ids = mentioned_br_ids.copy() if mentioned_br_ids else []
    
    # Only apply BR/TC-specific instructions if we're not using URL content
    if not state.get('use_url_context', False):
        # If we have mentioned BR IDs or TC IDs, add special instruction to only generate for these IDs
        if mentioned_br_ids or mentioned_tc_ids:
            id_instruction = "\n\nIMPORTANT: "
            
            if mentioned_br_ids:
                id_instruction += f"ONLY generate test data related to {', '.join(mentioned_br_ids)}. "
                # Update br_ids for use in chain
                br_ids = mentioned_br_ids
            
            if mentioned_tc_ids:
                id_instruction += f"FOCUS on generating test data for test cases {', '.join(mentioned_tc_ids)}. "
                # Update test_cases for use in chain
                test_cases = mentioned_tc_ids
                
            id_instruction += "Do not generate test data for other requirements or test cases unless they are directly related."
            prompt += id_instruction
    
    # Create the prompt template
    testdata_template = ChatPromptTemplate.from_template(prompt)
    
    # Set a fixed temperature and random seed for consistency
    llm_with_seed = llm.with_config(configurable={"temperature": 0.0, "seed": 42})
    
    # Add a final reminder about output format
    prompt += "\n\nFINAL REMINDER: Your response MUST follow proper tabular format with markdown tables where appropriate. Always include the Query Analysis and Agent Selection sections in your output, followed by your test data output. This is CRITICAL for proper display."
    
    llm_chain = (
        {"query": itemgetter("query"), 
         "context": itemgetter("documents"),
         "reasoning": itemgetter("agent_reasoning"),
         "classification": itemgetter("query_classification"),
         "br_ids": lambda x: br_ids,
         "chat_history": lambda x: chat_history,
         "test_cases": lambda x: test_cases} 
        | testdata_template 
        | llm_with_seed 
        | StrOutputParser()
    )
    
    # Check if this exact query has been processed before and is in session state
    # Use a more specific cache key that includes the specific BR ID or TC ID
    # to avoid incorrect cache hits for different BRs
    cache_key_parts = []
    if mentioned_br_ids:
        cache_key_parts.extend(mentioned_br_ids)
    if mentioned_tc_ids:
        cache_key_parts.extend(mentioned_tc_ids)
    
    if cache_key_parts:
        specific_cache_key = f"td_cache_{'_'.join(cache_key_parts)}"
    else:
        specific_cache_key = f"td_cache_{query.strip().lower()}"
        
    if hasattr(st, 'session_state') and specific_cache_key in st.session_state:
        testdata = st.session_state[specific_cache_key]
        print(f"Using cached test data for query with ids: {', '.join(cache_key_parts)}")
    else:
        try:
            testdata = llm_chain.invoke({
                "query": query, 
                "documents": formatted_docs,
                "agent_reasoning": reasoning,
                "query_classification": classification,
                "test_cases": test_cases
            })
            
            # Verify that the output has proper structure; if not, add warning
            if "## Test Data for" not in testdata:
                print("WARNING: Output missing proper structure, adding headers")
                testdata = f"## Test Data for {query[:30]}...\n\n" + testdata
            
            # Ensure test data has proper sections
            required_sections = ["### Valid/Typical Data", "### Edge Case/Boundary Data", "### Invalid/Error Data"]
            missing_sections = [section for section in required_sections if section not in testdata]
            
            if missing_sections:
                print(f"WARNING: Output missing sections: {missing_sections}")
                for section in missing_sections:
                    testdata += f"\n\n{section}\n**Description:** No {section.replace('###', '').strip()} provided\n"
                
            # Cache the result for future use with the specific cache key
            if hasattr(st, 'session_state'):
                st.session_state[specific_cache_key] = testdata
                
        except Exception as e:
            print(f"Error generating test data: {e}")
            # Provide fallback response with proper structure
            testdata = f"""
## Test Data for {query[:50]}...

### Valid/Typical Data
**Description:** Example valid data based on query
**Purpose:** Basic validation
**Test Cases:** N/A

```json
[
  {{
    "example": "data",
    "note": "Error occurred during generation"
  }}
]
```

### Edge Case/Boundary Data
**Description:** Example edge case data
**Purpose:** Testing boundaries
**Test Cases:** N/A

```json
[
  {{
    "example": "edge case",
    "note": "Error occurred during generation" 
  }}
]
```

### Invalid/Error Data
**Description:** Example invalid data
**Purpose:** Testing error handling
**Test Cases:** N/A

```json
[
  {{
    "example": "invalid data",
    "error": "{str(e)}"
  }}
]
```
"""
    
    # Generate unique IDs for this output - use a deterministic ID based on BR/TC IDs when possible
    if mentioned_br_ids or mentioned_tc_ids:
        id_base = '_'.join(mentioned_br_ids + mentioned_tc_ids)
        output_id = f"TDS-{hashlib.md5(id_base.encode()).hexdigest()[:8]}"
    else:
        output_id = f"TDS-{str(uuid.uuid4())[:8]}"
    
    # Extract associated test case IDs from the output
    test_case_ids = re.findall(r'US-\d+-TC-\d+|TC-\d+', testdata)
    
    # Extract user story IDs referenced in output
    user_story_ids = re.findall(r'US-\d+', testdata)
    
    # Extract BR IDs referenced in output
    br_ids_in_output = re.findall(r'BR-\d+', testdata)
    
    # Create a mapping entry for this output if it's new
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and output_id not in st.session_state.id_mappings:
        st.session_state.id_mappings[output_id] = {
            "type": "Test Data",
            "query": query,
            "associated_test_case_ids": test_case_ids,
            "referenced_user_story_ids": user_story_ids,
            "referenced_br_ids": br_ids_in_output,
            "content_snippet": testdata[:100] + "..." if len(testdata) > 100 else testdata,
            "full_content": testdata
        }
    
    # Add query classification and agent details to output
    query_analysis = f"""
## Query Analysis
| Category | Details |
|----------|---------|
| Query Type | {classification.get('query_type', 'Not specified')} |
| Intent | {classification.get('intent', 'Not specified')} |
| Complexity | {classification.get('complexity', 'Not specified')} |
| Key Terms | {', '.join(classification.get('key_terms', ['None']))} |
| Domain | {classification.get('domain', 'Not specified')} |

## Agent Selection
| Agent | Confidence Score |
|-------|-----------------|
| Test Case Generator | {state['confidence_scores'].get('test_case_generator', 0):.2f} |
| User Story Generator | {state['confidence_scores'].get('story_generator', 0):.2f} |
| Functional Requirement Generator | {state['confidence_scores'].get('functional_requirement_generator', 0):.2f} |
| Test Data Generator | {state['confidence_scores'].get('test_data_generator', 0):.2f} |

**Selected Agent**: Test Data Generator  
**Reasoning**: {reasoning}

## Test Data Output
Output ID: {output_id}

"""
    
    # Final response with analysis and generated test data
    state['response'] = query_analysis + testdata
    state['agent'] = "testdata_agent"
    return state

# Extract content from a response, removing the query analysis section
def extract_content(response):
    """
    Extract content from a response, removing the query analysis section
    
    Args:
        response: Full response string including metadata
        
    Returns:
        str: Extracted content without metadata
    """
    # Find the position after the "Output ID" line and the following newlines
    if "Output ID:" in response:
        parts = response.split("Output ID:")
        if len(parts) > 1:
            # Get the second part (after Output ID)
            content_part = parts[1]
            # Skip the ID and any newlines to get to the content
            content_part = content_part.split("\n\n", 1)
            if len(content_part) > 1:
                return content_part[1]
    # If we couldn't parse it properly, return the original
    return response

def generate_combined_response(state: OrchestratorState) -> OrchestratorState:
    query = state["query"]
    
    # Check if we already have a combined response for this query
    cache_key = f"combined_cache_{query.strip().lower()}"
    
    # Extract specific BR ID mentioned in the query to ensure all agents focus on same BR ID
    mentioned_br_ids = re.findall(r'BR-\d+', query)
    
    # If we have BR IDs, create a more specific cache key
    if mentioned_br_ids:
        br_id_str = '_'.join(mentioned_br_ids)
        combined_cache_key = f"combined_cache_{br_id_str}"
    else:
        combined_cache_key = cache_key
    
    # Check if we have a cached combined response
    if hasattr(st, 'session_state') and combined_cache_key in st.session_state:
        print(f"Using cached combined response for query: {query[:50]}...")
        combined_response = st.session_state[combined_cache_key]
        
        # Generate a deterministic combined output ID based on BR IDs
        if mentioned_br_ids:
            id_base = '_'.join(mentioned_br_ids)
            combined_output_id = f"COMB-{hashlib.md5(id_base.encode()).hexdigest()[:8]}"
        else:
            combined_output_id = f"COMB-{str(uuid.uuid4())[:8]}"
        
        # Add query classification and agent details to the combined output
        classification = state.get('query_classification', {})
        reasoning = state.get('agent_reasoning', "No specific reasoning provided")
        
        query_analysis = f"""
## Query Analysis
| Category | Details |
|----------|---------|
| Query Type | {classification.get('query_type', 'Not specified')} |
| Intent | {classification.get('intent', 'Not specified')} |
| Complexity | {classification.get('complexity', 'Not specified')} |
| Key Terms | {', '.join(classification.get('key_terms', ['None']))} |
| Domain | {classification.get('domain', 'Not specified')} |

## Agent Selection
| Agent | Confidence Score |
|-------|-----------------|
| Test Case Generator | {state['confidence_scores'].get('test_case_generator', 0):.2f} |
| User Story Generator | {state['confidence_scores'].get('story_generator', 0):.2f} |
| Functional Requirement Generator | {state['confidence_scores'].get('functional_requirement_generator', 0):.2f} |
| Test Data Generator | {state['confidence_scores'].get('test_data_generator', 0):.2f} |

**Selected Agent**: Combined Artifact Generator  
**Reasoning**: {reasoning}

## Combined Output
Output ID: {combined_output_id}

"""
        
        # Extract all BR IDs referenced in the combined response
        all_br_ids = set(re.findall(r'BR-\d+', combined_response))
        
        # Create a mapping entry for this combined output if it doesn't exist
        if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and combined_output_id not in st.session_state.id_mappings:
            st.session_state.id_mappings[combined_output_id] = {
                "type": "Combined Output",
                "query": query,
                "referenced_br_ids": list(all_br_ids),
                "content_snippet": combined_response[:100] + "..." if len(combined_response) > 100 else combined_response,
                "full_content": combined_response
            }
        
        # Update the state with the combined response
        state['response'] = query_analysis + combined_response
        state['agent'] = "combined_agent"
        
        return state
    
    # Create copies of the state for each agent
    fr_state = state.copy()
    story_state = state.copy() 
    testcase_state = state.copy()
    testdata_state = state.copy()
    
    # Check if we already have outputs for the specified BR IDs
    fr_content = None
    story_content = None
    testcase_content = None
    testdata_content = None
    
    if mentioned_br_ids and hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
        br_id = mentioned_br_ids[0]
        
        # Check for existing functional requirements
        for id_key, details in st.session_state.id_mappings.items():
            if (details.get('type') == 'Functional Requirement' and 
                br_id in details.get('referenced_br_ids', []) and 
                'full_content' in details):
                print(f"Found existing functional requirements for {br_id}, reusing")
                fr_content = details.get('full_content', '')
                break
        
        # Check for existing user stories
        for id_key, details in st.session_state.id_mappings.items():
            if (details.get('type') == 'User Story' and 
                br_id in details.get('referenced_br_ids', []) and 
                'full_content' in details):
                print(f"Found existing user stories for {br_id}, reusing")
                story_content = details.get('full_content', '')
                break
        
        # Check for existing test cases
        for id_key, details in st.session_state.id_mappings.items():
            if (details.get('type') == 'Test Case' and 
                br_id in details.get('referenced_br_ids', []) and 
                'full_content' in details):
                print(f"Found existing test cases for {br_id}, reusing")
                testcase_content = details.get('full_content', '')
                break
        
        # Check for existing test data
        for id_key, details in st.session_state.id_mappings.items():
            if (details.get('type') == 'Test Data' and 
                br_id in details.get('referenced_br_ids', []) and 
                'full_content' in details):
                print(f"Found existing test data for {br_id}, reusing")
                testdata_content = details.get('full_content', '')
                break
    
    # Follow the correct workflow: BR → FR → US → TC → TD
    # Only generate what we don't already have
    
    # Step 1: Generate functional requirements from BR if not already available
    if not fr_content:
        fr_output = functional_req_agent(fr_state)
        fr_content = extract_content(fr_output['response'])
    
    # Step 2: Generate user stories from functional requirements if not already available
    if not story_content:
        # Make story_state aware of the functional requirements already generated
        if 'query' in story_state:
            # This will ensure the user stories use the exact same functional requirements
            fr_cache_key = ""
            if mentioned_br_ids:
                br_id = mentioned_br_ids[0]
                fr_query = f"Generate functional requirements for {br_id}"
                fr_cache_key = f"fr_cache_{fr_query.strip().lower()}"
                
                # Cache the fr content if not already cached
                if hasattr(st, 'session_state') and fr_cache_key not in st.session_state:
                    st.session_state[fr_cache_key] = fr_content
        
        # Now generate user stories based on the functional requirements
        story_output = Story_Agent(story_state)
        story_content = extract_content(story_output['response'])
    
    # Step 3: Generate test cases based on the user stories if not already available
    if not testcase_content:
        # Make testcase_state aware of the user stories already generated
        if 'query' in testcase_state:
            if mentioned_br_ids:
                br_id = mentioned_br_ids[0]
                story_query = f"Generate user stories for {br_id}"
                story_cache_key = f"us_cache_{story_query.strip().lower()}"
                
                # Cache the user stories if not already cached
                if hasattr(st, 'session_state') and story_cache_key not in st.session_state:
                    st.session_state[story_cache_key] = story_content
        
        # Now generate test cases based on the user stories
        testcase_output = testcase_agent(testcase_state)
        testcase_content = extract_content(testcase_output['response'])
    
    # Step 4: Generate test data based on the test cases if not already available
    if not testdata_content:
        # Make testdata_state aware of the test cases already generated
        if 'query' in testdata_state:
            # This will ensure the test data uses the exact same test cases
            if mentioned_br_ids:
                br_id = mentioned_br_ids[0]
                tc_query = f"Generate test cases for {br_id}"
                tc_cache_key = f"tc_cache_{br_id}"
                
                # Cache the test cases if not already cached
                if hasattr(st, 'session_state') and tc_cache_key not in st.session_state:
                    st.session_state[tc_cache_key] = testcase_content
        
        # Now generate test data based on the test cases
        testdata_output = testdata_agent(testdata_state)
        testdata_content = extract_content(testdata_output['response'])
    
    # Add query classification and agent details to the combined output
    classification = state.get('query_classification', {})
    reasoning = state.get('agent_reasoning', "No specific reasoning provided")
    
    # Generate a deterministic combined output ID based on BR IDs
    if mentioned_br_ids:
        id_base = '_'.join(mentioned_br_ids)
        combined_output_id = f"COMB-{hashlib.md5(id_base.encode()).hexdigest()[:8]}"
    else:
        combined_output_id = f"COMB-{str(uuid.uuid4())[:8]}"
    
    query_analysis = f"""
## Query Analysis
| Category | Details |
|----------|---------|
| Query Type | {classification.get('query_type', 'Not specified')} |
| Intent | {classification.get('intent', 'Not specified')} |
| Complexity | {classification.get('complexity', 'Not specified')} |
| Key Terms | {', '.join(classification.get('key_terms', ['None']))} |
| Domain | {classification.get('domain', 'Not specified')} |

## Agent Selection
| Agent | Confidence Score |
|-------|-----------------|
| Test Case Generator | {state['confidence_scores'].get('test_case_generator', 0):.2f} |
| User Story Generator | {state['confidence_scores'].get('story_generator', 0):.2f} |
| Functional Requirement Generator | {state['confidence_scores'].get('functional_requirement_generator', 0):.2f} |
| Test Data Generator | {state['confidence_scores'].get('test_data_generator', 0):.2f} |

**Selected Agent**: Combined Artifact Generator  
**Reasoning**: {reasoning}

## Combined Output
Output ID: {combined_output_id}

"""
    
    # Combine all outputs with clear section headers, following the workflow order
    combined_response = f"""# Comprehensive Analysis for: "{query}"

## 1. Functional Requirements
{fr_content}

## 2. User Stories
{story_content}

## 3. Test Cases
{testcase_content}

## 4. Test Data
{testdata_content}
"""
    
    # Cache the combined response for future use
    if hasattr(st, 'session_state'):
        st.session_state[combined_cache_key] = combined_response
    
    # Extract all BR IDs referenced in the combined response
    all_br_ids = set(re.findall(r'BR-\d+', combined_response))
    
    # Create a mapping entry for this combined output
    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state and combined_output_id not in st.session_state.id_mappings:
        st.session_state.id_mappings[combined_output_id] = {
            "type": "Combined Output",
            "query": query,
            "referenced_br_ids": list(all_br_ids),
            "content_snippet": combined_response[:100] + "..." if len(combined_response) > 100 else combined_response,
            "full_content": combined_response
        }
    
    # Update the state with the combined response
    state['response'] = query_analysis + combined_response
    state['agent'] = "combined_agent"
    
    return state

# Modify the route_to_agent function to include the combined response option
def route_to_agent(state: OrchestratorState) -> OrchestratorState:
    # Check if the query contains keywords suggesting a need for all artifacts
    query = state['query'].lower()
    
    # First, extract any BR IDs from the query to help with routing
    mentioned_br_ids = re.findall(r'BR-\d+', state['query'])
    mentioned_tc_ids = re.findall(r'TC-\d+|US-\d+-TC-\d+', state['query'])
    
    # Check if we have URL content and a pre-selected agent
    if state.get('use_url_context', False) and 'selected_agent' in state and state['selected_agent']:
        print(f"Using pre-selected agent {state['selected_agent']} for URL content")
        # Ensure agent is set to match selected_agent
        state['agent'] = state['selected_agent']
        return state
    
    # For URL content without a pre-selected agent, choose based on the query
    if state.get('use_url_context', False):
        # Set the default agent for URL content based on query analysis
        if "test case" in query or "testcase" in query or "testing scenario" in query:
            state["selected_agent"] = "testcase_agent"
            state["agent"] = "testcase_agent"
            print("Selected testcase_agent based on URL content and query analysis")
        elif any(term in query for term in [
            "test data", "sample data", "test dataset", "data for test", 
            "testing data", "data generation", "generate data", "create test data"
        ]):
            state["selected_agent"] = "testdata_agent"
            state["agent"] = "testdata_agent"
            print("Selected testdata_agent based on URL content and query analysis")
        elif "user story" in query or "user stories" in query:
            state["selected_agent"] = "story_agent"
            state["agent"] = "story_agent"
            print("Selected story_agent based on URL content and query analysis")
        elif "requirement" in query or "functional" in query:
            state["selected_agent"] = "functional_req_agent"
            print("Selected functional_req_agent based on URL content and query analysis")
        else:
            # Default to test case agent for URL content
            state["selected_agent"] = "testcase_agent"
            print("Defaulting to testcase_agent for URL content (no specific agent mentioned)")
        return state
    
    # Standard routing for non-URL queries
    if "all artifacts" in query or "generate all" in query or "combined" in query or "comprehensive" in query:
        state["selected_agent"] = "combined_agent"
    
    # More precise detection of test data requests to avoid misrouting
    elif any(term in query for term in [
        "test data", "sample data", "test dataset", "data for test", 
        "testing data", "data generation", "generate data", "create test data"
    ]) and not any(term in query for term in [
        "create test case", "generate test case", "new test case",
        "test case creation", "make test case", "write test case"
    ]):
        # It's likely a test data request
        state["selected_agent"] = "testdata_agent"
    
    # If test cases are mentioned but we already have them for the BR, check if this is actually a test data request
    elif (any(term in query for term in ["test case", "test scenario", "test script"]) and
          mentioned_br_ids and hasattr(st, 'session_state') and 'id_mappings' in st.session_state):
        
        # Check if we already have test cases for this BR
        br_id = mentioned_br_ids[0]
        existing_test_cases = False
        
        for id_key, details in st.session_state.id_mappings.items():
            if (details.get('type') == 'Test Case' and 
                br_id in details.get('referenced_br_ids', [])):
                existing_test_cases = True
                break
        
        # If we have existing test cases and the query suggests test data rather than updating test cases
        if existing_test_cases and any(term in query for term in [
            "data", "values", "inputs", "parameters", "with what", "what data"
        ]) and not any(term in query for term in [
            "update test case", "new test case", "modify test case", "change test case"
        ]):
            state["selected_agent"] = "testdata_agent"
        else:
            # It's a genuine test case request, possibly to update existing ones
            state["selected_agent"] = "testcase_agent"
    
    # Explicit test case requests
    elif "test case" in query or "testcase" in query or "test scenario" in query:
        state["selected_agent"] = "testcase_agent"
    
    # Return the state with the potentially updated selected_agent
    return state

# Add this function definition before OrchestratorGraph.add_node calls
# Add before line 1340 (just before the graph nodes are added)

# Router function that determines which path to take
def route_decision(state: OrchestratorState) -> OrchestratorState:
    # Simply pass through the state with no changes
    return state

# Function to handle fallback response for out-of-scope queries
def handle_out_of_scope(state: OrchestratorState) -> OrchestratorState:
    """
    Generate a polite fallback response for out-of-scope queries.
    """
    query = state['query']
    
    # Double-check if the query mentions any document IDs or keywords from documents
    # This is a safety check to prevent false positives
    if hasattr(st, 'session_state') and st.session_state.id_mappings:
        # Check for BR IDs or other document references
        mentioned_br_ids = re.findall(r'BR-\d+', query)
        mentioned_sys_ids = re.findall(r'SYS-ARCH-\d+', query)
        
        # If any document IDs are explicitly mentioned, don't treat as out of scope
        if mentioned_br_ids or mentioned_sys_ids:
            print(f"Query contains document IDs {mentioned_br_ids + mentioned_sys_ids}, redirecting to appropriate agent")
            # Route to the appropriate agent based on the query classification
            if "test case" in query.lower() or "testing" in query.lower():
                state['selected_agent'] = "testcase_agent"
            elif "user story" in query.lower() or "feature" in query.lower():
                state['selected_agent'] = "story_agent"
            elif "requirement" in query.lower() or "functional" in query.lower():
                state['selected_agent'] = "functional_req_agent"
            elif "test data" in query.lower() or "sample data" in query.lower():
                state['selected_agent'] = "testdata_agent"
            else:
                # Default to combined agent if we can't determine specific agent
                state['selected_agent'] = "combined_agent"
            
            # Redirect to route_to_agent rather than showing fallback message
            return route_to_agent(state)
    
    # If we get here, the query is truly out of scope
    prompt = """
    You are a helpful AI assistant that specializes in software development artifacts like test cases, user stories, 
    and functional requirements. You've been asked a question that appears to be outside your specific area of expertise.
    
    The user query is: {query}
    
    Please craft a polite, helpful response explaining:
    1. That the query appears to be outside your specific focus areas
    2. What your core capabilities are (generating test cases, user stories, functional requirements, test data)
    3. Suggest how the user might rephrase their question to align with your capabilities
    4. Or, if appropriate, suggest alternative resources they might use
    
    Be respectful but firm in directing the conversation back to your core capabilities.
    
    YOUR RESPONSE SHOULD BE CONCISE BUT HELPFUL:
    """
    
    fallback_template = ChatPromptTemplate.from_template(prompt)
    chain = fallback_template | llm | StrOutputParser()
    
    try:
        fallback_response = chain.invoke({"query": query})
    except Exception as e:
        print(f"Error generating fallback response: {e}")
        # Provide a generic fallback if the LLM call fails
        fallback_response = (
            "I apologize, but your question appears to be outside my specific focus areas. " 
            "I'm specialized in creating software development artifacts like test cases, user stories, "
            "functional requirements, and test data. Could you rephrase your question to align with these capabilities?"
        )
    
    state['response'] = fallback_response
    state['agent'] = "fallback_agent"
    state['selected_agent'] = "fallback_agent"
    
    return state

# Add nodes to the graph
OrchestratorGraph.add_node("query_classifier", query_classifier)
OrchestratorGraph.add_node("agent_selector", agent_selector)
OrchestratorGraph.add_node("check_confidence", check_confidence)
OrchestratorGraph.add_node("analyze_further", analyze_further)
OrchestratorGraph.add_node("story_agent", Story_Agent)
OrchestratorGraph.add_node("testcase_agent", testcase_agent)
OrchestratorGraph.add_node("functional_req_agent", functional_req_agent)
OrchestratorGraph.add_node("testdata_agent", testdata_agent)
OrchestratorGraph.add_node("combined_agent", generate_combined_response)
OrchestratorGraph.add_node("route_to_agent", route_to_agent)
OrchestratorGraph.add_node("handle_out_of_scope", handle_out_of_scope)

# Set the entry point
OrchestratorGraph.set_entry_point("query_classifier")

# Add edge from query_classifier to agent_selector
OrchestratorGraph.add_edge("query_classifier", "agent_selector")

# Add conditional edges from agent_selector based on confidence
OrchestratorGraph.add_conditional_edges(
    "agent_selector",
    check_confidence,
    {
        "proceed": "route_to_agent",  # Continue to the next conditional edge
        "analyze_further": "analyze_further",
        "out_of_scope": "handle_out_of_scope"  # New edge to out of scope handler
    }
)
OrchestratorGraph.add_edge("analyze_further", "route_to_agent")
OrchestratorGraph.add_edge("handle_out_of_scope", END)

# Add conditional edges to route to the appropriate agent
OrchestratorGraph.add_conditional_edges(
    "route_to_agent",
    lambda state: state["selected_agent"],
    {
        "story_agent": "story_agent",
        "testcase_agent": "testcase_agent",
        "functional_req_agent": "functional_req_agent",
        "testdata_agent": "testdata_agent",
        "combined_agent": "combined_agent"
    }
)

# Add edge for new combined agent
OrchestratorGraph.add_edge("combined_agent", END)
OrchestratorGraph.add_edge("testdata_agent", END)

# Compile the workflow
workflow = OrchestratorGraph.compile()

# Visualization function
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(agent_graph):
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes and edges from the AgentGraph
    for node in agent_graph.nodes:
        G.add_node(node)

    for edge in agent_graph.edges:
        G.add_edge(edge[0], edge[1])

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes with fixed seed for reproducibility
    plt.figure(figsize=(12, 8))  # Larger figure size
    
    # Node colors based on node type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == "query_classifier":
            node_colors.append("#FFD700")  # Gold
            node_sizes.append(2000)
        elif node == "agent_selector":
            node_colors.append("#ADD8E6")  # Light blue
            node_sizes.append(2000)
        elif node == "check_confidence":
            node_colors.append("#FFA07A")  # Light salmon
            node_sizes.append(2000)
        elif node in ["story_agent", "testcase_agent", "functional_req_agent"]:
            node_colors.append("#90EE90")  # Light green
            node_sizes.append(2000)
        elif node == "END":
            node_colors.append("#D3D3D3")  # Light gray
            node_sizes.append(1500)
        else:
            node_colors.append("#D3D3D3")  # Light gray
            node_sizes.append(1500)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray', arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title("Multi-Agent Orchestrator Graph with Reasoning Flow", fontsize=16)
    plt.axis('off')  # Turn off axis
    
    # Save the plot to a file
    plt.savefig("orchestrator_graph_advanced.png", dpi=300, bbox_inches='tight')  # Higher resolution
    plt.close()  # Close the plot

# Draw the graph
draw_graph(OrchestratorGraph)

# Example usage
def test_orchestrator():
    # Test case 1: Query that should go to the test case generator
    test_query = "Create test cases for a login system that validates user credentials and provides appropriate error messages"
    result = workflow.invoke({"query": test_query})
    print("\n\n=== TEST CASE 1: Test Case Generator ===")
    print(f"Query: {test_query}")
    print(f"Classification: {result['query_classification']}")
    print(f"Selected Agent: {result['selected_agent']}")
    print(f"Reasoning: {result['agent_reasoning']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Response Preview: {result['response'][:200]}...\n")
    
    # Test case 2: Query that should go to the story generator
    story_query = "Develop user stories for a feature that allows customers to track their order status in real-time"
    result = workflow.invoke({"query": story_query})
    print("\n=== TEST CASE 2: Story Generator ===")
    print(f"Query: {story_query}")
    print(f"Classification: {result['query_classification']}")
    print(f"Selected Agent: {result['selected_agent']}")
    print(f"Reasoning: {result['agent_reasoning']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Response Preview: {result['response'][:200]}...\n")
    
    # Test case 3: Query that should go to the functional requirement generator
    fr_query = "Define the functional requirements for a password reset feature that includes email verification"
    result = workflow.invoke({"query": fr_query})
    print("\n=== TEST CASE 3: Functional Requirement Generator ===")
    print(f"Query: {fr_query}")
    print(f"Classification: {result['query_classification']}")
    print(f"Selected Agent: {result['selected_agent']}")
    print(f"Reasoning: {result['agent_reasoning']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Response Preview: {result['response'][:200]}...\n")
    
    # Test case 4: Ambiguous query that might need further analysis
    ambiguous_query = "I need to document the system's behavior when users interact with it"
    result = workflow.invoke({"query": ambiguous_query})
    print("\n=== TEST CASE 4: Ambiguous Query ===")
    print(f"Query: {ambiguous_query}")
    print(f"Classification: {result['query_classification']}")
    print(f"Selected Agent: {result['selected_agent']}")
    print(f"Reasoning: {result['agent_reasoning']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Agent Selection History: {json.dumps(result['agent_selection_history'], indent=2)}")
    print(f"Response Preview: {result['response'][:200]}...\n")

# Uncomment to run the test
# test_orchestrator()

# Streamlit interface
# Initialize session state
initialize_state()

# Set wide mode for the page to maximize display space
st.set_page_config(layout="wide")

# Update the sidebar section
with st.sidebar:
    st.header("Upload Documents")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document", 
                                    type=["pdf", "docx", "txt", "png", "jpg", "csv", "xlsx"],
                                    help="Upload business requirements, system architecture diagrams, or other project documents")
    
    document_type = st.selectbox("Document Type", 
                                ["Business Requirement", "System Architecture", "Other"],
                                help="Select the type of document you're uploading")
    
    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Processing document..."):
            # Mark that we're uploading a new file
            handle_file_upload()
            
            # Extract text from the uploaded file
            text = extract_text_from_file(uploaded_file)
            
            # Update document store
            success, extracted_ids = update_document_store(text, document_type, uploaded_file.name)
            
            if success:
                st.success(f"Successfully processed '{uploaded_file.name}'")
                
                # If IDs were found, show them
                if extracted_ids:
                    st.write(f"Found {len(extracted_ids)} items:")
                    for id_key in list(extracted_ids.keys())[:5]:  # Show only first 5 to save space
                        st.write(f"- {id_key}")
                    if len(extracted_ids) > 5:
                        st.write(f"... and {len(extracted_ids) - 5} more")
            else:
                st.error("Failed to process document. Please try again.")
    
    # Show uploaded documents details
    if hasattr(st, 'session_state') and 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        st.subheader("Uploaded Documents")
        with st.expander("View All Uploaded Documents", expanded=True):
            for doc_id, doc_info in st.session_state.uploaded_documents.items():
                st.write(f"**{doc_info['filename']}** ({doc_info['type']})")
                st.write(f"Uploaded: {doc_info['timestamp']}")
                st.write(f"IDs: {', '.join(doc_info['extracted_ids'][:5])}{'...' if len(doc_info['extracted_ids']) > 5 else ''}")
                st.divider()
    
    # Show available IDs by type
    if st.session_state.id_mappings:
        st.subheader("Available IDs")
        
        # Group by type
        id_types = {}
        for id_key, details in st.session_state.id_mappings.items():
            id_type = details.get('type', 'Other')
            if id_type not in id_types:
                id_types[id_type] = []
            id_types[id_type].append((id_key, details))
        
        # Display Business Requirements first
        if "Business Requirement" in id_types:
            with st.expander(f"Business Requirements ({len(id_types['Business Requirement'])})", expanded=True):
                for id_key, details in id_types["Business Requirement"]:
                    st.write(f"**{id_key}**: {details.get('content_snippet', '')}")
        
        # Display System Architecture second
        if "System Architecture" in id_types:
            with st.expander(f"System Architecture ({len(id_types['System Architecture'])})", expanded=True):
                for id_key, details in id_types["System Architecture"]:
                    st.write(f"**{id_key}**: {details.get('content_snippet', '')}")
        
        # Display other types
        for id_type, items in id_types.items():
            if id_type not in ["Business Requirement", "System Architecture"]:
                with st.expander(f"{id_type} ({len(items)})"):
                    for id_key, details in items:
                        st.write(f"**{id_key}**: {details.get('content_snippet', '')}")
    
    # Show option to clear all documents
    if st.session_state.document_store is not None and st.button("Clear All Documents"):
        st.session_state.document_store = None
        st.session_state.id_mappings = {}
        if 'uploaded_documents' in st.session_state:
            st.session_state.uploaded_documents = {}
        st.success("All documents cleared")
    
    # Chat history section in sidebar
    st.header("Chat History")
    
    # Add buttons for chat management
    new_chat_col, clear_chat_col = st.columns(2)
    with new_chat_col:
        if st.button("New Chat", key="new_chat_sidebar"):
            clear_chat()
            st.rerun()
    
    with clear_chat_col:
        if st.button("Clear Chat", key="clear_chat_sidebar"):
            clear_chat()
            st.rerun()
    
    # Display past chats
    if st.session_state.all_chats:
        for idx, chat in enumerate(st.session_state.all_chats):
            if st.button(f"{chat['timestamp']} - {chat['title']}", key=f"chat_{idx}"):
                # Load this chat
                st.session_state.chat_history = chat['messages']
                st.session_state.current_chat_id = chat['id']
                st.rerun()
    
    # Add help section in sidebar
    with st.expander("Help & Info"):
        st.markdown("""
        **Tips for Test Cases:**
        - Ask for "test cases for [feature]" to generate detailed test cases
        - Request "test data for [test case ID]" to get specific test data
        - Use BR-XXX references for more relevant test cases
        - For full screen display, collapse sidebar using ◀ icon
        """)

# Update the main page to include information about all agents including Test Data agent
st.markdown("<h1 style='text-align: center;'>AI-Orchestrated System for Test Engineering</h1>", unsafe_allow_html=True)
st.write("""
This intelligent orchestrator analyzes your requirements, classifies them, and routes them to the most appropriate specialized agent:
- **Test Case Generator**: Creates structured test cases for software features
- **User Story Generator**: Creates user stories that capture user needs and requirements
- **Functional Requirement Generator**: Creates formal functional requirements
- **Test Data Generator**: Creates comprehensive test data sets linked to test cases with sample values and edge cases
""")

# Update the tabs to remove Help & Limitations and Upload Documents tabs (since they're now in the sidebar)
tab1, tab2 = st.tabs(["Chat Interface", "ID Mapping & Traceability"])

with tab1:
    # Welcome message
    if not st.session_state.chat_history:
        st.markdown("<h2 style='text-align: center;'>How can I help you today?</h2>", unsafe_allow_html=True)
            
    # Display chat history in a continuous conversation flow
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    # Display assistant message without numbering
                    st.markdown(f"**Assistant:** {message['content']}", unsafe_allow_html=True)
                st.divider()
    
    # Input area
    query = st.text_area("Enter your query:", height=120, 
                        placeholder="Example: Generate test cases for login functionality based on BR-001")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate"):
            if not query:
                st.warning("Please enter a query")
            else:
                # Clear current chat if a new file was uploaded
                if st.session_state.new_file_uploaded and st.session_state.chat_history:
                    clear_chat()
                    st.session_state.new_file_uploaded = False
                
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                with st.spinner("Processing your query..."):
                    # Check if the query references specific IDs (both BR and SYS-ARCH)
                    referenced_br_ids = re.findall(r'BR-\d+', query)
                    referenced_sys_ids = re.findall(r'SYS-ARCH-\d+', query)
                    
                    # Enhanced query with both BR and System Architecture information
                    enhanced_query = query
                    
                    # If specific IDs are referenced and exist in our mappings, add their content to the query
                    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
                        # Add BR content if BR IDs are mentioned
                        if referenced_br_ids:
                            br_contents = []
                            for br_id in referenced_br_ids:
                                if br_id in st.session_state.id_mappings:
                                    br_detail = st.session_state.id_mappings[br_id]
                                    if 'content_snippet' in br_detail:
                                        br_contents.append(f"{br_id}: {br_detail['content_snippet']}")
                        
                            if br_contents:
                                enhanced_query = f"{enhanced_query}\n\nReferenced Business Requirements:\n" + "\n".join(br_contents)
                        
                        # Add System Architecture content if SYS-ARCH IDs are mentioned
                        if referenced_sys_ids:
                            sys_contents = []
                            for sys_id in referenced_sys_ids:
                                if sys_id in st.session_state.id_mappings:
                                    sys_detail = st.session_state.id_mappings[sys_id]
                                    if 'content_snippet' in sys_detail:
                                        sys_contents.append(f"{sys_id}: {sys_detail['content_snippet']}")
                        
                            if sys_contents:
                                enhanced_query = f"{enhanced_query}\n\nReferenced System Architecture Components:\n" + "\n".join(sys_contents)
                        
                        # If BR IDs mentioned but no SYS-ARCH IDs mentioned, include relevant sys arch content
                        if referenced_br_ids and not referenced_sys_ids:
                            # Find all System Architecture entries
                            sys_arch_entries = []
                            for id_key, details in st.session_state.id_mappings.items():
                                if details.get('type') == 'System Architecture':
                                    sys_arch_entries.append(f"{id_key}: {details.get('content_snippet', '')}")
                        
                            # Add up to 3 most relevant System Architecture entries
                            if sys_arch_entries:
                                enhanced_query = f"{enhanced_query}\n\nRelevant System Architecture Information:\n" + "\n".join(sys_arch_entries[:3])
                    
                    # Log the enhanced query for debugging
                    print(f"Enhanced query: {enhanced_query[:200]}...")
                    
                    # Invoke the workflow
                    result = workflow.invoke({"query": enhanced_query})
                    
                    # Display the response in a full-width container
                    full_width_container = st.container()
                    with full_width_container:
                        # Use markdown with unsafe_allow_html to ensure proper table rendering
                        response_content = result['response']
                        
                        # Display which agent was used (prepend to response)
                        agent_names = {
                            "testcase_agent": "Test Case Generator",
                            "story_agent": "User Story Generator",
                            "functional_req_agent": "Functional Requirement Generator",
                            "testdata_agent": "Test Data Generator",
                            "combined_agent": "Combined Artifact Generator"
                        }
                        agent_used = agent_names.get(result['agent'], result['agent'])
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_content,
                        "agent": agent_used,
                        "query_classification": result['query_classification'],
                        "confidence_scores": result['confidence_scores']
                    })
                    
                    # Rerun to update the chat history display
                    st.rerun()
    
    with col2:
        if st.button("Generate All Artifacts"):
            if not query:
                st.warning("Please enter a query")
            else:
                # Clear current chat if a new file was uploaded
                if st.session_state.new_file_uploaded and st.session_state.chat_history:
                    clear_chat()
                    st.session_state.new_file_uploaded = False
                
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": query + " [Generate all artifacts]"})
                
                with st.spinner("Processing your query and generating all artifacts..."):
                    # Check if the query references specific IDs (both BR and SYS-ARCH)
                    referenced_br_ids = re.findall(r'BR-\d+', query)
                    referenced_sys_ids = re.findall(r'SYS-ARCH-\d+', query)
                    
                    # Enhanced query with both BR and System Architecture information
                    enhanced_query = query + " [Generate all artifacts]"
                    
                    # If specific IDs are referenced and exist in our mappings, add their content to the query
                    if hasattr(st, 'session_state') and 'id_mappings' in st.session_state:
                        # Add BR content if BR IDs are mentioned
                        if referenced_br_ids:
                            br_contents = []
                            for br_id in referenced_br_ids:
                                if br_id in st.session_state.id_mappings:
                                    br_detail = st.session_state.id_mappings[br_id]
                                    if 'content_snippet' in br_detail:
                                        br_contents.append(f"{br_id}: {br_detail['content_snippet']}")
                        
                            if br_contents:
                                enhanced_query = f"{enhanced_query}\n\nReferenced Business Requirements:\n" + "\n".join(br_contents)
                        
                        # Add System Architecture content if SYS-ARCH IDs are mentioned
                        if referenced_sys_ids:
                            sys_contents = []
                            for sys_id in referenced_sys_ids:
                                if sys_id in st.session_state.id_mappings:
                                    sys_detail = st.session_state.id_mappings[sys_id]
                                    if 'content_snippet' in sys_detail:
                                        sys_contents.append(f"{sys_id}: {sys_detail['content_snippet']}")
                        
                            if sys_contents:
                                enhanced_query = f"{enhanced_query}\n\nReferenced System Architecture Components:\n" + "\n".join(sys_contents)
                        
                        # If BR IDs mentioned but no SYS-ARCH IDs mentioned, include relevant sys arch content
                        if referenced_br_ids and not referenced_sys_ids:
                            # Find all System Architecture entries
                            sys_arch_entries = []
                            for id_key, details in st.session_state.id_mappings.items():
                                if details.get('type') == 'System Architecture':
                                    sys_arch_entries.append(f"{id_key}: {details.get('content_snippet', '')}")
                        
                            # Add up to 3 most relevant System Architecture entries
                            if sys_arch_entries:
                                enhanced_query = f"{enhanced_query}\n\nRelevant System Architecture Information:\n" + "\n".join(sys_arch_entries[:3])
                
                    # Create a custom state with the combined agent pre-selected
                    custom_state = {"query": enhanced_query}
                    
                    # Process the query through the normal workflow first to get classification
                    initial_result = workflow.invoke(custom_state)
                    
                    # Then force it to use the combined agent
                    custom_state = initial_result.copy()
                    custom_state["selected_agent"] = "combined_agent"
                    
                    # Get the combined result
                    result = generate_combined_response(custom_state)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result['response'],
                        "agent": "Combined Artifacts"
                    })
                    
                    # Rerun to update the chat history display
                    st.rerun()

with tab2:
    st.header("ID Mapping & Traceability")
    
    # Create a network graph visualization of the ID mappings
    if st.session_state.id_mappings:
        st.subheader("Requirement Traceability")
        
        # Group by type for better organization
        id_types = {}
        for id_key, details in st.session_state.id_mappings.items():
            id_type = details.get('type', 'Other')
            if id_type not in id_types:
                id_types[id_type] = []
            id_types[id_type].append((id_key, details))
        
        # Show type distribution
        col1, col2 = st.columns(2)
        with col1:
            type_counts = {key: len(items) for key, items in id_types.items()}
            st.subheader("Artifact Counts")
            for type_name, count in type_counts.items():
                st.metric(label=type_name, value=count)
        
        # Show traceability matrix
        trace_data = []
        
        # Include Test Data in traceability
        for id_key, details in st.session_state.id_mappings.items():
            # Add referenced BR IDs
            if 'referenced_br_ids' in details and details['referenced_br_ids']:
                for ref_id in details['referenced_br_ids']:
                    trace_data.append({
                        'Source ID': id_key,
                        'Source Type': details.get('type', 'Unknown'),
                        'Referenced ID': ref_id,
                        'Referenced Type': st.session_state.id_mappings.get(ref_id, {}).get('type', 'Business Requirement'),
                        'Relationship': 'Implements'
                    })
            
            # Add test data to test case relationships
            if details.get('type') == 'Test Data' and 'associated_test_case_ids' in details:
                for tc_id in details['associated_test_case_ids']:
                    trace_data.append({
                        'Source ID': id_key,
                        'Source Type': 'Test Data',
                        'Referenced ID': tc_id,
                        'Referenced Type': 'Test Case',
                        'Relationship': 'Validates'
                    })
            
            # Add test case to user story relationships
            if details.get('type') == 'Test Case' and 'referenced_user_story_ids' in details:
                for us_id in details['referenced_user_story_ids']:
                    trace_data.append({
                        'Source ID': id_key,
                        'Source Type': 'Test Case',
                        'Referenced ID': us_id,
                        'Referenced Type': 'User Story',
                        'Relationship': 'Tests'
                    })
        
        if trace_data:
            st.subheader("Traceability Matrix")
            trace_df = pd.DataFrame(trace_data)
            st.dataframe(trace_df, use_container_width=True)
            
            # Add filter options
            st.subheader("Filter by Artifact Type")
            selected_type = st.selectbox("Select artifact type", 
                                        ["All Types"] + list(id_types.keys()))
            
            if selected_type != "All Types":
                filtered_items = id_types.get(selected_type, [])
                st.subheader(f"{selected_type} Items")
                for id_key, details in filtered_items:
                    with st.expander(f"{id_key}"):
                        # For Test Data, show associated test cases
                        if selected_type == "Test Data" and 'associated_test_case_ids' in details:
                            st.write("**Associated Test Cases:**")
                            for tc_id in details['associated_test_case_ids']:
                                st.write(f"- {tc_id}")
                        
                        # Show content snippet
                        st.write(f"**Content Preview**: {details.get('content_snippet', '')}")
                        
                        # Show references
                        if 'referenced_br_ids' in details and details['referenced_br_ids']:
                            st.write(f"**References**: {', '.join(details['referenced_br_ids'])}")
                        
                        # Show source query
                        if 'query' in details:
                            st.write(f"**Generated from query**: {details['query']}")
            
        else:
            st.info("No traceability links established yet. Generate more content to see relationships.")
        
        # Search functionality
        st.subheader("Search Requirements")
        search_term = st.text_input("Search for ID or content:", 
                                    placeholder="Enter BR ID, Output ID, or keywords")
        
        if search_term:
            search_results = []
            for id_key, details in st.session_state.id_mappings.items():
                # Check if the search term is in the ID or content
                if (search_term.lower() in id_key.lower() or
                    search_term.lower() in details.get('content_snippet', '').lower()):
                    search_results.append((id_key, details))
            
            if search_results:
                st.write(f"Found {len(search_results)} matching items:")
                for id_key, details in search_results:
                    with st.expander(f"{id_key} ({details.get('type', 'Unknown')})"):
                        st.write(f"**Content Preview**: {details.get('content_snippet', '')}")
                        if 'referenced_br_ids' in details and details['referenced_br_ids']:
                            st.write(f"**References**: {', '.join(details['referenced_br_ids'])}")
                        if 'query' in details:
                            st.write(f"**Generated from query**: {details['query']}")
            else:
                st.warning(f"No results found for '{search_term}'")

def extract_json_from_text(text):
    """Extract JSON from text that might contain additional explanations."""
    import re
    import json
    
    # Try to find JSON-like patterns in the text
    json_pattern = r'({[\s\S]*})'
    match = re.search(json_pattern, text)
    
    if match:
        potential_json = match.group(1)
        try:
            # Try to parse it as JSON
            parsed_json = json.loads(potential_json)
            return parsed_json
        except json.JSONDecodeError:
            pass
    
    # If we couldn't extract valid JSON, raise an error
    raise ValueError(f"Could not extract valid JSON from text: {text[:100]}...")

# Find and replace all instances of st.experimental_rerun()
if "generate_clicked" in st.session_state and st.session_state.generate_clicked:
    # Reset flag
    st.session_state.generate_clicked = False
    # Rerun to update UI
    st.rerun()

# Fix any other instances
if "generate_all_clicked" in st.session_state and st.session_state.generate_all_clicked:
    # Reset flag
    st.session_state.generate_all_clicked = False
    # Rerun to update UI
    st.rerun()

# Function to check if a query is relevant to the uploaded documents
def check_query_relevance(query: str) -> Dict[str, Any]:
    """
    Check if the user query is semantically relevant to the uploaded documents.
    
    Args:
        query: The user query string
        
    Returns:
        dict: A dictionary with relevance information
            - relevant (bool): Whether the query is relevant to docs
            - confidence (float): Confidence score (0-1)
            - reasoning (str): Explanation of the decision
    """
    # If no documents are uploaded, it's definitely not relevant
    if not hasattr(st, 'session_state') or not st.session_state.id_mappings:
        return {
            "relevant": False,
            "confidence": 1.0,
            "reasoning": "No documents have been uploaded, so the query cannot be relevant to any documents."
        }
    
    # Extract all document content into a summary representation
    doc_summary = ""
    for doc_id, details in st.session_state.id_mappings.items():
        # Add document type and ID to summary
        doc_summary += f"{details.get('type', 'Document')} {doc_id}: "
        # Add content snippet
        doc_summary += f"{details.get('content_snippet', '')}\n"
    
    prompt = """
    You are an expert in determining whether a user query is semantically relevant to a set of documents.
    
    User Query: {query}
    
    Document Summary:
    {doc_summary}
    
    Your task is to determine if the user query is seeking information contained in or related to the documents.
    Evaluate whether responding to this query requires using content from these documents.
    
    Consider these guidelines:
    1. If the query explicitly mentions BR IDs, document codes, or topics in the documents, it's relevant
    2. If the query is asking for information that would be found in these documents, it's relevant
    3. If the query is about a completely different domain or topic, it's not relevant
    4. If the query is a general question unrelated to the specific content in the documents, it's not relevant
    
    RESPOND WITH VALID JSON ONLY:
    {{
        "relevant": true/false,
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation of your decision"
    }}
    """
    
    relevance_template = ChatPromptTemplate.from_template(prompt)
    chain = relevance_template | llm | StrOutputParser()
    
    try:
        raw_response = chain.invoke({"query": query, "doc_summary": doc_summary})
        
        # Extract JSON response
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, raw_response)
        if match:
            json_str = match.group(1)
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # If still having trouble, try with a more aggressive cleanup
                cleaned_json = re.sub(r'```json|```|\n|\r', '', json_str)
                result = json.loads(cleaned_json)
                return result
        else:
            # Return a default response if we couldn't parse JSON
            return {
                "relevant": True,  # Default to True to be safe
                "confidence": 0.5,
                "reasoning": "Could not determine relevance, defaulting to using documents."
            }
            
    except Exception as e:
        print(f"Error checking document relevance: {e}")
        # Default to using documents in case of error
        return {
            "relevant": True,
            "confidence": 0.5,
            "reasoning": f"Error during relevance check: {str(e)}. Defaulting to using documents."
        }

# Function to check if input is a URL and process it
def process_url_input(query: str) -> Dict[str, Any]:
    """
    Check if the user input is a URL, and if so, extract content from the webpage.
    
    Args:
        query: The user query string
        
    Returns:
        dict: A dictionary with URL processing information
            - is_url (bool): Whether the input is a URL
            - url (str): The detected URL (if any)
            - content (str): The extracted content (if any)
            - title (str): The webpage title (if any)
            - success (bool): Whether the extraction was successful
            - error (str): Error message if extraction failed
    """
    result = {
        "is_url": False,
        "url": "",
        "content": "",
        "title": "",
        "success": False,
        "error": ""
    }
    
    # Simple URL regex pattern
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    
    # Check if the query contains a URL
    url_matches = re.findall(url_pattern, query)
    if not url_matches:
        return result
    
    # Extract the URL
    url = url_matches[0].strip()
    result["is_url"] = True
    result["url"] = url
    
    # Validate the URL
    if not validators.url(url):
        result["error"] = f"Invalid URL format: {url}"
        return result
    
    try:
        # Send request to the URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if request was successful
        if response.status_code != 200:
            result["error"] = f"Failed to fetch URL (status code: {response.status_code})"
            return result
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title = soup.title.string if soup.title else "Untitled webpage"
        result["title"] = title
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
        
        # Get the main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            # Get text and clean it
            text = main_content.get_text(separator='\n')
            # Clean text: remove multiple newlines and whitespace
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            result["content"] = text.strip()
        else:
            # Fallback to body text
            text = soup.get_text(separator='\n')
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            result["content"] = text.strip()
        
        result["success"] = True
        
        # Add the URL content to the document store if we extracted content successfully
        if result["content"]:
            # Generate a document ID for the URL
            url_doc_id = f"URL-{hashlib.md5(url.encode()).hexdigest()[:8]}"
            
            # Add to session state ID mappings
            if hasattr(st, 'session_state'):
                st.session_state.id_mappings[url_doc_id] = {
                    "type": "Web Content",
                    "filename": url,
                    "content_snippet": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"],
                    "full_content": result["content"],
                    "title": result["title"]
                }
                
                # Update the document store with the URL content
                update_document_store(
                    text=result["content"],
                    document_type="Web Content",
                    filename=f"{result['title']} ({url})"
                )
                
                # Do NOT parse BR IDs from URL content - only from user uploaded files
        
        return result
    
    except Exception as e:
        result["error"] = f"Error extracting content from URL: {str(e)}"
        return result

def extract_url_content(url):
    """
    Extract content from a given URL.
    
    Args:
        url (str): URL to extract content from
        
    Returns:
        dict: Dictionary containing the URL, title, and content
    """
    try:
        # Use a proper User-Agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make sure we're handling URL paths properly
        if not url.startswith('http'):
            url = 'https://' + url
        
        # Preserve the original URL with path for content extraction
        full_url = url
        
        # Handle FHIR-specific URLs that might require special handling
        if 'fhir' in url.lower() and '#' in url:
            # For FHIR spec URLs with fragments, ensure we get the full page
            url = url.split('#')[0]
        
        response = requests.get(full_url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the page title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract meaningful content
        # Remove script, style elements and comments
        for element in soup(['script', 'style']):
            element.decompose()
            
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Special handling for FHIR implementation guides
        if 'fhir' in url.lower():
            # For FHIR content, focus on the main content sections
            content_div = None
            # Try specific content divs used in FHIR implementation guides
            for div_id in ['content', 'main-content', 'page-content', 'container', 'wrapper']:
                content_div = soup.find('div', {'id': div_id})
                if content_div:
                    break
            
            # If no specific content div found, look for common FHIR page structures
            if not content_div:
                # Look for div with class 'content' or similar
                for div_class in ['content', 'main', 'main-content', 'container']:
                    content_div = soup.find('div', {'class': div_class})
                    if content_div:
                        break
            
            # If we found a content div, use that; otherwise use the whole body
            if content_div:
                content = content_div.get_text(separator='\n', strip=True)
            else:
                content = soup.body.get_text(separator='\n', strip=True) if soup.body else "No content found"
        else:
            # For non-FHIR content, use the whole body
            content = soup.body.get_text(separator='\n', strip=True) if soup.body else "No content found"
        
        # Clean up the content
        content = re.sub(r'\n+', '\n', content)  # Remove excessive newlines
        content = re.sub(r'\s+', ' ', content)   # Normalize whitespace
        content = content.strip()
        
        print(f"Successfully extracted content from URL: {full_url} (title: {title})")
        return {
            "url": full_url,
            "title": title,
            "content": content
        }
    except Exception as e:
        print(f"Error extracting content from URL {url}: {str(e)}")
        return {
            "url": url,
            "title": "Error extracting content",
            "content": f"Unable to extract content due to error: {str(e)}"
        }

