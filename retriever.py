import os
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Configuration
PDF_DIRECTORY = "knowledge"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_knowledge_base"
MAX_RESULTS = 10

def get_weighted_keywords(text):
    doc = nlp(text)
    
    # Part 1: POS-based filtering and initial weighting
    pos_weights = {
        'NOUN': 0.4,
        'PROPN': 0.4, 
        'VERB': 0.3,
        'ADJ': 0.2,
        'ADV': 0.1
    }
    
    # Part 2: TF-IDF weighting
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([text])
    tfidf_scores = dict(zip(tfidf.get_feature_names_out(), 
                          tfidf_matrix.toarray()[0]))
    
    # Combine weights
    word_weights = defaultdict(float)
    
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
            
        lemma = token.lemma_.lower()
        
        # Add POS weight
        if token.pos_ in pos_weights:
            word_weights[lemma] += pos_weights[token.pos_]
            
        # Add TF-IDF weight (if word appears in TF-IDF features)
        if lemma in tfidf_scores:
            word_weights[lemma] += tfidf_scores[lemma]
            
        # Bonus: Entity recognition boost
        if token.ent_type_:
            word_weights[lemma] += 0.15
    
    # Normalize weights to 0-1 range
    if word_weights:
        max_weight = max(word_weights.values())
        for word in word_weights:
            word_weights[word] /= max_weight
    
    return dict(word_weights)



def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def chunk_text_by_paragraph_and_length(text, chunk_size=1000, chunk_overlap=100):
    """Split text into chunks by paragraph and length."""
    paragraphs = text.split("\n\n")
    combined = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n", " ", ""]
        )
        chunks = splitter.split_text(para)
        combined.extend(chunks)
    
    return combined

def keyword_score(text, keyword_weights):
    """Calculate keyword-based relevance score."""
    score = 0.0
    keyword_weights = get_weighted_keywords(text)
    for keyword, weight in keyword_weights.items():
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
    return score

def hybrid_search(collection, query, keyword_weights, n_results=5):
    """Perform hybrid search combining vector similarity and keyword matching."""
    # Vector search
    results = collection.query(query_texts=[query], n_results=n_results*2)  # Get more results initially
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    
    # Score by keywords
    scored = [
        {
            "doc": doc,
            "meta": meta,
            "score": keyword_score(doc, keyword_weights),
            "id": _id
        }
        for doc, meta, _id in zip(documents, metadatas, ids)
    ]
    
    # Sort by score (keyword match count with weight)
    scored.sort(key=lambda x: (-x["score"]))  # Descending order by score
    top = scored[:n_results]
    
    return [r["doc"] for r in top], [r["meta"] for r in top]  # Get more results initially
    
    
    
    #return results["documents"][0],results["metadatas"][0]

def initialize_chroma():
    """Initialize ChromaDB and create collection."""
    client = chromadb.Client()
    
    # Delete collection if it already exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Collection {COLLECTION_NAME} does not exist or could not be deleted: {e}")  
    
    # Create new collection with the embedding function
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    
    return collection

def process_pdfs(directory, limit=None):
    """Process PDF files and return extracted texts."""
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    if limit:
        pdf_files = pdf_files[:limit]
        print(f"Processing {len(pdf_files)} PDF files (limited by configuration)")
    else:
        print(f"Processing {len(pdf_files)} PDF files")
    
    pdf_texts = {}
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            pdf_texts[pdf_file] = text
            print(f"Successfully extracted text from {pdf_file}")
        else:
            print(f"No text extracted from {pdf_file}")
    
    return pdf_texts

def build_knowledge_base(pdf_texts, collection):
    """Create vector store from PDF texts."""
    all_chunks = []
    chunk_to_source = {}
    
    for pdf_path, text in pdf_texts.items():
        chunks = chunk_text_by_paragraph_and_length(text)
        print(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        
        for chunk in chunks:
            chunk_id = f"chunk_{len(all_chunks)}"
            all_chunks.append(chunk)
            chunk_to_source[chunk_id] = pdf_path
    
    # Add chunks to ChromaDB collection
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": chunk_to_source[f"chunk_{i}"]} for i in range(len(all_chunks))]
    
    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=metadatas
    )
    
    print(f"Added {len(all_chunks)} chunks to the vector store")
    return all_chunks, chunk_to_source

def main():
    # Initialize ChromaDB
    collection = initialize_chroma()
    
    # Process PDFs - remove the limit for processing all files
    pdf_texts = process_pdfs(PDF_DIRECTORY, limit=None)
    
    # Build knowledge base
    all_chunks, chunk_to_source = build_knowledge_base(pdf_texts, collection)
    
    # Example query
    query = "How did Baker Hughes engineers use MATLAB to develop pump health monitoring software?"
    print(f"\nSearching for: '{query}'")
    results, metadata = hybrid_search(collection, query, keyword_weights=get_weighted_keywords(query), n_results=MAX_RESULTS)
    
    # Print results
    print("\nTop results:")
    for i, (doc, meta) in enumerate(zip(results, metadata)):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {os.path.basename(meta['source'])}")
        print(f"Text snippet: {doc[:400]}")



if __name__ == "__main__":
    main()