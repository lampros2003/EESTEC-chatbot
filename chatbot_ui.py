import streamlit as st
import os
import re
import tempfile
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

class ChatBot:
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):
        self.model_name = model_name
        # Load tokenizer and model
        try:
            print(f"Loading model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            # Fall back to a smaller model
            fallback_model = "gpt2"  # Much smaller model as fallback
            print(f"Falling back to smaller model: {fallback_model}")
            self.model_name = fallback_model
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model, 
                device_map="auto",
                low_cpu_mem_usage=True
            )
    def retrieve_context(self, chunks):
        return "  ".join(chunks)

    def build_prompt(self, context: str, question: str) -> str:
        return (
            "Answer the following question based only on the context provided.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Answer in one very small sentence with less than 20 words."
        )

    def generate_answer(self, question: str, chunks=None) -> str:
        context = self.retrieve_context(chunks or [])
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip off prompt itself if echoed
        return answer.replace(prompt, "").strip()

    def chat(self):
        while True:
            try:
                user_q = input("\nYou: ").strip()
                if user_q.lower() in ("exit", "quit"):
                    print("👋 Goodbye!")
                    break
                # Example static context; replace with dynamic retrieval if needed
                context_chunks = [
                    "The quantum computer is like a super-smart magic box that can solve hard puzzles quickly.",
                    "It uses qubits that can be in many states at once, like spinning coins.",
                ]
                answer = self.generate_answer(user_q, chunks=context_chunks)
                print("Bot:", answer)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


class RagSystem:
    """Retrieval-Augmented Generation system for PDF documents."""

    def __init__(self, uploaded_files=None, pdf_directory="knowledge"):
        """
        Initialize the RAG system.
        Args:
            uploaded_files: List of Streamlit uploaded files (optional)
            pdf_directory: Directory containing PDF files (default: 'knowledge')
        """
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.collection_name = "pdf_knowledge_base"
        self.max_results = 10
        self.pdf_directory = pdf_directory

        # Keyword weights for hybrid search

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB
        self.collection = self.initialize_chroma()

        # Then process documents if needed
        if uploaded_files:
            self.process_uploaded_files(uploaded_files)
        elif (
            pdf_directory and not self.collection.count()
        ):  # Only process if collection is empty
            if os.path.exists(pdf_directory):
                pdf_files = [
                    f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")
                ]
                if pdf_files:
                    self.process_directory(pdf_directory)
                else:
                    print(f"No PDF files found in {pdf_directory}")
            else:
                os.makedirs(pdf_directory)
                # print(f"Created empty {pdf_directory} folder")

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            if hasattr(self, "collection"):
                # Get all IDs in the collection
                results = self.collection.get()
                if results and "ids" in results:
                    # Delete all documents
                    self.collection.delete(ids=results["ids"])
                return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def initialize_chroma(self):
        """Initialize ChromaDB and create collection from knowledge folder if needed."""
        client = chromadb.PersistentClient(path="./chromadb")

        # Get all collection names
        collection_names = [col.name for col in client.list_collections()]
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        if self.collection_name in collection_names:
            # print(f"Loading existing knowledge base '{self.collection_name}'...")
            return client.get_collection(
                name=self.collection_name, embedding_function=embedding_function
            )
        else:
            # print(f"Creating new knowledge base '{self.collection_name}'...")
            return client.create_collection(
                name=self.collection_name, embedding_function=embedding_function
            )

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file."""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_by_page = {}

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_by_page[page_num + 1] = (
                            text  # Store with 1-indexed page number
                        )

                return text_by_page
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return {}

    def extract_text_from_uploaded_pdf(self, uploaded_file):
        """Extract text from an uploaded PDF file."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Extract text from the temporary file
            text_by_page = self.extract_text_from_pdf(tmp_path)

            # Clean up
            os.unlink(tmp_path)

            return text_by_page
        except Exception as e:
            print(f"Error extracting text from uploaded file {uploaded_file.name}: {e}")
            return {}

    def chunk_text_by_paragraph_and_length(self, text):
        """Split text into chunks by paragraph and length."""
        paragraphs = text.split("\n\n")
        combined = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n", " ", ""],
            )
            chunks = splitter.split_text(para)
            combined.extend(chunks)

        return combined

    def process_directory(self, directory):
        """Process PDF files from a directory."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        pdf_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".pdf")
        ]

        pdf_texts = {}
        for pdf_file in pdf_files:
            text_by_page = self.extract_text_from_pdf(pdf_file)
            if text_by_page:
                pdf_texts[os.path.basename(pdf_file)] = text_by_page

        self.build_knowledge_base(pdf_texts)

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files."""
        pdf_texts = {}
        for uploaded_file in uploaded_files:
            text_by_page = self.extract_text_from_uploaded_pdf(uploaded_file)
            if text_by_page:
                pdf_texts[uploaded_file.name] = text_by_page

        self.build_knowledge_base(pdf_texts)

    def get_weighted_keywords(self, text):
        """Returns a dictionary of words with their importance weights"""
        # Process text with spaCy

        doc = nlp(text)

        # Part 1: POS-based filtering and initial weighting
        pos_weights = {"NOUN": 0.4, "PROPN": 0.4, "VERB": 0.3, "ADJ": 0.2, "ADV": 0.1}

        # Part 2: TF-IDF weighting
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform([text])
        tfidf_scores = dict(
            zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])
        )

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

    def build_knowledge_base(self, pdf_texts):
        """Create vector store from PDF texts."""
        all_chunks = []
        all_metadatas = []
        all_ids = []

        chunk_counter = 0

        for pdf_name, pages in pdf_texts.items():
            for page_num, page_text in pages.items():
                chunks = self.chunk_text_by_paragraph_and_length(page_text)

                for chunk in chunks:
                    chunk_id = f"chunk_{chunk_counter}"
                    all_chunks.append(chunk)
                    all_metadatas.append({"source": pdf_name, "page": page_num})
                    all_ids.append(chunk_id)
                    chunk_counter += 1

        # Add chunks to ChromaDB collection if there are any
        if all_chunks:
            self.collection.add(
                ids=all_ids, documents=all_chunks, metadatas=all_metadatas
            )

    def keyword_score(self, text):
        """Calculate keyword-based relevance score."""
        keyword_weights = self.get_weighted_keywords(text)
        if keyword_weights is None:
            return 0.0

        score = 0.0
        for keyword, weight in keyword_weights.items():
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        return score

    def hybrid_search(self, query, n_results=5):
        """Perform hybrid search combining vector similarity and keyword matching."""
        # Vector search
        results = self.collection.query(
            query_texts=[query], n_results=n_results * 2
        )  # Get more results initially
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]

        # Score by keywords
        scored = [
            {"chunk": doc, "meta": meta, "score": self.keyword_score(doc), "id": _id}
            for doc, meta, _id in zip(documents, metadatas, ids)
        ]

        # Sort by keyword score
        scored.sort(key=lambda x: -x["score"])
        top = scored[:n_results]

        return [r["chunk"] for r in top], [r["meta"] for r in top]


def main():
    st.set_page_config(page_title="PDF RAG Explorer", layout="wide")

    # Initialize session state
    if "rag" not in st.session_state:
        st.session_state.rag = RagSystem()
    if "ChatBot" not in st.session_state:
        try:
            with st.spinner("Loading AI model... This may take a moment..."):
                st.session_state.ChatBot = ChatBot()
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.session_state.ChatBot = None
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False

    st.title("🔍 PDF smart search")

    # Sidebar for file upload and knowledge base management
    with st.sidebar:
        st.header("Knowledge Base Management")
        kb_exists = os.path.exists("./chromadb") and len(os.listdir("./chromadb")) > 0

        if kb_exists:
            st.success("Persistent knowledge base found.")
            if st.button("Load Existing Knowledge Base"):
                try:
                    st.session_state.rag = RagSystem()
                    st.session_state.docs_processed = True
                    st.success("Knowledge base loaded.")
                except Exception as e:
                    st.session_state.docs_processed = False
                    st.error(f"Error loading knowledge base: {str(e)}")
        else:
            st.info(
                "No persistent knowledge base found. Please upload PDF files to create one."
            )

        st.markdown("---")
        st.header("Upload New Files")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Choose up to 5 PDF files for processing and creating a new knowledge base.",
        )

        if uploaded_files and len(uploaded_files) > 0:
            if (
                st.button("Process & Create New Knowledge Base")
                and len(uploaded_files) <= 5
            ):
                with st.spinner(
                    "Processing documents and creating knowledge base, please wait..."
                ):
                    try:
                        # First cleanup existing RAG system
                        if (
                            hasattr(st.session_state, "rag")
                            and st.session_state.rag is not None
                        ):
                            # Clear existing collection instead of removing it
                            st.session_state.rag.clear_collection()
                            # Process new files with existing RAG system
                            st.session_state.rag.process_uploaded_files(uploaded_files)
                        else:
                            # Create new RAG system if none exists
                            st.session_state.rag = RagSystem(
                                uploaded_files=uploaded_files
                            )

                        st.session_state.docs_processed = True
                        st.success(
                            "Knowledge base updated with new documents successfully!"
                        )
                    except Exception as e:
                        st.session_state.docs_processed = False
                        st.error(f"Error: {str(e)}")

    # If documents are processed or knowledge base loaded, show search UI
    if st.session_state.docs_processed and st.session_state.rag is not None:
        query = st.text_input(
            "Enter your query",
            placeholder="Type your question here...",
            label_visibility="visible",
        )

        if st.button("Search") and query:
            with st.spinner("Searching..."):
                try:
                    chunks, metadatas = st.session_state.rag.hybrid_search(query)
                    if st.session_state.ChatBot is not None:
                        main_answer = st.session_state.ChatBot.generate_answer(query, chunks)
                    else:
                        main_answer = "Model could not be loaded. Showing search results only."

                    # First show the generated answer
                    st.subheader("💡 Generated Answer")
                    answer_container = st.container()
                    with answer_container:
                        st.text_area(
                            "AI-generated summary answer",
                            value=main_answer,
                            height=100,  # Reduced height for better visibility
                            key="main_answer_box",
                        )

                        if st.button("📋 Copy Answer"):
                            st.session_state.copied = True
                            st.code(main_answer, language="text")
                            st.success("Answer copied to clipboard!")

                    st.markdown("---")  # Visual separator

                    # Then show the search results
                    st.subheader("📌 Search results")
                    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                        source = meta.get("source", "Unknown")
                        page = meta.get("page", "-")

                        with st.expander(f"📄 Result {i+1}"):
                            col1, col2 = st.columns([1, 4])

                            with col1:
                                st.markdown("**Source Info:**")

                                # Check if file exists in knowledge directory
                                pdf_path = os.path.join("knowledge", source)
                                if os.path.exists(pdf_path):
                                    # Create download button for the PDF with enhanced unique key
                                    with open(pdf_path, "rb") as pdf_file:
                                        btn_key = f"download_btn_{i}_{source}_{page}"  # Add page number to make key more unique
                                        st.download_button(
                                            label="📚 Download PDF",
                                            data=pdf_file,
                                            file_name=source,
                                            mime="application/pdf",
                                            key=btn_key,
                                        )
                                else:
                                    st.markdown(f"📚 **File:** {source}")

                                st.markdown(f"📑 **Page:** {page}")

                            with col2:
                                st.markdown("**Extract:**")
                                st.markdown(f"{chunk}", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

    else:
        st.info("Please load or create a knowledge base to start searching.")


if __name__ == "__main__":
    main()
