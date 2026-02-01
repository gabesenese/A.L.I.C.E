"""
Advanced Memory System for A.L.I.C.E
Features:
- Vector embeddings for semantic search
- RAG (Retrieval Augmented Generation) capabilities
- Long-term memory with ChromaDB/FAISS
- Episodic and semantic memory
- Knowledge graph integration
- Document ingestion (PDF, TXT, MD, DOCX, etc.)
"""

import os
import json
import logging
import pickle
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural", "document"
    timestamp: str
    context: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    source_file: Optional[str] = None  # Source document path
    chunk_index: Optional[int] = None  # For document chunks
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class DocumentChunk:
    """Represents a chunk of a larger document"""
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """Handles document ingestion and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html', '.json', '.csv'}
    
    def process_file(self, file_path: str) -> Tuple[str, List[DocumentChunk], Dict[str, Any]]:
        """
        Process a file and extract text content with chunks
        
        Returns:
            (full_text, chunks, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Extract text based on file type
        try:
            if extension == '.txt':
                text = self._extract_text_from_txt(file_path)
            elif extension == '.md':
                text = self._extract_text_from_markdown(file_path)
            elif extension == '.pdf':
                text = self._extract_text_from_pdf(file_path)
            elif extension == '.docx':
                text = self._extract_text_from_docx(file_path)
            elif extension == '.html':
                text = self._extract_text_from_html(file_path)
            elif extension == '.json':
                text = self._extract_text_from_json(file_path)
            elif extension == '.csv':
                text = self._extract_text_from_csv(file_path)
            else:
                text = self._extract_text_from_txt(file_path)  # Fallback
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
        
        # Create document chunks
        chunks = self._create_chunks(text)
        
        # Create metadata
        metadata = {
            'filename': file_path.name,
            'filepath': str(file_path),  # Ensure string for JSON serialization
            'extension': extension,
            'file_size': file_path.stat().st_size,
            'created_date': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'file_hash': self._calculate_file_hash(file_path),
            'total_chunks': len(chunks),
            'total_characters': len(text)
        }
        
        return text, chunks, metadata
    
    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_text_from_markdown(self, file_path: Path) -> str:
        """Extract text from markdown file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Simple markdown parsing - remove basic formatting
        import re
        # Remove headers but keep content
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # Remove emphasis
        content = re.sub(r'[*_]+([^*_]+)[*_]+', r'\1', content)
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content.strip()
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise ImportError("PyPDF2 required for PDF processing")
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            raise ImportError("python-docx required for DOCX processing")
    
    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML file"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=' ', strip=True)
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            # Fallback: simple HTML tag removal
            import re
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            content = re.sub(r'<[^>]+>', '', content)
            return content
    
    def _extract_text_from_json(self, file_path: Path) -> str:
        """Extract text from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_text_values(obj):
            """Recursively extract text values from JSON"""
            text_parts = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    else:
                        text_parts.extend(extract_text_values(value))
            elif isinstance(obj, list):
                for item in obj:
                    text_parts.extend(extract_text_values(item))
            elif isinstance(obj, str):
                text_parts.append(obj)
            
            return text_parts
        
        return "\n".join(extract_text_values(data))
    
    def _extract_text_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        except ImportError:
            # Fallback without pandas
            import csv
            text_parts = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    text_parts.append(" | ".join(row))
            return "\n".join(text_parts)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create overlapping chunks from text"""
        if len(text) <= self.chunk_size:
            return [DocumentChunk(
                content=text,
                chunk_index=0,
                start_position=0,
                end_position=len(text),
                metadata={"is_complete_document": True}
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the overlap area
                sentence_endings = ['.', '!', '?', '\n\n']
                best_break = end
                
                for i in range(max(end - self.chunk_overlap, start), min(end + self.chunk_overlap, len(text))):
                    if text[i] in sentence_endings and i < len(text) - 1:
                        best_break = i + 1
                        break
                
                end = best_break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_position=start,
                    end_position=end,
                    metadata={
                        "is_complete_document": False,
                        "total_document_length": len(text)
                    }
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                break
        
        return chunks


class VectorStore:
    """
    Simple vector store for semantic search
    Can be replaced with ChromaDB, FAISS, or Pinecone for production
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict):
        """Add vector to store"""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata.append(metadata)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors using cosine similarity
        
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        # Calculate cosine similarities
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        similarities = []
        for i, vec in enumerate(self.vectors):
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            similarity = np.dot(query_norm, vec_norm)
            similarities.append((self.ids[i], float(similarity), self.metadata[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def delete(self, id: str) -> bool:
        """Delete vector by ID"""
        if id in self.ids:
            index = self.ids.index(id)
            del self.ids[index]
            del self.vectors[index]
            del self.metadata[index]
            return True
        return False
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'dimension': self.dimension,
            'ids': self.ids,
            'vectors': [v.tolist() for v in self.vectors],
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Vector store saved ({len(self.ids)} vectors)")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.dimension = data['dimension']
            self.ids = data['ids']
            self.vectors = [np.array(v) for v in data['vectors']]
            self.metadata = data['metadata']
            logger.info(f"[OK] Vector store loaded ({len(self.ids)} vectors)")


class MemorySystem:
    """
    Advanced memory management system for A.L.I.C.E
    Implements episodic, semantic, and procedural memory with RAG
    Enhanced with document ingestion capabilities
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Memory stores
        self.episodic_memory: List[MemoryEntry] = []  # Conversations and events
        self.semantic_memory: List[MemoryEntry] = []  # Facts and knowledge
        self.procedural_memory: List[MemoryEntry] = []  # How-to knowledge
        self.document_memory: List[MemoryEntry] = []  # Ingested documents
        
        # Vector store for semantic search
        self.vector_store = VectorStore(dimension=384)
        
        # Document processor for ingestion
        self.document_processor = DocumentProcessor()
        
        # Document registry to track ingested files
        self.document_registry: Dict[str, Dict] = {}
        
        # Embedding function (lightweight - can use sentence-transformers for better results)
        self._embedding_model = None
        
        # Consolidation tracking
        self.turns_since_consolidation = 0
        self.consolidation_interval = 100  # Consolidate every N turns
        
        # Load existing memories
        self._load_memories()
        
        logger.info("[OK] Memory System initialized with document ingestion")
    
    def periodic_consolidation_check(self):
        """
        Check if periodic consolidation is needed
        Call this at the end of each conversation turn
        """
        self.turns_since_consolidation += 1
        
        if self.turns_since_consolidation >= self.consolidation_interval:
            logger.info(f"⏰ Running periodic memory consolidation (after {self.turns_since_consolidation} turns)")
            self.consolidate_memories(max_episodic=1000, auto_deduplicate=True)
            self.turns_since_consolidation = 0
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import os
                # Set longer timeout for model download
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                logger.info("Embedding model loaded (sentence-transformers)")
            except ImportError:
                logger.warning("[WARNING] sentence-transformers not installed. Using simple embeddings.")
                logger.warning("   Install with: pip install sentence-transformers")
                # Fallback to simple TF-IDF based embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._embedding_model = TfidfVectorizer(max_features=384)
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load sentence-transformers model: {e}")
                logger.warning("   Falling back to simple TF-IDF embeddings.")
                # Fallback to simple TF-IDF based embeddings
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self._embedding_model = TfidfVectorizer(max_features=384)
                except ImportError:
                    logger.error("sklearn not available either. Embeddings will be disabled.")
                    self._embedding_model = None
        
        return self._embedding_model
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding vector for text"""
        model = self._get_embedding_model()
        
        try:
            # Try sentence-transformers first
            if hasattr(model, 'encode'):
                embedding = model.encode(text, convert_to_numpy=True)
                return embedding
            else:
                # Fallback to TF-IDF
                embedding = model.transform([text]).toarray()[0]
                # Pad or truncate to 384 dimensions
                if len(embedding) < 384:
                    embedding = np.pad(embedding, (0, 384 - len(embedding)))
                else:
                    embedding = embedding[:384]
                return embedding
        except Exception as e:
            logger.error(f"[ERROR] Embedding error: {e}")
            # Return random embedding as last resort
            return np.random.rand(384)
    
    def store_memory(
        self, 
        content: str, 
        memory_type: str = "episodic",
        context: Optional[Dict] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, procedural)
            context: Additional context information
            importance: Importance score (0-1)
            tags: Tags for categorization
            
        Returns:
            Memory ID
        """
        # Generate unique ID
        memory_id = f"{memory_type}_{datetime.now().timestamp()}"
        
        # Create embedding
        embedding = self._create_embedding(content)
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            importance=importance,
            embedding=embedding.tolist(),
            tags=tags or []
        )
        
        # Store in appropriate memory
        if memory_type == "episodic":
            self.episodic_memory.append(memory)
        elif memory_type == "semantic":
            self.semantic_memory.append(memory)
        elif memory_type == "procedural":
            self.procedural_memory.append(memory)
        elif memory_type == "document":
            self.document_memory.append(memory)
        else:
            # Default to semantic for unknown types
            self.semantic_memory.append(memory)
        
        # Add to vector store for search
        self.vector_store.add(
            id=memory_id,
            vector=embedding,
            metadata={
                "content": content,
                "type": memory_type,
                "timestamp": memory.timestamp,
                "importance": importance,
                "tags": tags or []
            }
        )
        
        logger.info(f"Memory stored: {memory_type} - {content[:50]}...")
        return memory_id
    
    def recall_memory(
        self, 
        query: str, 
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recall memories using semantic search
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant memories with metadata
        """
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Filter by memory type and similarity
        filtered_results = []
        for mem_id, similarity, metadata in results:
            if similarity < min_similarity:
                continue
            
            if memory_type and metadata.get('type') != memory_type:
                continue
            
            # Get full memory entry
            memory = self._get_memory_by_id(mem_id)
            if memory:
                # Update access statistics
                memory.access_count += 1
                memory.last_accessed = datetime.now().isoformat()
                
                filtered_results.append({
                    "id": mem_id,
                    "content": metadata['content'],
                    "type": metadata['type'],
                    "similarity": similarity,
                    "importance": metadata['importance'],
                    "timestamp": metadata['timestamp'],
                    "access_count": memory.access_count,
                    "tags": metadata.get('tags', [])
                })
            
            if len(filtered_results) >= top_k:
                break
        
        logger.info(f"Recalled {len(filtered_results)} memories for: {query[:50]}...")
        return filtered_results
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory entry by ID"""
        for memory_list in [self.episodic_memory, self.semantic_memory, 
                           self.procedural_memory, self.document_memory]:
            for memory in memory_list:
                if memory.id == memory_id:
                    return memory
        return None
    
    def get_context_for_llm(self, query: str, max_memories: int = 3) -> str:
        """
        Get relevant context from memory for LLM (RAG)
        
        Args:
            query: User query
            max_memories: Maximum number of memories to include
            
        Returns:
            Formatted context string
        """
        # Recall relevant memories
        memories = self.recall_memory(query, top_k=max_memories)
        
        if not memories:
            return ""
        
        # Format as context
        context_parts = ["Relevant information from memory:"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(f"{i}. {mem['content']} (from {mem['timestamp'][:10]})")
        
        return "\n".join(context_parts)
    
    def ingest_document(self, file_path: str, importance: float = 0.7, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest a document into memory with chunking and embedding
        
        Args:
            file_path: Path to document file
            importance: Importance score for the document (0-1)
            tags: Tags for categorization
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Process the document
            full_text, chunks, metadata = self.document_processor.process_file(file_path)
            
            # Check if document was already ingested (by hash)
            file_hash = metadata['file_hash']
            if file_hash in self.document_registry:
                existing_doc = self.document_registry[file_hash]
                logger.info(f"Document already ingested: {metadata['filename']}")
                return {
                    "status": "already_exists",
                    "document_id": existing_doc['document_id'],
                    "chunks_created": existing_doc['chunks_created'],
                    "message": f"Document '{metadata['filename']}' was already ingested"
                }
            
            # Generate document ID
            doc_id = f"doc_{int(datetime.now().timestamp())}_{file_hash[:8]}"
            
            # Store chunks as individual memories
            chunk_ids = []
            for chunk in chunks:
                memory_id = self.store_memory(
                    content=chunk.content,
                    memory_type="document",
                    context={
                        "document_id": doc_id,
                        "chunk_index": chunk.chunk_index,
                        "start_position": chunk.start_position,
                        "end_position": chunk.end_position,
                        "document_metadata": metadata,
                        **chunk.metadata
                    },
                    importance=importance,
                    tags=(tags or []) + [f"document:{metadata['filename']}", f"filetype:{metadata['extension']}"]
                )
                chunk_ids.append(memory_id)
            
            # Register document
            self.document_registry[file_hash] = {
                "document_id": doc_id,
                "file_path": str(file_path),  # Convert Path to string for JSON serialization
                "metadata": metadata,
                "chunk_ids": chunk_ids,
                "chunks_created": len(chunks),
                "ingestion_timestamp": datetime.now().isoformat(),
                "importance": importance,
                "tags": tags or []
            }
            
            # Save updated registry and memories
            self._save_document_registry()
            self._save_memories()
            
            logger.info(f"Document ingested: {metadata['filename']} ({len(chunks)} chunks)")
            
            return {
                "status": "success",
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "total_characters": metadata['total_characters'],
                "file_hash": file_hash,
                "message": f"Successfully ingested '{metadata['filename']}' with {len(chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to ingest document: {e}"
            }
    
    def ingest_directory(self, directory_path: str, recursive: bool = True, 
                        file_patterns: Optional[List[str]] = None, 
                        importance: float = 0.7) -> Dict[str, Any]:
        """
        Ingest all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            file_patterns: List of file patterns to match (e.g., ['*.pdf', '*.txt'])
            importance: Default importance for all documents
            
        Returns:
            Summary of ingestion results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            return {
                "status": "error",
                "message": f"Directory not found: {directory_path}"
            }
        
        results = {
            "status": "success",
            "total_files": 0,
            "successful": 0,
            "already_existed": 0,
            "failed": 0,
            "details": []
        }
        
        # Find all supported files
        if recursive:
            files = directory_path.rglob("*")
        else:
            files = directory_path.glob("*")
        
        supported_files = []
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.document_processor.supported_extensions:
                # Apply file patterns if specified
                if file_patterns:
                    if not any(file_path.match(pattern) for pattern in file_patterns):
                        continue
                supported_files.append(file_path)
        
        results["total_files"] = len(supported_files)
        
        # Ingest each file
        for file_path in supported_files:
            try:
                result = self.ingest_document(
                    str(file_path), 
                    importance=importance,
                    tags=[f"directory:{directory_path.name}"]
                )
                
                if result["status"] == "success":
                    results["successful"] += 1
                elif result["status"] == "already_exists":
                    results["already_existed"] += 1
                else:
                    results["failed"] += 1
                    
                results["details"].append({
                    "file": str(file_path),
                    "status": result["status"],
                    "message": result.get("message", "")
                })
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "file": str(file_path),
                    "status": "error",
                    "message": str(e)
                })
        
        logger.info(f"Directory ingestion complete: {results['successful']}/{results['total_files']} files")
        return results
    
    def search_documents(self, query: str, top_k: int = 5, min_similarity: float = 0.6) -> List[Dict]:
        """
        Search specifically in ingested documents
        
        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant document chunks
        """
        return self.recall_memory(
            query=query,
            memory_type="document",
            top_k=top_k,
            min_similarity=min_similarity
        )
    
    def get_document_info(self, document_id: str = None, file_path: str = None) -> Optional[Dict]:
        """
        Get information about an ingested document
        
        Args:
            document_id: Document ID
            file_path: Original file path
            
        Returns:
            Document information or None if not found
        """
        for file_hash, doc_info in self.document_registry.items():
            if (document_id and doc_info["document_id"] == document_id) or \
               (file_path and doc_info["file_path"] == file_path):
                return doc_info
        return None
    
    def list_ingested_documents(self) -> List[Dict]:
        """
        List all ingested documents
        
        Returns:
            List of document information
        """
        documents = []
        for file_hash, doc_info in self.document_registry.items():
            # Get file type from extension
            extension = doc_info["metadata"]["extension"]
            file_type = extension.upper() if extension else "UNKNOWN"
            
            # Calculate size in MB
            size_bytes = doc_info["metadata"]["file_size"]
            size_mb = size_bytes / (1024 * 1024)
            
            documents.append({
                "document_id": doc_info["document_id"],
                "filename": doc_info["metadata"]["filename"],
                "file_path": doc_info["file_path"],
                "file_type": file_type,
                "file_size": doc_info["metadata"]["file_size"],
                "size_mb": size_mb,
                "chunks": doc_info["chunks_created"],
                "ingestion_date": doc_info["ingestion_timestamp"][:10],
                "ingested_at": doc_info["ingestion_timestamp"][:19].replace('T', ' '),
                "tags": doc_info["tags"]
            })
        return sorted(documents, key=lambda x: x["ingestion_date"], reverse=True)
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove an ingested document and all its chunks
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            True if removed successfully
        """
        # Find document in registry
        doc_info = None
        file_hash_to_remove = None
        for file_hash, info in self.document_registry.items():
            if info["document_id"] == document_id:
                doc_info = info
                file_hash_to_remove = file_hash
                break
        
        if not doc_info:
            return False
        
        # Remove all chunk memories
        removed_count = 0
        for chunk_id in doc_info["chunk_ids"]:
            if self._remove_memory_by_id(chunk_id):
                removed_count += 1
        
        # Remove from registry
        del self.document_registry[file_hash_to_remove]
        self._save_document_registry()
        
        logger.info(f"Removed document {document_id} ({removed_count} chunks)")
        return True
    
    def _remove_memory_by_id(self, memory_id: str) -> bool:
        """Remove memory by ID from all memory stores"""
        for memory_list in [self.episodic_memory, self.semantic_memory, self.procedural_memory, self.document_memory]:
            for i, memory in enumerate(memory_list):
                if memory.id == memory_id:
                    del memory_list[i]
                    self.vector_store.delete(memory_id)
                    return True
        return False
    
    def _save_document_registry(self):
        """Save document registry to disk"""
        registry_file = os.path.join(self.data_dir, "document_registry.json")
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.document_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document registry: {e}")
    
    def _load_document_registry(self):
        """Load document registry from disk"""
        registry_file = os.path.join(self.data_dir, "document_registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    self.document_registry = json.load(f)
                logger.info(f"Document registry loaded ({len(self.document_registry)} documents)")
            except Exception as e:
                logger.error(f"Error loading document registry: {e}")
                self.document_registry = {}
    
    def calculate_memory_importance(self, memory: MemoryEntry) -> float:
        """
        Calculate dynamic importance score for a memory
        
        Factors:
        1. Access frequency (0-0.3)
        2. Recency (0-0.3)
        3. Base importance (0-0.4)
        
        Returns:
            Importance score (0-1)
        """
        # Access frequency score (more accessed = more important)
        # Cap at 20 accesses for scoring
        access_score = min(memory.access_count / 20.0, 1.0) * 0.3
        
        # Recency score (newer = more important)
        try:
            mem_time = datetime.fromisoformat(memory.timestamp)
            age_days = (datetime.now() - mem_time).days
            # Decay over 30 days
            recency_score = max(0, (30 - age_days) / 30.0) * 0.3
        except:
            recency_score = 0.0
        
        # Base importance (set at creation)
        base_score = memory.importance * 0.4
        
        return min(access_score + recency_score + base_score, 1.0)
    
    def deduplicate_memories(self, similarity_threshold: float = 0.95):
        """
        Remove duplicate memories based on semantic similarity
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicates (0-1)
        """
        removed_count = 0
        
        for memory_list_name in ['episodic_memory', 'semantic_memory']:
            memory_list = getattr(self, memory_list_name)
            
            if len(memory_list) < 2:
                continue
            
            # Build similarity matrix
            to_remove = set()
            
            for i in range(len(memory_list)):
                if i in to_remove:
                    continue
                    
                for j in range(i + 1, len(memory_list)):
                    if j in to_remove:
                        continue
                    
                    # Compare embeddings if available
                    if memory_list[i].embedding and memory_list[j].embedding:
                        similarity = self._cosine_similarity(
                            memory_list[i].embedding,
                            memory_list[j].embedding
                        )
                        
                        if similarity >= similarity_threshold:
                            # Keep the one with higher importance
                            importance_i = self.calculate_memory_importance(memory_list[i])
                            importance_j = self.calculate_memory_importance(memory_list[j])
                            
                            if importance_i >= importance_j:
                                to_remove.add(j)
                                # Merge access counts
                                memory_list[i].access_count += memory_list[j].access_count
                            else:
                                to_remove.add(i)
                                memory_list[j].access_count += memory_list[i].access_count
                                break  # Move to next i
            
            # Remove duplicates
            if to_remove:
                indices_to_keep = [i for i in range(len(memory_list)) if i not in to_remove]
                new_list = [memory_list[i] for i in indices_to_keep]
                setattr(self, memory_list_name, new_list)
                removed_count += len(to_remove)
        
        if removed_count > 0:
            logger.info(f"🔄 Deduplicated memories: removed {removed_count} duplicates")
            self._save_memories()
        
        return removed_count
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def consolidate_memories(self, max_episodic: int = 1000, auto_deduplicate: bool = True):
        """
        Consolidate old episodic memories with importance scoring
        
        Args:
            max_episodic: Maximum episodic memories to keep
            auto_deduplicate: Automatically remove duplicates first
        
        Process:
        1. Deduplicate memories (optional)
        2. Recalculate importance scores
        3. Keep top N by importance
        4. Archive rest
        """
        if auto_deduplicate:
            self.deduplicate_memories()
        
        if len(self.episodic_memory) <= max_episodic:
            logger.info(f"[OK] Memory consolidation not needed ({len(self.episodic_memory)} < {max_episodic})")
            return
        
        # Recalculate importance for all episodic memories
        for memory in self.episodic_memory:
            memory.importance = self.calculate_memory_importance(memory)
        
        # Sort by importance
        self.episodic_memory.sort(
            key=lambda m: m.importance,
            reverse=True
        )
        
        # Keep top memories
        archived = self.episodic_memory[max_episodic:]
        self.episodic_memory = self.episodic_memory[:max_episodic]
        
        # Save archived memories to separate file
        self._save_archived_memories(archived)
        
        logger.info(f"[OK] Consolidated memories: kept {max_episodic} (avg importance: {sum(m.importance for m in self.episodic_memory)/len(self.episodic_memory):.2f}), archived {len(archived)}")
        
        # Save updated memories
        self._save_memories()
    
    def _save_archived_memories(self, archived: List[MemoryEntry]):
        """Save archived memories to separate file"""
        try:
            archive_file = os.path.join(self.data_dir, "archived_memories.json")
            
            # Load existing archive if exists
            existing_archive = []
            if os.path.exists(archive_file):
                try:
                    with open(archive_file, 'r', encoding='utf-8') as f:
                        existing_archive = json.load(f)
                except:
                    pass
            
            # Append new archived memories
            new_archive = existing_archive + [asdict(m) for m in archived]
            
            # Save
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(new_archive, f, indent=2)
            
            logger.info(f"[OK] Archived {len(archived)} memories to {archive_file}")
        except Exception as e:
            logger.error(f"Error archiving memories: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "total_memories": len(self.episodic_memory) + len(self.semantic_memory) + len(self.procedural_memory),
            "vector_count": len(self.vector_store.ids)
        }
    
    def _save_memories(self):
        """Save all memories to disk"""
        try:
            # Save memory lists
            memories_data = {
                "episodic": [asdict(m) for m in self.episodic_memory],
                "semantic": [asdict(m) for m in self.semantic_memory],
                "procedural": [asdict(m) for m in self.procedural_memory],
                "document": [asdict(m) for m in self.document_memory]
            }
            
            with open(os.path.join(self.data_dir, "memories.json"), 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2)
            
            # Save vector store
            self.vector_store.save(os.path.join(self.data_dir, "vectors.pkl"))
            
            # Save document registry
            self._save_document_registry()
            
            total_memories = len(self.episodic_memory) + len(self.semantic_memory) + len(self.procedural_memory) + len(self.document_memory)
            logger.info(f"All {total_memories} memories saved")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving memories: {e}")
    
    def _load_memories(self):
        """Load memories from disk"""
        try:
            memories_path = os.path.join(self.data_dir, "memories.json")
            if os.path.exists(memories_path):
                with open(memories_path, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                # Handle both dict and list formats (backward compatibility)
                if isinstance(memories_data, dict):
                    self.episodic_memory = [MemoryEntry(**m) for m in memories_data.get("episodic", [])]
                    self.semantic_memory = [MemoryEntry(**m) for m in memories_data.get("semantic", [])]
                    self.procedural_memory = [MemoryEntry(**m) for m in memories_data.get("procedural", [])]
                    self.document_memory = [MemoryEntry(**m) for m in memories_data.get("document", [])]
                elif isinstance(memories_data, list):
                    # Old format: just a list of memories, treat as episodic
                    self.episodic_memory = [MemoryEntry(**m) if isinstance(m, dict) else m for m in memories_data]
                    logger.info("[Memory] Loaded legacy format, converted to new structure")
                else:
                    logger.warning(f"[Memory] Unknown memories format: {type(memories_data)}")
                
                total_memories = len(self.episodic_memory) + len(self.semantic_memory) + len(self.procedural_memory) + len(self.document_memory)
                logger.info(f"[OK] Loaded {total_memories} memories "
                          f"({len(self.episodic_memory)} episodic, "
                          f"{len(self.semantic_memory)} semantic, "
                          f"{len(self.procedural_memory)} procedural, "
                          f"{len(self.document_memory)} document)")
            
            # Load vector store
            vectors_path = os.path.join(self.data_dir, "vectors.pkl")
            if os.path.exists(vectors_path):
                self.vector_store.load(vectors_path)
                
            # Load document registry
            self._load_document_registry()
                
        except Exception as e:
            logger.warning(f"[WARNING] Could not load memories: {e}")
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save on exit"""
        self._save_memories()


# Example usage
if __name__ == "__main__":
    print("Testing Memory System...\n")
    
    with MemorySystem() as memory:
        # Store different types of memories
        print("Storing memories...")
        
        # Episodic memories (conversations/events)
        memory.store_memory(
            "User asked about the weather in New York on January 24, 2026",
            memory_type="episodic",
            context={"location": "New York", "topic": "weather"},
            importance=0.6,
            tags=["weather", "conversation"]
        )
        
        memory.store_memory(
            "User mentioned they like Python programming",
            memory_type="episodic",
            context={"topic": "preferences"},
            importance=0.8,
            tags=["preferences", "programming"]
        )
        
        # Semantic memories (facts)
        memory.store_memory(
            "Python is a high-level programming language known for readability",
            memory_type="semantic",
            importance=0.7,
            tags=["python", "programming", "facts"]
        )
        
        memory.store_memory(
            "The Eiffel Tower is located in Paris, France",
            memory_type="semantic",
            importance=0.5,
            tags=["facts", "geography"]
        )
        
        # Procedural memory (how-to)
        memory.store_memory(
            "To create a Python virtual environment, use: python -m venv env",
            memory_type="procedural",
            importance=0.9,
            tags=["python", "tutorial", "setup"]
        )
        
        print("[OK] Stored memories\n")
        
        # Recall memories
        print("🔍 Testing memory recall...")
        
        queries = [
            "Tell me about Python",
            "What do I like?",
            "How to setup Python environment?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = memory.recall_memory(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['type']}] {result['content']}")
                print(f"     Similarity: {result['similarity']:.3f}, "
                      f"Importance: {result['importance']}, "
                      f"Accessed: {result['access_count']} times")
        
        # Get RAG context
        print(f"\nRAG Context for 'Python programming':")
        context = memory.get_context_for_llm("Python programming", max_memories=3)
        print(context)
        
        # Statistics
        print(f"\n Memory Statistics:")
        stats = memory.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    print("\n[OK] Memories saved automatically on exit")
