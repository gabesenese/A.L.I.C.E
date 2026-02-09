"""
Document Ingestion Plugin for A.L.I.C.E
Handles document ingestion, searching, and management
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from ai.plugins.plugin_system import PluginInterface
from ai.memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class DocumentPlugin(PluginInterface):
    """Plugin for handling document ingestion and search operations"""
    
    def __init__(self):
        super().__init__()
        self.name = "DocumentPlugin"
        self.version = "1.0.0"
        self.description = "Document ingestion and search capabilities"
        self.capabilities = [
            "ingest documents", "search documents", "list documents",
            "process PDF files", "process text files", "RAG capabilities"
        ]
        self.memory_system = None
    
    def initialize(self) -> bool:
        """Initialize the document plugin with memory system"""
        try:
            self.memory_system = MemorySystem()
            logger.info("Document Plugin initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Document Plugin: {e}")
            return False
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if this plugin can handle document-related requests"""
        document_intents = ["file_operations", "search", "query", "information"]
        document_keywords = [
            "ingest", "document", "file", "pdf", "text", "markdown",
            "process", "upload", "analyze", "search documents", 
            "find in documents", "document search", "add document"
        ]
        
        # Check if intent is document-related
        if intent in document_intents:
            # Check for document-related keywords in entities or intent
            query_text = entities.get('query', '').lower()
            if any(keyword in query_text for keyword in document_keywords):
                return True
        
        return False
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """Execute document operations"""
        try:
            query_lower = query.lower()
            
            # Document ingestion commands
            if any(keyword in query_lower for keyword in ["ingest", "add document", "process document", "upload document"]):
                return self._handle_document_ingestion(query, entities)
            
            # Document search commands
            elif any(keyword in query_lower for keyword in ["search documents", "find in documents", "search files"]):
                return self._handle_document_search(query, entities)
            
            # List documents command
            elif any(keyword in query_lower for keyword in ["list documents", "show documents", "what documents"]):
                return self._handle_list_documents()
            
            # Document-related question that might need RAG
            elif self._is_document_question(query_lower):
                return self._handle_document_question(query, entities)
            
            else:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "unknown_request",
                        "message_code": "docs:help"
                    }
                }
                
        except Exception as e:
            logger.error(f"Document plugin execution error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "docs:execute_error"
                }
            }
    
    def _handle_document_ingestion(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle document ingestion requests"""
        try:
            # Extract file path from query
            file_path = self._extract_file_path(query)
            
            if not file_path:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_path",
                        "message_code": "docs:need_path"
                    }
                }
            
            path = Path(file_path)
            
            # Check if it's a file or directory
            if path.is_file():
                success = self.memory_system.ingest_document(path)
                if success:
                    return {
                        "success": True,
                        "response": None,
                        "data": {
                            "file": path.name,
                            "message_code": "docs:ingest_file_success"
                        }
                    }
                else:
                    return {
                        "success": False,
                        "response": None,
                        "data": {
                            "file": path.name,
                            "error": "ingest_failed",
                            "message_code": "docs:ingest_file_failed"
                        }
                    }
                    
            elif path.is_dir():
                count = self.memory_system.ingest_directory(path)
                return {
                    "success": True,
                    "response": None,
                    "data": {
                        "directory": path.name,
                        "count": count,
                        "message_code": "docs:ingest_dir_success"
                    }
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "path": file_path,
                        "error": "path_not_found",
                        "message_code": "docs:path_not_found"
                    }
                }
                
        except Exception as e:
            logger.error(f"Document ingestion error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "docs:ingest_error"
                }
            }
    
    def _handle_document_search(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle document search requests"""
        try:
            # Extract search query
            search_terms = self._extract_search_terms(query)
            
            if not search_terms:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_query",
                        "message_code": "docs:need_search_terms"
                    }
                }
            
            # Perform document search
            results = self.memory_system.search_documents(search_terms, top_k=5)
            
            if not results:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "query": search_terms,
                        "error": "no_results",
                        "message_code": "docs:search_no_results"
                    }
                }
            
            # Format results into structured data for LLM
            formatted_results = []
            for i, result in enumerate(results, 1):
                # Get source file and chunk info from memory entry ID
                memory_id = result.get('id', '')
                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                similarity = result.get('similarity', 0.0)
                
                # Try to extract source file from tags
                tags = result.get('tags', [])
                source_file = "Unknown"
                chunk_index = 0
                
                for tag in tags:
                    if tag.startswith('document:'):
                        source_file = tag.replace('document:', '')
                    elif tag.startswith('chunk:'):
                        try:
                            chunk_index = int(tag.replace('chunk:', ''))
                        except ValueError:
                            pass
                
                formatted_results.append({
                    "index": i,
                    "source_file": source_file,
                    "chunk_index": chunk_index + 1,
                    "similarity": round(similarity, 2),
                    "preview": content_preview
                })
            
            return {
                "success": True,
                "response": None,
                "data": {
                    "query": search_terms,
                    "count": len(formatted_results),
                    "results": formatted_results,
                    "message_code": "docs:search_results"
                }
            }
            
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "docs:search_error"
                }
            }
    
    def _handle_list_documents(self) -> Dict[str, Any]:
        """Handle list documents request"""
        try:
            documents = self.memory_system.list_ingested_documents()
            
            if not documents:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_documents",
                        "message_code": "docs:no_documents"
                    }
                }
            formatted_docs = []
            for doc in documents:
                formatted_docs.append({
                    "filename": doc.get('filename'),
                    "file_type": doc.get('file_type'),
                    "chunks": doc.get('chunks'),
                    "size_mb": doc.get('size_mb'),
                    "ingested_at": doc.get('ingested_at')
                })
            
            return {
                "success": True,
                "response": None,
                "data": {
                    "count": len(formatted_docs),
                    "documents": formatted_docs,
                    "message_code": "docs:list"
                }
            }
            
        except Exception as e:
            logger.error(f"List documents error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "docs:list_error"
                }
            }
    
    def _handle_document_question(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Handle questions that might be answered from documents"""
        try:
            # Use document search for RAG
            results = self.memory_system.search_documents(query, top_k=3)
            
            if not results:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_relevant_info",
                        "message_code": "docs:no_relevant_info"
                    }
                }
            
            # Create context from document results
            context = "Based on my document knowledge base:\n\n"
            for i, result in enumerate(results, 1):
                # Get source file from tags
                tags = result.get('tags', [])
                source_file = "Unknown"
                for tag in tags:
                    if tag.startswith('document:'):
                        source_file = tag.replace('document:', '')
                        break
                
                context += f"From {source_file}:\n{result['content']}\n\n"
            
            return {
                "success": True,
                "response": None,
                "data": {
                    "rag_context": True,
                    "rag_context_text": context,
                    "message_code": "docs:rag_context"
                }
            }
            
        except Exception as e:
            logger.error(f"Document question error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "docs:question_error"
                }
            }
    
    def _extract_file_path(self, query: str) -> Optional[str]:
        """Extract file path from query"""
        import re
        
        # Common patterns for file paths
        patterns = [
            r'from\s+["\']?([^"\']+)["\']?',  # "from C:\path\file.txt"
            r'path\s+["\']?([^"\']+)["\']?',   # "path C:\path\file.txt"
            r'["\']([^"\']+\.[a-zA-Z0-9]{2,5})["\']',  # "C:\path\file.pdf"
            r'([A-Za-z]:[^\s]+\.[a-zA-Z0-9]{2,5})',   # C:\path\file.pdf
            r'(/[^\s]+\.[a-zA-Z0-9]{2,5})',           # /path/file.pdf
            r'(\.?/[^\s]+\.[a-zA-Z0-9]{2,5})',        # ./path/file.pdf
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_search_terms(self, query: str) -> str:
        """Extract search terms from query"""
        import re
        
        # Remove command keywords and extract the actual search terms
        patterns = [
            r'search documents for\s+(.+)',
            r'find in documents\s+(.+)',
            r'search files for\s+(.+)',
            r'look for\s+(.+)\s+in documents',
            r'find\s+(.+)\s+in files',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matched, return the whole query (minus common words)
        words_to_remove = ['search', 'documents', 'find', 'files', 'for', 'in', 'the']
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in words_to_remove]
        return ' '.join(filtered_words)
    
    def shutdown(self):
        """Cleanup when plugin is disabled"""
        # No cleanup needed for document plugin
        pass
    
    def _is_document_question(self, query: str) -> bool:
        """Check if query is a question that might be answered from documents"""
        question_indicators = ["what is", "how to", "tell me about", "explain", "describe"]
        knowledge_keywords = ["definition", "tutorial", "guide", "documentation", "manual"]
        
        return any(indicator in query for indicator in question_indicators) or \
               any(keyword in query for keyword in knowledge_keywords)
        """Check if query is a question that might be answered from documents"""
        question_indicators = ["what is", "how to", "tell me about", "explain", "describe"]
    def shutdown(self):
        """Shutdown the plugin"""
        # No cleanup needed for document plugin
        pass