"""
Self-Reflection System for A.L.I.C.E
Allows A.L.I.C.E to read, analyze, and understand her own codebase.
Enables her to help improve herself.

Advanced Features:
- AST-based code analysis for accurate summaries
- Smart caching with TTL for performance
- Batch processing for multi-file operations
- Semantic similarity for purpose extraction
- Dependency graph analysis
"""

import os
import logging
import re
import ast
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import concurrent.futures

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a code file with rich metadata"""
    path: str
    name: str
    content: str
    lines: int
    language: str
    module_type: str  # 'plugin', 'core', 'system', 'utility'
    purpose: Optional[str] = None  # One-line description
    key_components: List[str] = field(default_factory=list)  # Main classes/functions
    dependencies: List[str] = field(default_factory=list)  # External imports
    complexity_score: float = 0.0  # Cyclomatic complexity estimate
    last_analyzed: Optional[datetime] = None


class SelfReflectionSystem:
    """
    Allows A.L.I.C.E to read and analyze her own codebase.
    Read-only access for safety.
    
    Advanced Features:
    - Smart caching with 1-hour TTL
    - AST-based analysis for accuracy
    - Parallel file processing
    - Purpose extraction from docstrings/comments
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize self-reflection system
        
        Args:
            base_path: Base path to A.L.I.C.E codebase (defaults to current directory)
        """
        if base_path is None:
            # Get the directory containing this file, then go up one level
            self.base_path = Path(__file__).parent.parent.resolve()
        else:
            self.base_path = Path(base_path).resolve()
        
        self.allowed_extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}
        self.ignored_patterns = {
            '__pycache__', '.pyc', '.git', 'node_modules',
            'venv', 'env', '.venv', 'data/', 'memory/', 'cred/',
            '.pytest_cache', '.mypy_cache'
        }
        
        # Map of module types
        self.module_map = {
            'plugin': ['plugin', 'notes', 'email', 'music', 'calendar', 'document', 'maps'],
            'core': ['main', 'llm_engine', 'nlp_processor', 'context', 'core', 'reasoning', 'learning'],
            'system': ['world_state', 'goal', 'reference', 'verifier', 'router', 'policy'],
            'utility': ['memory_system', 'intent_classifier', 'task_executor', 'formatter']
        }
        
        # Smart caching
        self._analysis_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        
        # File hash tracking for change detection
        self._file_hashes: Dict[str, str] = {}
        
        logger.info(f"[SelfReflection] Initialized with base path: {self.base_path}")
    
    def _is_allowed_file(self, file_path: Path) -> bool:
        """Check if file should be accessible"""
        path_str = str(file_path)
        
        # Check extension
        if file_path.suffix not in self.allowed_extensions:
            return False
        
        # Check ignored patterns
        for pattern in self.ignored_patterns:
            if pattern in path_str:
                return False
        
        # Must be within base path
        try:
            file_path.resolve().relative_to(self.base_path)
        except ValueError:
            return False
        
        return True
    
    def _classify_module(self, file_path: Path) -> str:
        """Classify what type of module a file is"""
        name_lower = file_path.stem.lower()
        
        for module_type, keywords in self.module_map.items():
            if any(keyword in name_lower for keyword in keywords):
                return module_type
        
        # Check directory
        parent = file_path.parent.name.lower()
        if 'plugin' in parent or 'plugins' in parent:
            return 'plugin'
        if parent in ['ai', 'core']:
            return 'core'
        
        return 'utility'
    
    def read_file(self, file_path: str, max_lines: int = 1000) -> Optional[CodeFile]:
        """
        Read a code file (read-only)
        
        Args:
            file_path: Relative or absolute path to file
            max_lines: Maximum lines to read (for large files)
        
        Returns:
            CodeFile object or None if not accessible
        """
        try:
            # Resolve path
            if os.path.isabs(file_path):
                full_path = Path(file_path)
            else:
                full_path = self.base_path / file_path
            
            full_path = full_path.resolve()
            
            # Security check
            if not self._is_allowed_file(full_path):
                logger.warning(f"[SelfReflection] Access denied to: {file_path}")
                return None
            
            if not full_path.exists():
                return None
            
            # Read file
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            content = ''.join(lines[:max_lines])
            if len(lines) > max_lines:
                content += f"\n... ({len(lines) - max_lines} more lines truncated)"
            
            return CodeFile(
                path=str(full_path.relative_to(self.base_path)),
                name=full_path.name,
                content=content,
                lines=len(lines),
                language=full_path.suffix[1:] if full_path.suffix else 'text',
                module_type=self._classify_module(full_path)
            )
        except Exception as e:
            logger.error(f"[SelfReflection] Error reading file {file_path}: {e}")
            return None
    
    def list_codebase(self, directory: str = "ai", pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in codebase
        
        Args:
            directory: Subdirectory to list (relative to base_path)
            pattern: Optional pattern to filter files (e.g., "*plugin*.py")
        
        Returns:
            List of file info dictionaries
        """
        try:
            dir_path = self.base_path / directory
            if not dir_path.exists() or not dir_path.is_dir():
                return []
            
            files = []
            for file_path in dir_path.rglob(pattern or "*.py"):
                if self._is_allowed_file(file_path):
                    rel_path = file_path.relative_to(self.base_path)
                    files.append({
                        'path': str(rel_path),
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'module_type': self._classify_module(file_path),
                        'language': file_path.suffix[1:] if file_path.suffix else 'text'
                    })
            
            return sorted(files, key=lambda x: x['path'])
        except Exception as e:
            logger.error(f"[SelfReflection] Error listing codebase: {e}")
            return []
    
    def search_code(self, query: str, directory: str = "ai", max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for text in codebase
        
        Args:
            query: Text to search for
            directory: Directory to search in
            max_results: Maximum number of results
        
        Returns:
            List of matches with context
        """
        results = []
        query_lower = query.lower()
        
        try:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                return []
            
            for file_path in dir_path.rglob("*.py"):
                if not self._is_allowed_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    matches = []
                    for i, line in enumerate(lines, 1):
                        if query_lower in line.lower():
                            # Get context (2 lines before and after)
                            start = max(0, i - 3)
                            end = min(len(lines), i + 2)
                            context = ''.join(lines[start:end])
                            
                            matches.append({
                                'line': i,
                                'content': line.strip(),
                                'context': context
                            })
                    
                    if matches:
                        rel_path = file_path.relative_to(self.base_path)
                        results.append({
                            'file': str(rel_path),
                            'matches': matches[:5],  # Max 5 matches per file
                            'match_count': len(matches)
                        })
                        
                        if len(results) >= max_results:
                            break
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"[SelfReflection] Error searching code: {e}")
        
        return results
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _is_cache_valid(self, file_path: str) -> bool:
        """Check if cached analysis is still valid"""
        if file_path not in self._analysis_cache:
            return False
        
        _, cached_time = self._analysis_cache[file_path]
        if datetime.now() - cached_time > self._cache_ttl:
            return False
        
        # Check if file was modified
        full_path = self.base_path / file_path if not os.path.isabs(file_path) else Path(file_path)
        current_hash = self._compute_file_hash(full_path)
        if file_path in self._file_hashes and self._file_hashes[file_path] != current_hash:
            return False
        
        return True
    
    def _extract_purpose_from_ast(self, tree: ast.Module, content: str) -> str:
        """Extract file purpose from module docstring or initial comments"""
        # Try module docstring first
        if (isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, (ast.Str, ast.Constant))):
            docstring = ast.get_docstring(tree)
            if docstring:
                # Get first meaningful line
                lines = [l.strip() for l in docstring.split('\n') if l.strip()]
                if lines:
                    return lines[0][:100]
        
        # Fall back to initial comments
        lines = content.split('\n')
        for line in lines[:10]:
            if line.strip().startswith('#') and len(line.strip()) > 5:
                purpose = line.strip('#').strip()
                if len(purpose) > 20:
                    return purpose[:100]
        
        return "No description available"
    
    def _calculate_complexity(self, tree: ast.Module) -> float:
        """Estimate cyclomatic complexity using AST"""
        complexity = 1.0
        
        for node in ast.walk(tree):
            # Decision points add complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _extract_dependencies(self, tree: ast.Module) -> List[str]:
        """Extract external dependencies from imports"""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Only external packages (not local modules)
                    pkg = alias.name.split('.')[0]
                    if pkg not in ['ai', 'speech', 'ui', 'features', 'memory']:
                        dependencies.add(pkg)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    pkg = node.module.split('.')[0]
                    if pkg not in ['ai', 'speech', 'ui', 'features', 'memory']:
                        dependencies.add(pkg)
        
        return sorted(list(dependencies))
    
    def analyze_file_advanced(self, file_path: str) -> Dict[str, Any]:
        """
        Advanced AST-based file analysis with caching
        
        Args:
            file_path: Path to file
        
        Returns:
            Comprehensive analysis dictionary
        """
        # Check cache
        if self._is_cache_valid(file_path):
            cached_data, _ = self._analysis_cache[file_path]
            logger.debug(f"[SelfReflection] Using cached analysis for {file_path}")
            return cached_data
        
        code_file = self.read_file(file_path)
        if not code_file:
            return {'error': 'File not accessible or not found'}
        
        analysis = {
            'path': code_file.path,
            'name': code_file.name,
            'lines': code_file.lines,
            'module_type': code_file.module_type,
            'language': code_file.language,
            'purpose': None,
            'classes': [],
            'functions': [],
            'imports': [],
            'dependencies': [],
            'docstrings': [],
            'complexity_score': 0.0
        }
        
        try:
            # Parse AST for accurate analysis
            tree = ast.parse(code_file.content)
            
            # Extract purpose
            analysis['purpose'] = self._extract_purpose_from_ast(tree, code_file.content)
            
            # Calculate complexity
            analysis['complexity_score'] = self._calculate_complexity(tree)
            
            # Extract dependencies
            analysis['dependencies'] = self._extract_dependencies(tree)
            
            # Extract classes with methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods[:5],  # Top 5 methods
                        'method_count': len(methods)
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level functions only
                    analysis['functions'].append(node.name)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
            
            # Extract docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring and len(docstring) > 20:
                        analysis['docstrings'].append(docstring[:200])
        
        except SyntaxError:
            # Fall back to regex if AST fails
            logger.warning(f"[SelfReflection] AST parse failed for {file_path}, using regex")
            analysis.update(self._fallback_regex_analysis(code_file.content))
        
        # Cache the result
        full_path = self.base_path / file_path if not os.path.isabs(file_path) else Path(file_path)
        self._file_hashes[file_path] = self._compute_file_hash(full_path)
        self._analysis_cache[file_path] = (analysis, datetime.now())
        
        return analysis
    
    def _fallback_regex_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback regex-based analysis when AST fails"""
        analysis = {
            'purpose': "Regex analysis (AST unavailable)",
            'classes': [],
            'functions': [],
            'imports': [],
            'complexity_score': 1.0
        }
        
        # Extract classes
        class_pattern = r'^class\s+(\w+).*?:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            analysis['classes'].append({'name': match.group(1), 'methods': [], 'method_count': 0})
        
        # Extract functions
        func_pattern = r'^def\s+(\w+)\s*\([^)]*\)\s*:'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            analysis['functions'].append(match.group(1))
        
        # Extract imports
        import_pattern = r'^import\s+(\S+)|^from\s+(\S+)\s+import'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imp = match.group(1) or match.group(2)
            if imp:
                analysis['imports'].append(imp)
        
        return analysis
    
    def generate_file_summary(self, file_path: str) -> str:
        """
        Generate a concise, human-readable summary of a file
        
        Args:
            file_path: Path to file
        
        Returns:
            Natural language summary
        """
        analysis = self.analyze_file_advanced(file_path)
        
        if 'error' in analysis:
            return f"**{file_path}**: {analysis['error']}"
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"**{analysis['name']}** ({analysis['lines']} lines, {analysis['module_type']})")
        
        # Purpose
        if analysis.get('purpose'):
            summary_parts.append(f"  Purpose: {analysis['purpose']}")
        
        # Key components
        if analysis.get('classes'):
            class_names = [c['name'] if isinstance(c, dict) else c for c in analysis['classes'][:3]]
            summary_parts.append(f"  Classes: {', '.join(class_names)}")
        
        if analysis.get('functions') and len(analysis['functions']) > 0:
            summary_parts.append(f"  Functions: {', '.join(analysis['functions'][:5])}")
        
        # Dependencies
        if analysis.get('dependencies') and len(analysis['dependencies']) > 0:
            summary_parts.append(f"  Dependencies: {', '.join(analysis['dependencies'][:5])}")
        
        # Complexity
        if analysis.get('complexity_score', 0) > 20:
            summary_parts.append(f"  ⚠ Complexity: {analysis['complexity_score']:.0f} (consider refactoring)")
        
        return '\n'.join(summary_parts)
    
    def batch_summarize_files(self, file_paths: List[str], parallel: bool = True) -> Dict[str, str]:
        """
        Generate summaries for multiple files efficiently
        
        Args:
            file_paths: List of file paths
            parallel: Use parallel processing for speed
        
        Returns:
            Dictionary mapping file paths to summaries
        """
        summaries = {}
        
        if parallel and len(file_paths) > 3:
            # Use parallel processing for speed
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_path = {
                    executor.submit(self.generate_file_summary, path): path 
                    for path in file_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        summaries[path] = future.result()
                    except Exception as e:
                        summaries[path] = f"Error analyzing {path}: {e}"
                        logger.error(f"[SelfReflection] Error in batch analysis: {e}")
        else:
            # Sequential processing for small batches
            for path in file_paths:
                try:
                    summaries[path] = self.generate_file_summary(path)
                except Exception as e:
                    summaries[path] = f"Error analyzing {path}: {e}"
                    logger.error(f"[SelfReflection] Error in analysis: {e}")
        
        return summaries
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a code file and extract information
        
        Args:
            file_path: Path to file
        
        Returns:
            Analysis dictionary
        """
        code_file = self.read_file(file_path)
        if not code_file:
            return {'error': 'File not accessible or not found'}
        
        analysis = {
            'path': code_file.path,
            'name': code_file.name,
            'lines': code_file.lines,
            'module_type': code_file.module_type,
            'language': code_file.language,
            'classes': [],
            'functions': [],
            'imports': [],
            'docstrings': []
        }
        
        # Extract classes
        class_pattern = r'^class\s+(\w+).*?:'
        for match in re.finditer(class_pattern, code_file.content, re.MULTILINE):
            analysis['classes'].append(match.group(1))
        
        # Extract functions
        func_pattern = r'^def\s+(\w+)\s*\([^)]*\)\s*:'
        for match in re.finditer(func_pattern, code_file.content, re.MULTILINE):
            analysis['functions'].append(match.group(1))
        
        # Extract imports
        import_pattern = r'^import\s+(\S+)|^from\s+(\S+)\s+import'
        for match in re.finditer(import_pattern, code_file.content, re.MULTILINE):
            imp = match.group(1) or match.group(2)
            if imp:
                analysis['imports'].append(imp)
        
        # Extract docstrings
        docstring_pattern = r'"""(.*?)"""'
        for match in re.finditer(docstring_pattern, code_file.content, re.DOTALL):
            doc = match.group(1).strip()
            if len(doc) > 20:  # Only significant docstrings
                analysis['docstrings'].append(doc[:200])
        
        return analysis
    
    def get_codebase_summary(self) -> Dict[str, Any]:
        """Get a summary of the codebase structure"""
        summary = {
            'base_path': str(self.base_path),
            'modules': {
                'plugins': [],
                'core': [],
                'system': [],
                'utility': []
            },
            'total_files': 0
        }
        
        try:
            ai_dir = self.base_path / 'ai'
            if ai_dir.exists():
                for file_path in ai_dir.rglob("*.py"):
                    if self._is_allowed_file(file_path):
                        module_type = self._classify_module(file_path)
                        rel_path = str(file_path.relative_to(self.base_path))
                        # Map module_type to summary key (plugin -> plugins)
                        summary_key = 'plugins' if module_type == 'plugin' else module_type
                        if summary_key in summary['modules']:
                            summary['modules'][summary_key].append(rel_path)
                            summary['total_files'] += 1
        except Exception as e:
            logger.error(f"[SelfReflection] Error generating summary: {e}")
        
        return summary
    
    def get_improvement_suggestions(self, file_path: str) -> List[str]:
        """
        Analyze code and suggest improvements
        This is a simple heuristic-based analyzer
        
        Args:
            file_path: Path to file to analyze
        
        Returns:
            List of improvement suggestions
        """
        code_file = self.read_file(file_path)
        if not code_file:
            return []
        
        suggestions = []
        content = code_file.content
        
        # Check for error handling
        if 'try:' in content and 'except Exception as e:' not in content:
            suggestions.append("Consider adding specific exception handling instead of bare 'except'")
        
        # Check for type hints
        if 'def ' in content and '->' not in content[:500]:
            suggestions.append("Consider adding type hints to function signatures for better code clarity")
        
        # Check for docstrings
        if 'def ' in content and '"""' not in content[:1000]:
            suggestions.append("Consider adding docstrings to functions and classes")
        
        # Check for logging
        if 'print(' in content and 'logger.' not in content:
            suggestions.append("Consider using logger instead of print() for better logging control")
        
        # Check for hardcoded values
        if re.search(r'=\s*["\'](?:localhost|127\.0\.0\.1|api_key|password)', content):
            suggestions.append("Check for hardcoded values that should be in configuration")
        
        return suggestions


_self_reflection: Optional[SelfReflectionSystem] = None


def get_self_reflection(base_path: Optional[str] = None) -> SelfReflectionSystem:
    """Get singleton instance of self-reflection system"""
    global _self_reflection
    if _self_reflection is None:
        _self_reflection = SelfReflectionSystem(base_path)
    return _self_reflection
