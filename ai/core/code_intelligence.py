"""
Code Intelligence Module
=========================
Advanced algorithms for code analysis, understanding, and generation.
Provides Alice with sophisticated code comprehension capabilities.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Advanced code analysis engine with AST parsing and pattern recognition.
    """

    def __init__(self):
        self.supported_languages = {'python', 'javascript', 'typescript', 'java', 'cpp'}

    def analyze_python_code(self, code: str) -> Dict[str, Any]:
        """
        Deep analysis of Python code using AST.

        Returns:
            Dictionary containing:
            - complexity: Cyclomatic complexity score
            - functions: List of function definitions with metadata
            - classes: List of class definitions with metadata
            - imports: List of imported modules
            - variables: Top-level variables
            - docstrings: Extracted documentation
            - patterns: Detected design patterns
            - quality_score: Code quality assessment (0-1)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                'error': f'Syntax error: {e}',
                'valid': False
            }

        analysis = {
            'valid': True,
            'complexity': 0,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'docstrings': [],
            'patterns': [],
            'quality_score': 0.0,
            'metrics': {}
        }

        # Analyze AST nodes
        for node in ast.walk(tree):
            # Functions
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node)
                analysis['functions'].append(func_info)
                analysis['complexity'] += func_info['complexity']

            # Classes
            elif isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                analysis['classes'].append(class_info)

            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    analysis['imports'].append(f"{module}.{alias.name}")

            # Variables
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        analysis['variables'].append(target.id)

        # Extract docstrings
        analysis['docstrings'] = self._extract_docstrings(tree)

        # Detect design patterns
        analysis['patterns'] = self._detect_patterns(tree, analysis)

        # Calculate quality metrics
        analysis['metrics'] = self._calculate_metrics(code, analysis)
        analysis['quality_score'] = self._calculate_quality_score(analysis['metrics'])

        return analysis

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition"""
        # Calculate cyclomatic complexity
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Each decision point adds to complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Extract parameters
        args = [arg.arg for arg in node.args.args]

        # Get return type if annotated
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'annotated'

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Count lines
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            lines = node.end_lineno - node.lineno + 1
        else:
            lines = 0

        return {
            'name': node.name,
            'args': args,
            'arg_count': len(args),
            'return_type': return_type,
            'docstring': docstring,
            'complexity': complexity,
            'lines': lines,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }

    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition"""
        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        # Get base classes
        bases = [self._get_base_name(base) for base in node.bases]

        # Extract docstring
        docstring = ast.get_docstring(node)

        return {
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'method_count': len(methods),
            'attributes': attributes,
            'docstring': docstring,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        return 'unknown'

    def _get_base_name(self, base: ast.expr) -> str:
        """Extract base class name"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        return 'unknown'

    def _extract_docstrings(self, tree: ast.AST) -> List[str]:
        """Extract all docstrings from AST"""
        docstrings = []

        # Module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            docstrings.append(module_doc)

        # Function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.append(doc)

        return docstrings

    def _detect_patterns(self, tree: ast.AST, analysis: Dict[str, Any]) -> List[str]:
        """Detect common design patterns in code"""
        patterns = []

        # Singleton pattern
        for cls in analysis['classes']:
            if 'singleton' in cls['name'].lower():
                patterns.append('Singleton')
            elif any('__new__' in m for m in cls['methods']):
                patterns.append('Singleton (via __new__)')

        # Factory pattern
        for func in analysis['functions']:
            if 'factory' in func['name'].lower() or 'create' in func['name'].lower():
                patterns.append('Factory')

        # Decorator pattern
        for func in analysis['functions']:
            if func['decorators']:
                patterns.append('Decorator')
                break

        # Observer pattern (event-based)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if 'observer' in node.id.lower() or 'listener' in node.id.lower():
                    patterns.append('Observer')
                    break

        return list(set(patterns))  # Remove duplicates

    def _calculate_metrics(self, code: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])

        # Average function complexity
        if analysis['functions']:
            avg_complexity = sum(f['complexity'] for f in analysis['functions']) / len(analysis['functions'])
        else:
            avg_complexity = 1.0

        # Documentation ratio
        documented_functions = len([f for f in analysis['functions'] if f['docstring']])
        documented_classes = len([c for c in analysis['classes'] if c['docstring']])
        total_definitions = len(analysis['functions']) + len(analysis['classes'])
        doc_ratio = (documented_functions + documented_classes) / total_definitions if total_definitions > 0 else 0.0

        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'comment_ratio': comment_lines / total_lines if total_lines > 0 else 0.0,
            'avg_complexity': avg_complexity,
            'max_complexity': max((f['complexity'] for f in analysis['functions']), default=1),
            'function_count': len(analysis['functions']),
            'class_count': len(analysis['classes']),
            'doc_ratio': doc_ratio,
            'import_count': len(analysis['imports'])
        }

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall code quality score (0-1).

        Factors:
        - Low complexity (better)
        - Good documentation (better)
        - Reasonable comment ratio (better)
        - Not too long functions (better)
        """
        score = 1.0

        # Penalize high complexity
        if metrics['avg_complexity'] > 10:
            score -= 0.2
        elif metrics['avg_complexity'] > 5:
            score -= 0.1

        # Reward good documentation
        if metrics['doc_ratio'] > 0.8:
            score += 0.1
        elif metrics['doc_ratio'] < 0.3:
            score -= 0.15

        # Reward balanced comments
        comment_ratio = metrics['comment_ratio']
        if 0.1 <= comment_ratio <= 0.3:
            score += 0.05
        elif comment_ratio < 0.05:
            score -= 0.1

        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    def suggest_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest code improvements based on analysis"""
        suggestions = []

        metrics = analysis.get('metrics', {})

        # Complexity suggestions
        if metrics.get('avg_complexity', 0) > 10:
            suggestions.append(
                "High complexity detected. Consider breaking down complex functions "
                "into smaller, more focused functions."
            )

        # Documentation suggestions
        if metrics.get('doc_ratio', 1.0) < 0.5:
            suggestions.append(
                "Low documentation coverage. Add docstrings to functions and classes "
                "to explain their purpose and usage."
            )

        # Comment suggestions
        if metrics.get('comment_ratio', 0) < 0.05:
            suggestions.append(
                "Very few comments. Add comments to explain complex logic and "
                "non-obvious implementations."
            )

        # Function length suggestions
        long_functions = [f for f in analysis.get('functions', []) if f.get('lines', 0) > 50]
        if long_functions:
            suggestions.append(
                f"Found {len(long_functions)} long functions (>50 lines). "
                "Consider refactoring for better readability."
            )

        # Class design suggestions
        large_classes = [c for c in analysis.get('classes', []) if c.get('method_count', 0) > 20]
        if large_classes:
            suggestions.append(
                f"Found {len(large_classes)} large classes (>20 methods). "
                "Consider whether they violate Single Responsibility Principle."
            )

        return suggestions

    def detect_code_smells(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Detect common code smells"""
        smells = []

        # Long parameter list
        for func in analysis.get('functions', []):
            if func['arg_count'] > 5:
                smells.append({
                    'type': 'Long Parameter List',
                    'location': f"Function '{func['name']}'",
                    'description': f"Has {func['arg_count']} parameters. Consider using a config object."
                })

        # God class
        for cls in analysis.get('classes', []):
            if cls['method_count'] > 20:
                smells.append({
                    'type': 'God Class',
                    'location': f"Class '{cls['name']}'",
                    'description': f"Has {cls['method_count']} methods. Consider splitting responsibilities."
                })

        # High complexity
        for func in analysis.get('functions', []):
            if func['complexity'] > 15:
                smells.append({
                    'type': 'High Complexity',
                    'location': f"Function '{func['name']}'",
                    'description': f"Complexity: {func['complexity']}. Refactor to reduce branching."
                })

        return smells


class CodeGenerator:
    """
    Intelligent code generation with pattern recognition.
    """

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load code templates for common patterns"""
        return {
            'function': '''def {name}({args}):
    """
    {description}

    Args:
        {arg_docs}

    Returns:
        {return_doc}
    """
    {body}
''',
            'class': '''class {name}({bases}):
    """
    {description}
    """

    def __init__(self{init_args}):
        """Initialize {name}"""
        {init_body}

    {methods}
''',
            'singleton': '''class {name}:
    """
    {description}
    Singleton pattern implementation.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            {init_body}
'''
        }

    def generate_function(
        self,
        name: str,
        args: List[str],
        description: str,
        return_type: str = 'Any'
    ) -> str:
        """Generate a function template"""
        arg_docs = '\n        '.join(f"{arg}: Description of {arg}" for arg in args)
        args_str = ', '.join(args)

        return self.templates['function'].format(
            name=name,
            args=args_str,
            description=description,
            arg_docs=arg_docs,
            return_doc=f"Description of return value ({return_type})",
            body="    pass  # TODO: Implement"
        )

    def generate_class(
        self,
        name: str,
        description: str,
        bases: List[str] = None,
        methods: List[str] = None
    ) -> str:
        """Generate a class template"""
        bases_str = ', '.join(bases) if bases else ''
        methods_str = '\n    '.join(methods) if methods else 'pass'

        return self.templates['class'].format(
            name=name,
            bases=bases_str,
            description=description,
            init_args='',
            init_body='pass',
            methods=methods_str
        )


# Global singleton instances
_code_analyzer = None
_code_generator = None


def get_code_analyzer() -> CodeAnalyzer:
    """Get global code analyzer instance"""
    global _code_analyzer
    if _code_analyzer is None:
        _code_analyzer = CodeAnalyzer()
    return _code_analyzer


def get_code_generator() -> CodeGenerator:
    """Get global code generator instance"""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator
