"""
Tool definitions for the agent using @tool decorator
Think of these as the agent's Swiss Army knife 🔧 - each tool has a specific purpose!
"""

from typing import Dict, Any, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field
import re
import json
from datetime import datetime


class ToolLogger:
    """Logs tool usage with automatic persistence"""

    def __init__(self, logs_dir: str = "./logs", session_id: str = None):
        self.logs = []
        self.logs_dir = logs_dir
        self.session_id = session_id

        # Make sure logs directory exists
        import os
        os.makedirs(logs_dir, exist_ok=True)

        # Create session-specific log file if session_id provided
        if session_id:
            self.log_file = os.path.join(logs_dir, f"session_{session_id}.json")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(logs_dir, f"tool_usage_{timestamp}.json")

    def log_tool_use(self, tool_name: str, input_data: Dict[str, Any], output: Any):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "input": input_data,
            "output": str(output),
        }
        self.logs.append(log_entry)

        # Automatically save to persistent file
        self._auto_save()
        return log_entry

    def _auto_save(self):
        """Automatically save logs to persistent file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to auto-save logs: {e}")

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def save_logs(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)


# TODO: Implement the calculator tool using the @tool decorator.
# This tool should safely evaluate mathematical expressions and log its usage.
# Refer to README.md Task 4.1 for detailed implementation requirements.

def create_calculator_tool(logger: ToolLogger):
    """
    Creates a calculator tool that safely evaluates mathematical expressions and logs its usage.
    """
    import ast
    import math
    import operator as op

    # Allowed operators mapped to Python functions
    _ALLOWED_BIN_OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }
    _ALLOWED_UNARY_OPS = {
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    # Allowed names and functions (keep it tight for safety)
    _ALLOWED_NAMES = {
        "pi": math.pi,
        "e": math.e,
    }
    _ALLOWED_FUNCS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
    }

    def _safe_eval(expr: str) -> float:
        """
        Safely evaluate a mathematical expression using an AST whitelist approach.
        """
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Expression must be a non-empty string.")

        # Quick hardening: only allow typical math characters and function/name letters/underscores
        if not re.fullmatch(r"[0-9+\-*/().,\s%_a-zA-Z]+", expr):
            raise ValueError("Expression contains invalid characters.")

        # Basic size guard
        if len(expr) > 200:
            raise ValueError("Expression too long.")

        node = ast.parse(expr, mode="eval")

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Expression):
                return _eval(n.body)

            # Numbers
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)

            # Binary operations
            if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN_OPS:
                left = _eval(n.left)
                right = _eval(n.right)

                # Optional guard: prevent absurd exponents
                if isinstance(n.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("Exponent too large.")

                return float(_ALLOWED_BIN_OPS[type(n.op)](left, right))

            # Unary operations (+x, -x)
            if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY_OPS:
                return float(_ALLOWED_UNARY_OPS[type(n.op)](_eval(n.operand)))

            # Names (pi, e)
            if isinstance(n, ast.Name) and n.id in _ALLOWED_NAMES:
                return float(_ALLOWED_NAMES[n.id])

            # Function calls (abs(...), round(...), sqrt(...), etc.)
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name):
                    raise ValueError("Only simple function calls are allowed.")

                func_name = n.func.id
                if func_name not in _ALLOWED_FUNCS:
                    raise ValueError(f"Function '{func_name}' is not allowed.")

                args = [_eval(a) for a in n.args]
                # Disallow keyword args for simplicity/safety
                if n.keywords:
                    raise ValueError("Keyword arguments are not allowed.")

                return float(_ALLOWED_FUNCS[func_name](*args))

            raise ValueError("Unsupported expression structure.")

        return _eval(node)

    @tool
    def calculator(expression: str) -> str:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression (e.g., "1200 + 35.5", "round(10/3, 2)", "sqrt(16)")

        Returns:
            A formatted string with the numeric result or an error message.
        """
        try:
            result = _safe_eval(expression)

            logger.log_tool_use(
                "calculator",
                {"expression": expression},
                {"result": result},
            )

            return f"Result: {result}"
        except Exception as e:
            error_msg = f"Error evaluating expression: {str(e)}"
            logger.log_tool_use(
                "calculator",
                {"expression": expression},
                {"error": error_msg},
            )
            return error_msg

    return calculator

def create_document_search_tool(retriever, logger: ToolLogger):
    """
    Creates a document search tool.
    """

    @tool
    def document_search(
            query: str,
            search_type: Literal["keyword", "type", "amount", "amount_range", "all"] = "keyword",
            doc_type: Optional[str] = None,
            min_amount: Optional[float] = None,
            max_amount: Optional[float] = None,
            comparison: Optional[Literal["over", "under", "between", "exact", "approximate"]] = None,
            amount: Optional[float] = None
    ) -> str:
        """
        Search for relevant documents using various criteria. Handles natural language amount queries.

        Args:
            query: Search query (e.g., "invoices over $50,000", "contracts", "insurance claims")
            search_type: Type of search - 'keyword', 'type', 'amount', 'amount_range' or 'all'
            doc_type: Document type filter (e.g., 'invoice', 'contract', 'claim')
            min_amount: Minimum amount (for range queries or "over" queries)
            max_amount: Maximum amount (for range queries or "under" queries)
            comparison: Type of amount comparison - 'over', 'under', 'between', 'exact', 'approximate'
            amount: Single amount value for comparisons (used with 'over', 'under', 'exact', 'approximate')

        Examples:
            - "Find documents over $50,000" → comparison='over', amount=50000
            - "Show invoices under $10,000" → search_type='type', doc_type='invoice', comparison='under', amount=10000
            - "Documents between $20,000 and $80,000" → min_amount=20000, max_amount=80000
            - "Contracts around $100,000" → comparison='approximate', amount=100000

        Returns:
            Formatted search results with document details
        """
        try:
            results = []

            # Handle different search types
            if search_type == "all":
                results = retriever.retrieve_all()

            if search_type == "keyword":
                results = retriever.retrieve_by_keyword(query)

            elif search_type == "type" and doc_type:
                results = retriever.retrieve_by_type(doc_type)
                # If amount criteria also specified, filter further
                if comparison or min_amount is not None or max_amount is not None:
                    amount_results = _handle_amount_search(
                        retriever, comparison, amount, min_amount, max_amount, query
                    )
                    # Intersect results
                    result_ids = {r.doc_id for r in amount_results}
                    results = [r for r in results if r.doc_id in result_ids]

            elif search_type == "amount" or search_type == "amount_range":
                results = _handle_amount_search(
                    retriever, comparison, amount, min_amount, max_amount, query
                )

            else:
                # Try to intelligently parse the query
                query_lower = query.lower()

                # Check if it's an amount query
                if any(word in query_lower for word in
                       ['over', 'under', 'above', 'below', 'between', 'around', 'exactly', '$']):
                    results = retriever._parse_and_retrieve_by_amount(query)
                # Check if it's a type query
                elif any(word in query_lower for word in ['invoice', 'contract', 'claim']):
                    for doc_type in ['invoice', 'contract', 'claim']:
                        if doc_type in query_lower:
                            results = retriever.retrieve_by_type(doc_type)
                            break
                else:
                    # Default to keyword search
                    results = retriever.retrieve_by_keyword(query)

            # Format results with amount information
            if not results:
                formatted = "No documents found matching your search criteria."
            else:
                formatted = f"Found {len(results)} document(s):\n\n"
                for i, chunk in enumerate(results, 1):
                    formatted += f"Document {i} (ID: {chunk.doc_id}):\n"
                    formatted += f"Title: {chunk.metadata.get('title', 'Unknown')}\n"
                    formatted += f"Type: {chunk.metadata.get('doc_type', 'Unknown')}\n"

                    # Include amount information if available
                    amount_value = None
                    for field in ['total', 'amount', 'value']:
                        if field in chunk.metadata:
                            amount_value = chunk.metadata[field]
                            formatted += f"Amount: ${amount_value:,.2f}\n"
                            break

                    if hasattr(chunk, 'relevance_score'):
                        formatted += f"Relevance Score: {chunk.relevance_score:.2f}\n"

                    formatted += f"Preview: {chunk.content[:200]}...\n"
                    formatted += "-" * 50 + "\n"

            # Log the tool use
            logger.log_tool_use(
                "document_search",
                {
                    "query": query,
                    "search_type": search_type,
                    "doc_type": doc_type,
                    "min_amount": min_amount,
                    "max_amount": max_amount,
                    "comparison": comparison,
                    "amount": amount
                },
                {"results_count": len(results)}
            )

            return formatted

        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            logger.log_tool_use(
                "document_search",
                {"query": query, "search_type": search_type},
                {"error": error_msg}
            )
            return error_msg

    def _handle_amount_search(retriever, comparison, amount, min_amount, max_amount, query):
        """Helper function to handle amount-based searches"""
        if comparison:
            if comparison == "over" and amount is not None:
                return retriever.retrieve_by_amount_range(min_amount=amount)
            elif comparison == "under" and amount is not None:
                return retriever.retrieve_by_amount_range(max_amount=amount)
            elif comparison == "exact" and amount is not None:
                return retriever.retrieve_by_exact_amount(amount)
            elif comparison == "approximate" and amount is not None:
                return retriever.retrieve_by_approximate_amount(amount)
            elif comparison == "between" and min_amount is not None and max_amount is not None:
                return retriever.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount)

        # Handle direct min/max specifications
        if min_amount is not None or max_amount is not None:
            return retriever.retrieve_by_amount_range(min_amount=min_amount, max_amount=max_amount)

        # Try parsing from query
        return retriever._parse_and_retrieve_by_amount(query)

    # Store helper function as attribute
    document_search._handle_amount_search = _handle_amount_search

    return document_search


def create_document_reader_tool(retriever, logger: ToolLogger):
    """
    Creates a tool to read full document content.
    """

    @tool
    def document_reader(doc_id: str) -> str:
        """
        Read the full content of a specific document by its ID.

        Args:
            doc_id: The exact document ID to read (e.g., 'INV-001', 'CON-001')

        Returns:
            The full content of the document or an error message if not found
        """
        try:
            doc = retriever.get_document_by_id(doc_id)
            if doc:
                # Include amount information in the output
                amount_info = ""
                for field in ['total', 'amount', 'value']:
                    if field in doc.metadata:
                        amount_info = f"\nAmount: ${doc.metadata[field]:,.2f}"
                        break

                result = f"Document {doc_id}:{amount_info}\n\n{doc.content}"
                logger.log_tool_use(
                    "document_reader",
                    {"doc_id": doc_id},
                    {"found": True, "doc_type": doc.metadata.get('doc_type')}
                )
                return result
            else:
                logger.log_tool_use(
                    "document_reader",
                    {"doc_id": doc_id},
                    {"found": False}
                )
                return f"Document with ID {doc_id} not found."
        except Exception as e:
            error_msg = f"Error reading document: {str(e)}"
            logger.log_tool_use(
                "document_reader",
                {"doc_id": doc_id},
                {"error": error_msg}
            )
            return error_msg

    return document_reader


def create_document_statistics_tool(retriever, logger: ToolLogger):
    """
    Creates a tool to get statistics about the document collection.
    """

    @tool
    def document_statistics() -> str:
        """
        Get statistics about all documents in the system.

        Returns:
            Summary statistics including document counts, amount totals, and averages
        """
        try:
            stats = retriever.get_statistics()

            formatted = "DOCUMENT COLLECTION STATISTICS:\n\n"
            formatted += f"Total Documents: {stats['total_documents']}\n"
            formatted += f"Documents with Amounts: {stats['documents_with_amounts']}\n"
            formatted += f"\nDocument Types:\n"

            for doc_type, count in stats['document_types'].items():
                formatted += f"  - {doc_type.capitalize()}: {count}\n"

            if stats['documents_with_amounts'] > 0:
                formatted += f"\nFinancial Summary:\n"
                formatted += f"  - Total Amount: ${stats['total_amount']:,.2f}\n"
                formatted += f"  - Average Amount: ${stats['average_amount']:,.2f}\n"
                formatted += f"  - Minimum Amount: ${stats['min_amount']:,.2f}\n"
                formatted += f"  - Maximum Amount: ${stats['max_amount']:,.2f}\n"

            logger.log_tool_use(
                "document_statistics",
                {},
                {"stats": stats}
            )

            return formatted

        except Exception as e:
            error_msg = f"Error getting statistics: {str(e)}"
            logger.log_tool_use(
                "document_statistics",
                {},
                {"error": error_msg}
            )
            return error_msg

    return document_statistics


def get_all_tools(retriever, logger: ToolLogger) -> List:
    """
    Get all available tools for the agent.
    """
    return [
        create_calculator_tool(logger),
        create_document_search_tool(retriever, logger),
        create_document_reader_tool(retriever, logger),
        create_document_statistics_tool(retriever, logger)
    ]