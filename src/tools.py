"""
Tool implementations for the LangGraph agent.

Provides: calculator, google_search, deep_research, neo4j_retrieveqa
"""

import ast
import json
import operator
import os
import re
from typing import Any, Dict, List, Optional

import requests
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.prompts.neo4j_prompts import CYPHER_PROMPT, QA_PROMPT


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        operator_func = SAFE_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        operator_func = SAFE_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return operator_func(operand)
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval_expression(expression: str) -> float:
    node = ast.parse(expression, mode="eval").body
    return _eval_node(node)


def _calculator_run(args: Dict[str, Any]) -> str:
    expression = args.get("expression", "").strip()
    if not expression:
        return "Calculator error: missing 'expression'."
    try:
        result = safe_eval_expression(expression)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return "Calculator error: Division by zero."
    except ValueError as e:
        return f"Calculator error: {str(e)}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


def _calculator_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations. Supports basic arithmetic: +, -, *, /, **, %, and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * (5 + 3)')",
                    },
                },
                "required": ["expression"],
            },
        },
    }


def _google_search_run(args: Dict[str, Any]) -> str:
    query = args.get("query") or ""
    if not query:
        return "GoogleSearch error: missing 'query'."

    api_key = os.getenv("GEMINI_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key:
        return "GoogleSearch error: GOOGLE_API_KEY not configured."
    if not cse_id:
        return "GoogleSearch error: GOOGLE_CSE_ID not configured."

    max_results = int(args.get("max_results", 5))
    num_results = min(max_results, 10)

    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cse_id, "q": query, "num": num_results}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            error_data = resp.json() if resp.content else {}
            error_msg = error_data.get("error", {}).get("message", resp.text)
            return f"GoogleSearch error: {resp.status_code} - {error_msg}"

        data = resp.json()
        items = data.get("items", [])
        results = [
            {"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")}
            for item in items
        ]
        output = {"query": query, "results": results, "total_results": len(results)}
        return json.dumps(output, indent=2)
    except requests.exceptions.RequestException as e:
        return f"GoogleSearch error: Network error - {str(e)}"
    except Exception as e:
        return f"GoogleSearch error: {str(e)}"


def _google_search_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Web search via Google Custom Search API; returns JSON with search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }


_NEO4J_CHAIN = None


def _neo4j_retrieveqa_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "neo4j_retrieveqa",
            "description": "Query Neo4j with natural language. Generates Cypher, retrieves context, and answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural-language question about the Neo4j graph."},
                    "top_k": {"type": "integer", "default": 10},
                    "include_context": {"type": "boolean", "default": True},
                },
                "required": ["question"],
            },
        },
    }


def _extract_cypher_query(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("cypher") or text.startswith("Cypher"):
            text = text[6:].strip()
    if text.lower().startswith("cypher:"):
        text = text[7:].strip()
    valid_starts = ["MATCH", "RETURN", "CREATE", "MERGE", "WITH", "UNWIND", "CALL", "OPTIONAL"]
    first_word = text.split()[0].upper() if text.split() else ""
    if first_word not in valid_starts:
        for line in text.split("\n"):
            line = line.strip()
            if line and line.split()[0].upper() in valid_starts:
                idx = text.index(line)
                text = text[idx:]
                break
    return text.strip()


def _build_neo4j_chain(top_k: int = 10):
    global _NEO4J_CHAIN
    if _NEO4J_CHAIN is not None:
        return _NEO4J_CHAIN

    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME") or "neo4j"
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD") or "password"
    db = os.getenv("NEO4J_DATABASE", None)

    graph = Neo4jGraph(url=uri, username=user, password=pwd, database=db)

    cypher_prompt_template = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_PROMPT)
    qa_prompt_template = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT)

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not configured for Neo4j tool usage.")

    cypher_llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api_key, temperature=0.0)
    qa_llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api_key, temperature=0.2)

    _NEO4J_CHAIN = GraphCypherQAChain.from_llm(
        llm=cypher_llm,
        qa_llm=qa_llm,
        graph=graph,
        cypher_prompt=cypher_prompt_template,
        qa_prompt=qa_prompt_template,
        return_intermediate_steps=True,
        top_k=top_k,
        verbose=True,
        allow_dangerous_requests=True,
        validate_cypher=True,
    )
    return _NEO4J_CHAIN


def _neo4j_retrieveqa_run(args: Dict[str, Any]) -> str:
    question = (args or {}).get("question", "").strip()
    if not question:
        return "Neo4jRetrieveQA error: missing 'question'."
    top_k = int((args or {}).get("top_k", 10))
    include_context = bool((args or {}).get("include_context", True))

    try:
        chain = _build_neo4j_chain(top_k=top_k)
        res = chain.invoke({"query": question})
        answer = res.get("result", "")
        interm = res.get("intermediate_steps", []) or []
        cypher = ""
        context = []

        if isinstance(interm, list) and len(interm) > 0:
            if isinstance(interm[0], dict):
                cypher = interm[0].get("query", "")
                context = interm[0].get("context", [])
            else:
                cypher = str(interm[0]) if len(interm) > 0 else ""
                context = interm[1] if len(interm) > 1 else []
        elif isinstance(interm, dict):
            cypher = interm.get("query", "")
            context = interm.get("context", [])

        payload = {"answer": answer, "generated_cypher": cypher, "top_k": top_k}
        if include_context:
            payload["context"] = context
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"error": f"Neo4jRetrieveQA error: {str(e)}", "question": question})


def _deep_research_run(args: Dict[str, Any]) -> str:
    try:
        from langchain_community.tools.tavily_search.tool import TavilySearchResults
    except ImportError:
        return "DeepResearch error: langchain_community or tavily-python not installed."

    query = args.get("query") or ""
    if not query:
        return "DeepResearch error: missing 'query'."
    tool = TavilySearchResults(
        max_results=int(args.get("max_results", 5)),
        include_answer=bool(args.get("include_answer", True)),
        include_raw_content=bool(args.get("include_raw_content", False)),
        search_depth=str(args.get("search_depth", "advanced")),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    out = tool.invoke({"query": query})
    try:
        return json.dumps(out)
    except Exception:
        return str(out)


def _deep_research_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Web deep-research via Tavily; returns aggregated JSON with sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "default": 5},
                    "search_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "advanced"},
                    "include_answer": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        },
    }
