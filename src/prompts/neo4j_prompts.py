"""
Neo4j Database Query Prompts

Prompts used for generating Cypher queries and answering questions
based on Neo4j graph database context.
"""

CYPHER_PROMPT = """Task: Generate a valid Cypher query statement to query a Neo4j graph database.

CRITICAL: You MUST output ONLY a valid Cypher query. Do NOT include any explanations, comments, or other text.

Instructions:
- Use only the provided relationship types and properties in the schema below
- Do not use any other relationship types or properties that are not provided
- Always filter out NULL values when finding the highest value of a property
- Use LIMIT to prevent returning too many results (default: 10)
- For aggregations (count, sum, avg), use appropriate Cypher functions
- When ordering results, use ORDER BY DESC for "top" or "highest" queries
- Use DISTINCT when counting unique entities
- Return meaningful column names using AS
- Output ONLY the Cypher query, nothing else

Schema:
{schema}

Examples:
Q: "How many nodes of type X are there?"
Cypher: MATCH (n:X) RETURN count(n) as total

Q: "Which node has the highest value of property Y?"
Cypher: MATCH (n) WHERE n.Y IS NOT NULL RETURN n.name as name, n.Y as value ORDER BY n.Y DESC LIMIT 1

Q: "List all relationships between A and B"
Cypher: MATCH (a:A)-[r]->(b:B) RETURN a, type(r) as relationship, b LIMIT 10

IMPORTANT RULES:
1. Output ONLY the Cypher query
2. Do not include the word "Cypher:" before your query
3. Do not wrap your query in code blocks or backticks
4. Start directly with MATCH, RETURN, CREATE, or another valid Cypher keyword
5. If you cannot generate a valid query, output: MATCH (n) RETURN n LIMIT 0

The question is:
{question}

Generate the Cypher query now:"""

QA_PROMPT = """Use the following pieces of context retrieved from a Neo4j graph database to answer the question at the end.

If you don't know the answer or if the context is insufficient, just say that you don't have enough information based on the retrieved data. Don't try to make up an answer.

Be concise and direct. Format your answer in a clear, readable way. If the context contains structured data (like lists or multiple records), present it in an organized manner.

Context:
{context}

Question: {question}

Answer:"""
