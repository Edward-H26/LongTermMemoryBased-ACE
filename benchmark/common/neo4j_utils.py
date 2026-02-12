"""Neo4j database management helpers for CL-bench.

Provides ``clear_neo4j_all`` which wipes every node and relationship
in the configured database — used by ``run_v*.py`` and ACE inference
scripts when the ``--clear-db`` flag is set.
"""

import os


def clear_neo4j_all() -> bool:
    """Delete all nodes and relationships from the Neo4j database.

    Reads connection details from ``NEO4J_URI``, ``NEO4J_USERNAME``,
    ``NEO4J_PASSWORD``, and optionally ``NEO4J_DATABASE`` environment
    variables (falls back to ``NEXT_PUBLIC_*`` prefixed variants).

    Returns
    -------
    bool
        ``True`` on success.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE") or None

    driver = GraphDatabase.driver(uri, auth = (user, pwd))
    with driver.session(database = db) as session:
        result = session.run("MATCH (n) DETACH DELETE n")
        result.consume()
    driver.close()
    return True
