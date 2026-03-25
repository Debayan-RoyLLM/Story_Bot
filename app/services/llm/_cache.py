"""
Query Cache

Caches SQL queries by question text.
If the same question appears again, reuse the SQL — skip LLM generation.
Only caches successful queries (validated + executed without error).
"""

_query_cache = {}


def get_cached_query(question):
    """Return cached SQL for this question, or None."""
    return _query_cache.get(question)


def cache_query(question, sql):
    """Store a successful SQL query for reuse."""
    _query_cache[question] = sql


def clear_cache():
    """Clear all cached queries (e.g., between matches)."""
    _query_cache.clear()
