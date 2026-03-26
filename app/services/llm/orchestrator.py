"""
Orchestrator

Main entry point called by fixtures_services.py every 5 balls.
Ties everything together:
    1. Generate valid questions (question_gen + ml_filter)
    2. For each question, run the LangGraph (graph_nodes)
    3. Collect results: {sentence, query, answer, error}
"""

from app.services.llm._config import config, logger
from app.services.llm._cache import clear_cache
from app.services.llm.ml_filter import generate_valid_statements


def run_statements(narrative, game_state_dict, graph, metadata, fixture_id=None):
    """
    Generate questions from narrative and execute SQL for each.

    Called by fixtures_services.py every 5 balls.

    Args:
        narrative: Current match narrative text
        game_state_dict: Dict with 11 game state features
        graph: Compiled LangGraph (from graph_nodes.py)
        metadata: Table/column metadata (from prompt.py)
        fixture_id: Current match fixture_id (scopes queries to this match)

    Returns:
        List of dicts: [{sentence, query, answer, error}, ...]
        Saved to CSV by fixtures_services.py
    """
    # Clear cache each batch — context changes between balls
    clear_cache()

    valid_statements = generate_valid_statements(narrative, game_state_dict)
    outputs = []

    logger.info("=" * 70)
    logger.info(f"Running {len(valid_statements)} validated statements")
    if fixture_id:
        logger.info(f"Current match fixture_id: {fixture_id}")
    logger.info("=" * 70)

    for idx, stmt in enumerate(valid_statements, 1):
        sentence = stmt["sentence"]

        logger.info(f"[{idx}/{len(valid_statements)}] Processing: {sentence[:60]}...")

        item = {
            "sentence": sentence,
            "query": None,
            "answer": None,
            "error": None
        }

        try:
            # Prepend fixture_id context — LLM decides when to use it
            question_with_context = sentence
            if fixture_id:
                question_with_context = (
                    f"[Current match fixture_id = {fixture_id}.\n"
                    f"Use WHERE fixture_id = {fixture_id} ONLY when the question asks about "
                    f"'this match', 'current match', 'ongoing match', 'today', or 'current innings'.\n"
                    f"For historical questions ('in T20 history', 'overall', 'all matches', "
                    f"'how many times', 'highest ever', 'last N matches/seasons'), "
                    f"do NOT filter by fixture_id — query across all matches.]\n"
                    f"{sentence}"
                )

            input_state = {
                "question": question_with_context,
                "query": "",
                "result": "",
                "answer": "",
                "table_info": metadata,
                "dialect": config.db.DIALECT,
                "din_context": "",
                "query_warnings": [],
            }

            final_state = graph.invoke(input_state)

            item["query"] = final_state.get("query", "")
            item["answer"] = final_state.get("answer", "")
            # Store original sentence (without fixture_id prefix) for CSV
            item["sentence"] = sentence

            if item["answer"] and item["answer"].startswith("Error"):
                item["error"] = item["answer"]
                item["answer"] = None

            if item["query"]:
                logger.debug(f"  Query: {item['query'][:70]}...")
            if item["answer"]:
                logger.info(f"  Answer: {item['answer'][:70]}...")
            else:
                logger.warning("  No answer generated")

        except Exception as e:
            logger.error(f"  Error: {str(e)}")
            item["error"] = str(e)
            item["answer"] = f"Error: {str(e)}"

        outputs.append(item)

    return outputs
