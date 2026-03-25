"""
Orchestrator

Main entry point called by fixtures_services.py every 5 balls.
Ties everything together:
    1. Generate valid questions (question_gen + ml_filter)
    2. For each question, run the LangGraph (graph_nodes)
    3. Collect results: {sentence, query, answer, error}
"""

from app.services.llm._config import config, logger
from app.services.llm.ml_filter import generate_valid_statements


def run_statements(narrative, game_state_dict, graph, metadata):
    """
    Generate questions from narrative and execute SQL for each.

    Called by fixtures_services.py every 5 balls.

    Args:
        narrative: Current match narrative text
        game_state_dict: Dict with 11 game state features
        graph: Compiled LangGraph (from graph_nodes.py)
        metadata: Table/column metadata (from prompt.py)

    Returns:
        List of dicts: [{sentence, query, answer, error}, ...]
        Saved to CSV by fixtures_services.py
    """
    valid_statements = generate_valid_statements(narrative, game_state_dict)
    outputs = []

    logger.info("=" * 70)
    logger.info(f"Running {len(valid_statements)} validated statements")
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
            input_state = {
                "question": sentence,
                "query": "",
                "result": "",
                "answer": "",
                "table_info": metadata,
                "dialect": config.db.DIALECT
            }

            final_state = graph.invoke(input_state)

            item["query"] = final_state.get("query", "")
            item["answer"] = final_state.get("answer", "")

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
