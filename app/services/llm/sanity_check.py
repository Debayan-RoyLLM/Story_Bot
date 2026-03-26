"""
Result Sanity Checker

Validates SQL query results against cricket domain constraints.
Catches impossible values before they reach the answer generation LLM.

Handles:
    - Single-value results (e.g., economy rate = 8.31)
    - Multi-row results (e.g., GROUP BY fixture_id returning per-match rows)
    - Per-player stricter bounds (e.g., a bowler can't take 12 wickets in a match)
    - Mismatch between question intent ("total") and result shape (multiple rows)
"""

import re
import logging

logger = logging.getLogger(__name__)


# Realistic upper bounds for cricket stats (T20 context)
SANITY_LIMITS = {
    "score": 500,           # Max innings score (even ODI rarely exceeds 500)
    "runs": 500,
    "chase": 500,
    "total": 500,
    "wickets": 10,          # Max 10 wickets per innings
    "strike_rate": 700,     # Highest T20 SR ~400, leave margin
    "average": 500,         # Batting avg rarely above 100
    "economy": 50,          # Economy rate rarely above 20
    "percentage": 100,      # Can't exceed 100%
    "count": 50000,         # Sanity cap for match counts
}

# Per-player limits (stricter — applies when question is about a specific player)
PER_PLAYER_MATCH_LIMITS = {
    "wickets": 10,          # A bowler can take max 10 wickets in a match
}


def _extract_numeric_values(result_text):
    """
    Extract all numeric values from a result string.
    Handles both single values and multi-row results like "(75,)\\n(30,)".
    Returns list of floats.
    """
    values = []
    for line in result_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("..."):
            continue
        # Try direct float parse (single value)
        try:
            values.append(float(line.replace(",", "").strip()))
            continue
        except (ValueError, TypeError):
            pass
        # Try extracting numbers from tuple-like rows: (75,) or (3, 'name')
        nums = re.findall(r'(?<![a-zA-Z_])(-?\d+\.?\d*)(?![a-zA-Z_])', line)
        for n in nums:
            try:
                values.append(float(n))
            except (ValueError, TypeError):
                pass
    return values


def _is_player_query(question):
    """
    Detect if the question is about a specific player (not a team).
    Uses role keywords and possessive patterns (e.g., "Rohit's economy").
    """
    question_lower = question.lower()
    role_keywords = ["bowler", "batsman", "player", "batter", "fielder", "all-rounder"]
    if any(kw in question_lower for kw in role_keywords):
        return True
    # Possessive pattern: "PlayerName's" (e.g., "Cummins's", "Markram's")
    if re.search(r"\b[A-Z][a-z]+(?:'s|\u2019s)\b", question):
        return True
    return False


def _check_value_limits(value, question_lower, is_player):
    """Check a single numeric value against cricket domain limits."""

    # Negative values for counts/totals are always wrong
    count_keywords = ["dot ball", "wicket", "boundar", "runs", "score", "fours", "sixes"]
    if value < 0 and any(kw in question_lower for kw in count_keywords):
        return (
            f"Result {value} is negative for a count/total metric, which is impossible. "
            f"Likely cause: incorrect arithmetic in the query (e.g., computing dot balls "
            f"via subtraction instead of COUNT WHERE score__runs = 0)."
        )

    # Check against main limits
    for keyword, limit in SANITY_LIMITS.items():
        if keyword in question_lower and abs(value) > limit:
            return (
                f"Result {value} is unrealistic for '{keyword}' "
                f"(expected max ~{limit}). "
                f"Likely cause: query is aggregating across multiple matches "
                f"without proper GROUP BY fixture_id, or missing WHERE filters."
            )

    # Per-player stricter checks — scale limit by number of matches mentioned
    if is_player:
        # Try to detect "last N matches" from question
        match_count_match = re.search(r'last\s+(\d+)', question_lower)
        match_multiplier = int(match_count_match.group(1)) if match_count_match else 5

        for keyword, limit in PER_PLAYER_MATCH_LIMITS.items():
            if keyword in question_lower and abs(value) > limit * match_multiplier:
                return (
                    f"Result {value} is unrealistic for a single player's '{keyword}' "
                    f"(expected max ~{limit} per match, ~{limit * match_multiplier} across {match_multiplier} matches). "
                    f"Likely cause: query is summing across all players or all matches "
                    f"without filtering to the specific player."
                )

    # Run rate / economy — max possible is 36 (6 runs × 6 balls per over)
    if ("run rate" in question_lower or "economy" in question_lower) and value > 36:
        return (
            f"Result {value} is unrealistic for a run/economy rate "
            f"(max possible ~36). Likely dividing by very small overs value."
        )

    # Batting average — below 1.0 is almost certainly a query error
    if ("average" in question_lower or "batting average" in question_lower) and 0 < value < 1:
        return (
            f"Result {value} is unrealistically low for a batting average. "
            f"Likely cause: division error or wrong columns in calculation."
        )

    # Large count check
    count_keywords = ["how many", "how often", "instances", "times"]
    if any(kw in question_lower for kw in count_keywords) and value > 5000:
        return (
            f"Result {value} is suspiciously large for a match count. "
            f"Likely missing inning filter or counting all balls instead of matches."
        )

    # Generic absurdly large
    if abs(value) > 100000:
        return (
            f"Result {value} is unrealistically large. "
            f"Likely aggregating across all matches without GROUP BY fixture_id."
        )

    return None


def _check_row_count_mismatch(result_text, question_lower):
    """Check if question expects a single total but got multiple rows."""
    lines = [l for l in result_text.strip().split("\n")
             if l.strip() and not l.strip().startswith("...")]

    if len(lines) <= 1:
        return None

    total_keywords = ["total", "overall", "combined", "aggregate", "sum"]
    matching = [kw for kw in total_keywords if kw in question_lower]

    if matching:
        return (
            f"Question asks for a '{matching[0]}' but query returned {len(lines)} rows "
            f"instead of one. Likely missing a final SUM() aggregation — the query has "
            f"GROUP BY that returns per-match rows instead of a single total."
        )

    return None


def check_result_sanity(result_text, question):
    """
    Check if the SQL result is realistic for cricket data.
    Handles both single-value and multi-row results.

    Returns:
        (is_sane, reason) — reason explains why it's unrealistic.
    """
    if not result_text or result_text == "No results returned":
        return True, ""

    values = _extract_numeric_values(result_text)
    if not values:
        return True, ""

    question_lower = question.lower()
    is_player = _is_player_query(question)

    # Check each extracted value
    for value in values:
        reason = _check_value_limits(value, question_lower, is_player)
        if reason:
            logger.warning(f"Sanity check FAILED: {reason}")
            return False, reason

    # Check row count vs question intent
    reason = _check_row_count_mismatch(result_text, question_lower)
    if reason:
        logger.warning(f"Sanity check FAILED: {reason}")
        return False, reason

    return True, ""
