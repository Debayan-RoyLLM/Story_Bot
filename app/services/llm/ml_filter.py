"""
ML-Based Question Classification

Filters generated questions using BERT embeddings + PCA + ML ensemble.
No LLM calls here — pure ML inference.

Pipeline per question:
    1. BERT embedding (768-dim)
    2. PCA reduction (768 → 59-dim)
    3. Concatenate with game state (59 + 11 = 70-dim)
    4. Classify with Logistic Regression, Random Forest, XGBoost
    5. Keep if Random Forest probability > 0.5
"""

import numpy as np
import torch
import joblib
from transformers import BertTokenizer, BertModel

from app.services.llm._config import config, logger
from app.services.llm.question_gen import generate_questions, stat_questions


# ── Load BERT ────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(config.ml.BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(config.ml.BERT_MODEL_NAME)

# ── Load ML Models ───────────────────────────────────────────────
pca_model = None
logistic_model = None
random_forest_model = None
xgb_model = None

try:
    pca_model = joblib.load(config.ml.PCA_MODEL_PATH)
    logistic_model = joblib.load(config.ml.LOGISTIC_MODEL_PATH)
    random_forest_model = joblib.load(config.ml.RANDOM_FOREST_MODEL_PATH)
    xgb_model = joblib.load(config.ml.XGB_MODEL_PATH)
    logger.info("ML models loaded successfully (BERT, PCA, Logistic, Random Forest, XGBoost)")
except FileNotFoundError as e:
    logger.warning(f"ML model files not found: {e}")
    logger.warning("ML-based question classification will be disabled")
except Exception as e:
    logger.error(f"Error loading ML models: {e}")
    logger.warning("ML-based question classification will be disabled")


def get_bert_embedding(text):
    """
    Convert text to 768-dimensional BERT embedding.

    Input:  "What is Kohli's strike rate?"
    Output: numpy array of 768 floats
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=config.ml.BERT_MAX_LENGTH)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


def classify_with_models(combined_input):
    """
    Run 70-dim input through ML ensemble.

    Input:  70-dim array (59 PCA + 11 game state)
    Output: {"Logistic Regression": [0.3, 0.7], "Random Forest": [0.4, 0.6], "XGBoost": [0.2, 0.8]}
            probability of [invalid, valid]
    """
    if not all([logistic_model, random_forest_model, xgb_model]):
        logger.warning("ML models not loaded, returning default probabilities")
        return {
            "Logistic Regression": [0.0, 1.0],
            "Random Forest": [0.0, 1.0],
            "XGBoost": [0.0, 1.0]
        }

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgb_model,
    }
    probabilities = {}
    for model_name, model in models.items():
        prob = model.predict_proba(combined_input.reshape(1, -1))[0]
        probabilities[model_name] = prob
    return probabilities


def process_sentences(sentences, game_state):
    """
    Run full ML pipeline on a list of sentences.

    For each sentence:
        1. BERT embedding (768-dim)
        2. PCA reduce (768 → 59)
        3. Concatenate with game state (59 + 11 = 70)
        4. Classify with 3 models

    Returns: list of {"sentence": str, "probabilities": dict}
    """
    results = []
    for sentence in sentences:
        embedding = get_bert_embedding(sentence)

        if pca_model is not None:
            reduced_embedding = pca_model.transform(embedding.reshape(1, -1)).flatten()
        else:
            reduced_embedding = embedding[:config.ml.PCA_DIMENSIONS]

        combined_input = np.concatenate([np.array(game_state), reduced_embedding])
        probabilities = classify_with_models(combined_input)

        results.append({
            "sentence": sentence,
            "probabilities": probabilities
        })
    return results


def generate_valid_statements(narrative, game_state_dict):
    """
    Generate questions and let the ML classifier decide how many are valid.

    Steps:
        1. LLM #1: generate 8 questions from narrative
        2. LLM #2: refine questions
        3. ML pipeline: filter by Random Forest > threshold
        4. Return however many pass — could be 1, 4, 8, etc.

    Returns: list of {"sentence": str, "probabilities": dict}
    """
    raw_questions = generate_questions(narrative)
    sentences = stat_questions(raw_questions)

    keys = config.statement.GAME_STATE_KEYS
    game_state = {k: game_state_dict.get(k, 0) for k in keys}
    game_state_array = np.array([game_state[k] for k in keys], dtype=float)
    results = process_sentences(sentences, game_state_array)

    # predict_proba returns [P(invalid), P(valid)] — use index [1] for valid class
    valid_statements = [
        result for result in results
        if result["probabilities"].get("Random Forest", [0, 0])[1] > config.statement.RANDOM_FOREST_THRESHOLD
    ]

    logger.info(f"ML classifier: {len(valid_statements)}/{len(results)} questions passed (threshold: {config.statement.RANDOM_FOREST_THRESHOLD})")

    return valid_statements
