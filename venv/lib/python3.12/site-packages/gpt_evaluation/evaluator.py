import pandas as pd
from typing import Dict, Tuple, Any
import openai
from .metrics import EVALUATION_PROMPT_TEMPLATE
from .utils import normalize_score, calculate_weighted_accuracy

def evaluate(criteria: str, steps: str, query: str, document: str, response: str,
             metric_name: str, client: Any, model: str = "gpt-4o-mini") -> int:
    """
    Generates an evaluation score for a given query, document, response, and metric.
    
    Parameters:
        criteria (str): The evaluation criteria.
        steps (str): The evaluation steps.
        query (str): The user's query.
        document (str): The knowledge source document.
        response (str): The generated response by the chatbot.
        metric_name (str): The name of the metric being evaluated.
        client (Any): The OpenAI API client.
        model (str): The model to use for evaluation.
    
    Returns:
        int: The evaluation score.
    
    Raises:
        RuntimeError: If the API call fails.
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        query=query,
        document=document,
        response=response,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    except Exception as e:
        raise RuntimeError(f"Evaluation API call failed: {e}")
    
    return int(resp.choices[0].message.content.strip())

def evaluate_all(metrics: Dict[str, Tuple[str, str]], query: str, response: str,
                 document: str, client: Any, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """
    Performs evaluation for all metrics using a single query, response, and document.
    
    Parameters:
        metrics (dict): Dictionary mapping metric names to (criteria, steps).
        query (str): The user's query.
        response (str): The generated response.
        document (str): The knowledge source document.
        client (Any): The OpenAI API client.
        model (str): The model to use for evaluation.
    
    Returns:
        pd.DataFrame: A DataFrame with evaluation scores for each metric.
    """
    data = {"Evaluation Type": [], "Score": []}
    for metric_name, (criteria, steps) in metrics.items():
        data["Evaluation Type"].append(metric_name)
        score = evaluate(criteria, steps, query, document, response, metric_name, client, model)
        data["Score"].append(score)
    return pd.DataFrame(data).set_index("Evaluation Type")

def evaluate_response(query: str, response: str, document: str, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """
    High-level interface to evaluate a response using default metrics.
    
    This function accepts the user query, generated response, and the source document.
    Optionally, a model name ("gpt-4o" or "gpt-4o-mini" etc) can be provided.
    It returns a report as a pandas DataFrame with normalized scores and overall accuracy.
    
    Parameters:
        query (str): The user's query.
        response (str): The generated response by the chatbot.
        document (str): The knowledge source document.
        model (str): Optional. The OpenAI model to use (default "gpt-4o-mini").
    
    Returns:
        pd.DataFrame: Report containing metric scores.
    """
    # Import default metrics from the metrics module
    from .metrics import (
        QUERY_RELEVANCE_CRITERIA, QUERY_RELEVANCE_STEPS,
        FACTUAL_ACCURACY_CRITERIA, FACTUAL_ACCURACY_STEPS,
        COVERAGE_CRITERIA, COVERAGE_STEPS,
        COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS,
        FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS,
    )
    from .utils import generate_report
    
    # Define default evaluation metrics and their weights
    evaluation_metrics = {
        "Query Relevance": (QUERY_RELEVANCE_CRITERIA, QUERY_RELEVANCE_STEPS),
        "Factual Accuracy": (FACTUAL_ACCURACY_CRITERIA, FACTUAL_ACCURACY_STEPS),
        "Coverage": (COVERAGE_CRITERIA, COVERAGE_STEPS),
        "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
        "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
    }
    default_weights = [2, 2, 2, 1, 1]
    
    # Use OpenAI's API client (ensure your API key is set via environment or config)
    client = openai
    
    # Evaluate all metrics using the provided model
    eval_df = evaluate_all(evaluation_metrics, query, response, document, client, model)
    metric_names = list(evaluation_metrics.keys())
    scores = [eval_df.loc[metric, "Score"] for metric in metric_names]
    
    # Generate the final report using normalized and weighted scores
    report = generate_report(scores, default_weights, metric_names)
    return report
