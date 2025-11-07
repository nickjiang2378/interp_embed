#!/usr/bin/env python3
"""
Utility functions for SAE hypothesis generation.
"""
from typing import List, Optional, Dict, Union
import json
import os
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
import numpy as np
import pandas as pd
from dataclasses import dataclass
# Pydantic models for feature labeling and scoring
class FeatureLabelingRequest(BaseModel):
    positive_samples: List[str] = Field(
        ..., description="Samples where the feature activated, with << >> markers around activated tokens."
    )
    negative_samples: List[str] = Field(
        ..., description="Samples where the feature did not activate, with << >> markers (should be absent or empty)."
    )


class FeatureLabelingResponse(BaseModel):
    brief_description: str | None = Field(
        ..., description="1-2 sentence description of what this feature detects"
    )
    detailed_explanation: str | None = Field(
        ..., description="Longer explanation of what this feature lights up on, referencing positive and negative samples"
    )


class SingleSampleScoringRequest(BaseModel):
    feature_description: str = Field(
        ..., description="The description of what the feature should detect"
    )
    sample: str = Field(
        ..., description="A single sample to score, with << >> markers if positive, or no markers if negative."
    )
    sample_type: str = Field(
        ..., description="Whether this is a positive or negative sample."
    )


class SingleSampleScoringResponse(BaseModel):
    score: int = Field(
        ..., description="1 if the sample matches the feature description as expected, 0 otherwise."
    )
    explanation: str = Field(
        ..., description="Explanation for the score."
    )

@dataclass
class Hypothesis:
    """Represents a generated hypothesis about dataset differences."""
    description: str
    examples: List[str]
    percentage_difference: float
    feature_ids: List[int]
    confidence: float


def extract_json_from_response(content: str) -> str:
    """
    Extract JSON from LLM response content, handling markdown code blocks.

    Args:
        content: The raw response content from the LLM

    Returns:
        The extracted JSON string
    """
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        json_str = content[json_start:json_end].strip()
    elif "```" in content:
        json_start = content.find("```") + 3
        json_end = content.find("```", json_start)
        json_str = content[json_start:json_end].strip()
    else:
        json_str = content.strip()

    return json_str


def ensure_prompts_list(prompts: Optional[List[str]], samples_length: int) -> List[str]:
    """
    Ensure prompts is a list of the correct length, defaulting to empty strings if None.

    Args:
        prompts: Optional list of prompts
        samples_length: Expected length of the prompts list

    Returns:
        List of prompts with the correct length
    """
    if prompts is None:
        return [""] * samples_length
    return prompts




def get_average_score(result: dict) -> float:
    """
    Calculate the average score from a result dictionary.

    Args:
        result: Dictionary containing 'total_score' and 'total_count'

    Returns:
        Average score, or 0.0 if total_count is 0
    """
    if result["total_count"] == 0:
        return 0.0
    return result["total_score"] / result["total_count"]


def build_gpt4_labeling_prompt(positive_samples: List[str], negative_samples: List[str],
                              prompts: List[str] = None,
                              feature_score: float = None, current_label: str = None) -> str:
    """Build prompt for feature labeling, pairing each positive and negative sample on the same line."""

    # Default to empty prompts if not provided
    prompts = ensure_prompts_list(prompts, len(positive_samples))

    # Pair up positive and negative samples for side-by-side comparison
    n = len(positive_samples)
    assert len(positive_samples) == len(negative_samples), f"Positive samples doesn't match negative samples length, {len(positive_samples) != len(negative_samples)}"
    paired_lines = []
    for i in range(n):
        prompt_info = ""
        if i < len(prompts) and prompts[i]:
            prompt_info = f"\n  PROMPT: {prompts[i]}"

        paired_lines.append(
            f"Pair {i+1}:{prompt_info}\n  POSITIVE (feature activated, << >> marks the activated tokens): {positive_samples[i]}\n  NEGATIVE (feature did NOT activate, no << >> markers): {negative_samples[i]}"
        )

    pairs_text = "\n\n".join(paired_lines)

    # Add refinement context if score is low
    refinement_context = ""
    if current_label is not None:
        refinement_context = f"""
REFINEMENT CONTEXT:
The current label for this feature is: "{current_label}".\n
"""
        if feature_score is not None and feature_score < 0.75:
            refinement_context += f"""
However, this label scored only {feature_score:.2f} when tested on samples, indicating it may not accurately capture what the feature detects.
Please provide a more accurate description based on the samples below.
"""
        else:
            refinement_context += f"""
This label scored {feature_score:.2f} when tested on samples, suggesting that it accurately captures what the feature detects.
Please refine the label based on the samples below.
"""

    prompt = f"""You are an expert at interpreting features from sparse autoencoders (SAEs) for language models.
{refinement_context}
Below are {len(positive_samples)} POSITIVE samples (where the feature activated, with tokens surrounded by << and >>) and {len(negative_samples)} NEGATIVE samples (where it did not activate, no << >> markers). Each pair is shown together for comparison.

For each pair, the POSITIVE sample contains tokens that caused the feature to activate (marked with << >>), while the NEGATIVE sample does not. When a PROMPT is provided, it shows the user input that generated the response.

IMPORTANT NOTES:
1. The << >> markers indicate where the feature activated, but you should NOT restrict your understanding to just those marked tokens. Look at the context BEFORE the marked tokens as well - the preceding tokens often provide crucial information about what the feature is detecting.
2. The feature may be responding to a pattern or concept that spans both the marked tokens AND the tokens before the marked token.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.

PAIRED SAMPLES:
{pairs_text}

Your task:
- Carefully compare the POSITIVE and NEGATIVE samples in each pair.
- Look at BOTH the tokens before the << >> markers AND the marked tokens themselves to understand what the feature is detecting.
- Identify the most specific and concise property that is present in the POSITIVE samples (considering both context and marked tokens), but absent in the NEGATIVE samples.
- Try to give a unified property that isn't just a list of properties, if possible.
- Summarize the common attribute or property that causes the feature to activate. Be as specific as possible, but keep your description concise and clear.
- Do not reference specific pair IDs in either the brief description or the detailed explanation.

Return your answer as a JSON object with exactly these fields:
- "brief_description": "A concise sentence describing the property present in the positive samples (considering both context and marked tokens) but not in the negative samples. You can phrase it as, the feature is detecting X, etc."
- "detailed_explanation": "An extended explanation of what this featudre is detecting, including how the context before the marked tokens contributes to the feature's meaning. The explanation should be sufficient on its own to understand what the feature detects. Keep it to <5 concise sentences."

Make sure your response is valid JSON that can be parsed directly.
"""
    return prompt


def build_single_sample_prompt(feature_description: str, positive_sample: str, negative_sample: str,
                              prompt: str = "") -> str:
    """Build prompt for single sample scoring, comparing positive and negative samples."""

    # Build prompt section
    prompt_section = ""
    if prompt:
        prompt_section = f"\nPROMPT (user input that generated the responses):\n{prompt}\n"

    prompt = f"""You are an expert at evaluating sparse autoencoder feature descriptions.

You are given a feature description, a POSITIVE sample (where the feature activated, with tokens surrounded by << and >>), and a NEGATIVE sample (where the feature did NOT activate, and there should be no << >> markers).

IMPORTANT NOTES:
1. The << >> markers indicate where the feature activated, but you should NOT restrict your understanding to just those marked tokens. Look at the context BEFORE the marked tokens as well - the preceding tokens often provide crucial information about what the feature is detecting.
2. The feature may be responding to a pattern or concept that spans both the context tokens AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
4. You shouldn't be trying to infer what the feature description should be from the positive and negative samples; rather, you should use the feature description to evaluate the samples.

Feature description:
"{feature_description}"
{prompt_section}
POSITIVE SAMPLE (feature activated, << >> marks WHERE it activated):
{positive_sample}

NEGATIVE SAMPLE (feature did NOT activate, no << >> markers):
{negative_sample}

Your task:
- Evaluate if the feature description accurately describes whether or not the feature activates, considering BOTH the context before the << >> markers AND the marked tokens themselves to understand what triggered the feature
- Score 1 if the property described by the feature description is clearly present in the positive sample (considering both context and marked tokens) and absent in the negative sample.
- Score 0 if the property described by the feature description is not clearly present in the positive sample, or if the negative sample also contains the property. If the feature description is not a valid property (ex. "feature_#"), mark 0.

Return your answer as a JSON object with exactly these fields:
- "explanation": "<brief explanation for the score, focusing on how the context and marked tokens together show the difference between samples>"
- "score": <0 or 1>

Make sure your response is valid JSON that can be parsed directly. Keep the explanation brief (1-2 sentences)."""
    return prompt


def build_middle_out_batch_prompt(batch: List[Dict], query: str, batch_index: int, total_batches: int) -> str:
    """Build prompt for summarizing a batch of features in middle-out approach."""
    prompt = f"""Summarize these feature differences for the query: "{query}"
Features batch {batch_index+1}/{total_batches}:

"""
    for f in batch:
        prompt += f"""Feature ID: {f['feature_id']}
Description: {f['description']}
Prompt used to generate both responses below: {f['prompt']}

Positive Example (feature activated at tokens marked with << >>, but consider preceding context too): {f['example_positive']}

Negative Example (no feature activation): {f['example_negative']}
Difference Strength: {f['difference']:.2f}

"""

    prompt += """\nProvide a detailed, clear summary of the key patterns relevant to the query. Cluster similar feature descriptions together and update their difference strengths. You should not leave out any feature descriptions in the summary, but if needed, you should prioritize features with a greater difference strength. For each cluster, create a feature description that captures the pattern (start with 'this response detects..'), have a difference strength that is max of the features in the cluster, and a list of feature IDs that are part of the cluster.

    IMPORTANT NOTES:
    1. You should prioritize features with a greater difference strength.
    2. For each cluster, create a feature description that captures the pattern (start with 'this response detects..'), have a difference strength that is max of the features in the cluster, and a list of feature IDs that are part of the cluster. A cluster could just be a single feature.
    3. The << >> markers in examples indicate WHERE features activated, but you should NOT restrict your understanding to just those marked tokens. The context BEFORE the marked tokens often provides crucial information about what the feature is detecting.
    4. Features often respond to patterns that span both the preceding context AND the marked tokens together.
    6. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
    7. Note that some features are not accurate. If the feature description does not accurately describe the tokens marked with << >>, you should disregard the feature. Only use features that you are certain are valid.
    8. Excluding inaccurate or duplicate features and features clearly irrelevant to the query, you should not leave out any feature descriptions in the summary.

    Generate the best hypotheses that answer the user's query. Each hypothesis should be formatted as a JSON object with these exact fields:
    - "description": Describe a response that would validly have property X. Start with "This response .." Use 1-2 sentences to clearly and specifically describe the property, such that using this description could be used to identify the property on its own. Do not mention the model names. Be specific so that responses that don't have this property could not be misclassified as having this property based on this description.
    - "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID, or a list of multiple feature IDs. List at most five feature IDs.
    - "examples": List of examples. Be concise. For each example, cite the feature ID and feature description and explain how the positive / negative example pairs from the dataset illustrate the hypothesis, considering both the marked tokens AND their preceding context). You should just highlight the portion of the example pairs that are relevant for the feature; do not print out the entire positive / negative example pairs unless it is necessary to understand the feature.
    - "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Use the maximum difference strength among the features used. Positive percentage if target has more of this property, negative otherwise.
    - "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)

    """
    return prompt


HYPOTHESIS_GENERATION_START = """You are analyzing differences between two datasets. Below are the most significant features that are differences between a "target" and "other" dataset:

IMPORTANT NOTES:
1. The << >> markers in examples indicate WHERE features activated, but you should NOT restrict your understanding to just those marked tokens. The context BEFORE the marked tokens often provides crucial information about what the feature is detecting.
2. Features often respond to patterns that span both the preceding context AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
4. Note that some features are not accurate. If the feature description does not accurately describe the tokens marked with << >>, you should disregard the feature. Only use features that you are certain are valid.
5. Please ensure that all hypothesis descriptions are clearly distinct from each other. You do not need to generate the exact amount of hypotheses to meet the quota.
6. Each feature will have a "difference strength", which is the percentage difference between the target and other dataset. If it is positive, the target dataset has more of the feature than the other dataset. If it is negative, the other dataset has more of the feature than the target dataset.
7. Please try to make each hypothesis specific and focused, having one main theme or idea.
"""


def build_middle_out_final_prompt(batch_summaries: List[Dict], query: str, num_hypotheses: int, total_features: int, both_directions: bool) -> str:
    """Build final prompt for hypothesis generation from batch summaries."""
    import json
    prompt = f"""{HYPOTHESIS_GENERATION_START}\n\nBased on these feature pattern summaries, generate {num_hypotheses} hypotheses about: "{query}"

You will be given summaries of all the features that were analyzed. You should use these summaries to generate the best hypotheses. Each summary will be of the form:
- "description": The textual property described by the feature.
- "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID, or a list of multiple feature IDs.
- "examples": List of examples. Token activations are shown with << >> markers. Note that the feature may be responding to a pattern or concept that spans both the preceding context AND the marked tokens together.
- "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Positive percentage if the target dataset has more of this property, negative otherwise.
- "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)

Summaries from analyzing {total_features} total features:
{json.dumps(batch_summaries, indent=2)}

USER QUERY: {query}

IMPORTANT NOTES:
1. The << >> markers in examples indicate WHERE features activated, but you should NOT restrict your understanding to just those marked tokens. The context BEFORE the marked tokens often provides crucial information about what the feature is detecting.
2. Features often respond to patterns that span both the preceding context AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
4. Note that some features are not accurate. If the feature description does not accurately describe the tokens marked with << >>, you should disregard the feature. Only use features that you are certain are valid.
5. Please ensure that all hypothesis descriptions are clearly distinct from each other. You do not need to generate the exact amount of hypotheses to meet the quota.
6. Each feature will have a "difference strength", which is the percentage difference between the target and other dataset. If it is positive, the target dataset has more of the feature than the other dataset. If it is negative, the other dataset has more of the feature than the target dataset.
7. Please try to make each hypothesis specific and focused, having one main theme or idea.

Generate at most {num_hypotheses} hypotheses that answer the user's query{', including both "target" and "other" properties' if both_directions else ' for the "target" dataset'}. Only give the best hypotheses that answer the user's query. Using the summaries, you can merge hypotheses, pick certain hypotheses, or even create new hypotheses based on the summaries. I'm looking for differences of the format Dataset A{'/B' if both_directions else ''} is more X than Dataset {'B/A' if both_directions else 'B'}, where X is the difference. Each hypothesis should be formatted as a JSON object with these exact fields:
- "dataset": "target" or "other" (the dataset that has more of this property)
- "description": Describe a response that would validly have property X. Start with "This response .." Use 1-2 sentences to clearly and specifically describe the property, such that using this description could be used to identify the property on its own. Do not mention the model names. Be specific so that responses that don't have this property could not be misclassified as having this property based on this description.
- "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID, or a list of multiple feature IDs. List at most five feature IDs.
- "examples": List of examples. Provide at most 3 examples. Be concise. For each example, cite the feature ID and feature description and explain how the positive / negative example pairs from the dataset illustrate the hypothesis, considering both the marked tokens AND their preceding context). You should just highlight the portion of the example pairs that are relevant for the feature; do not print out the entire positive / negative example pairs unless it is necessary to understand the feature.
- "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Use the maximum difference strength among the features used. Positive percentage if target has more of this property, negative otherwise.
- "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)

Remember that <eot_id> tokens should be ignored as they are just EOS markers, not meaningful feature activations.

Return the response as a JSON array of at most {num_hypotheses} hypothesis objects. Make sure the JSON is valid and can be parsed directly."""
    return prompt


def build_hypotheses_prompt(significant_features: List[Dict], query: str, format: str = "hypotheses", num_hypotheses: int = 10, both_directions: bool = True) -> str:
    """Build prompt for hypothesis generation from features."""

    # Build the initial prompt with both positive and negative examples
    prompt = HYPOTHESIS_GENERATION_START
    for feature in significant_features:
        # Add prompt information if available
        prompt_info = ""
        if feature.get('prompt'):
            prompt_info = f"Prompt: {feature['prompt']}\n"

        prompt += f"""Feature ID: {feature['feature_id']}
Description: {feature['description']}
{prompt_info}Positive Example (feature activated at tokens marked with << >>, but consider preceding context too): {feature['example_positive']}
Negative Example (no feature activation): {feature['example_negative']}
Confidence about difference strength: {feature['feature_confidence']:.2f}
Difference Strength: {feature['difference']:.2f}

"""

    # Add format-specific instructions
    if format == "paragraph":
        prompt += f"""Based on the above differences, answer this specific query: {query}

Create a concise paragraph that answers this query based on the differences between the two datasets. Do not include mention of an "assistant" or "user", just focus on the other parts of the feature. When explaining a difference, cite the corresponding feature using the format <FEATURE feature_id>; you should put them in parentheses, just like you'd cite a source. Where relevant, you should refer to the datasets as <DATASET 1> and <DATASET 2>. Be specific about the differences. Remember that the << >> markers show WHERE features activated, but the feature's meaning often depends on the preceding context as well. <eot_id> tokens should be ignored as they are just EOS markers."""
    else:  # format == "hypotheses"
        prompt += f"""USER QUERY: {query}

Generate at most {num_hypotheses} distinct hypotheses that answer the user's query{', including both "target" and "other" properties' if both_directions else ' for the "target" dataset'}. I'm looking for differences of the format Dataset A{'/B' if both_directions else ''} is more X than Dataset {'B/A' if both_directions else 'B'}, where X is the difference. Each hypothesis should be formatted as a JSON object with these exact fields:
- "dataset": "target" or "other" (the dataset that has more of this property)
- "description": Describe a response that would validly have property X. Start with "This response .." Use 1-2 sentences to clearly and specifically describe the property, such that using this description could be used to identify the property on its own. Do not mention the model names. Be specific so that responses that don't have this property could not be misclassified as having this property based on this description.
- "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID, or a list of multiple feature IDs.
- "examples": List of examples. Provide at most 3 examples. Be concise. For each example, cite the feature ID and feature description and explain how the positive / negative example pairs from the dataset illustrate the hypothesis, considering both the marked tokens AND their preceding context). You should just highlight the portion of the example pairs that are relevant for the feature; do not print out the entire positive / negative example pairs unless it is necessary to understand the feature.
- "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Use the maximum difference strength among the features used. Positive percentage if target has more of this property, negative otherwise.
- "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)

Remember that <eot_id> tokens should be ignored as they are just EOS markers, not meaningful feature activations.

Return the response as a JSON array of at most {num_hypotheses} DISTINCT hypothesis objects. Make sure the JSON is valid and can be parsed directly."""

    return prompt

def limit_feature_differences(significant_diffs: pd.DataFrame, max_feature_diffs: Optional[int], both_directions: bool) -> pd.DataFrame:
    """
    Limit the number of feature differences to analyze based on max_feature_diffs parameter.

    Args:
        significant_diffs: DataFrame with 'feature_id' and 'diff_activation' columns
        max_feature_diffs: Maximum number of features to analyze (None for no limit)
        both_directions: Whether to split budget between positive and negative differences

    Returns:
        Limited DataFrame of feature differences
    """
    if max_feature_diffs is None or len(significant_diffs) <= max_feature_diffs:
        return significant_diffs

    if both_directions:
        # Split budget between positive and negative differences
        positive_diffs = significant_diffs[significant_diffs["diff_activation"] > 0]
        negative_diffs = significant_diffs[significant_diffs["diff_activation"] < 0]

        # Allocate half budget to each direction
        budget_per_direction = max_feature_diffs // 2
        remaining_budget = max_feature_diffs % 2

        # Take top features from each direction
        positive_limited = positive_diffs.head(budget_per_direction + remaining_budget)
        negative_limited = negative_diffs.head(budget_per_direction)

        # Combine and sort by absolute difference
        significant_diffs = pd.concat([positive_limited, negative_limited])
        significant_diffs = significant_diffs.sort_values("diff_activation", ascending=False, key=abs)
    else:
        # Simple limit for single direction
        significant_diffs = significant_diffs.head(max_feature_diffs)

    print(f"Limited to {len(significant_diffs)} features (max_feature_diffs={max_feature_diffs})")
    return significant_diffs


def diff_features_multi(target_activations, other_activations, feature_activation_type="max", min_coverage=0.0, max_coverage=1.0):
    """
    Calculate the difference in feature activations between one target dataset and multiple other datasets.
    Computes the max activation across all other datasets for each feature.

    :param target_activations: The target activations
    :param other_activations: List of other activations to compare against
    :param feature_activation_type: Type of feature activation ('max', 'mean', or 'sum')
    :param metric: Metric to use for calculating the difference ('absolute', 'relative')
    :param min_coverage: Minimum percentage of samples that must have a non-zero activation to consider a feature
    :param max_coverage: Maximum percentage of samples that must have a non-zero activation to consider a feature
    :return: DataFrame with feature_id and diff_activation columns
    """

    # Get feature activations for target dataset
    target_nonzero_F = np.count_nonzero(target_activations, axis=0) / target_activations.shape[0]

    # Initialize arrays to store max activations across other datasets
    n_features = target_activations.shape[1]

    # Compute max activation percentage across all other datasets
    other_activations_max_DF = other_activations.max(axis = 0)
    other_activations_all_DF = np.all(other_activations, axis = 0)
    other_max_nonzero_F = np.count_nonzero(other_activations_max_DF, axis=0) / other_activations_max_DF.shape[0]
    other_all_nonzero_F = np.count_nonzero(other_activations_all_DF, axis=0) / other_activations_all_DF.shape[0]

    # Apply coverage filters
    feature_mask = (np.minimum(target_nonzero_F, other_max_nonzero_F) < min_coverage) | \
                   (np.maximum(target_nonzero_F, other_max_nonzero_F) > max_coverage)

    target_nonzero_F = target_nonzero_F.copy()
    other_max_nonzero_F = other_max_nonzero_F.copy()
    target_nonzero_F[feature_mask] = -1
    other_max_nonzero_F[feature_mask] = -1

    # Calculate differences
    target_diff_others_F = target_nonzero_F - other_max_nonzero_F
    others_diff_target_F = other_all_nonzero_F - target_nonzero_F

    # Create DataFrame with results
    feature_ids = list(range(n_features))
    diffs_df = pd.DataFrame({
        'feature_id': feature_ids,
        'target_diff_others': target_diff_others_F,
        'others_diff_target': others_diff_target_F,
        'target_dataset_coverage': target_nonzero_F,
        'other_datasets_max_coverage': other_max_nonzero_F,
        "other_datasets_all_coverage": other_all_nonzero_F
    })

    return diffs_df
