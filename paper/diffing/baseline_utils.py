"""
Utility functions and prompts for baseline difference analysis.

This module contains:
- System prompts for analysis
- JSON extraction and parsing utilities
- Common utility functions shared between baseline analyzers
"""

import json
from typing import Dict, List, Any, Union

# System prompt for pairwise analysis
PAIRWISE_SYSTEM_PROMPT = """You are an expert model behavior analyst. Your task is to meticulously compare two model responses to a given user prompt and identify unique qualitative properties belonging to one model but not the other. For each significant property, you must determine if it's more likely a **general trait** of the model or a **context-specific** behavior triggered by the current prompt.

**Prioritize conciseness and clarity in all your descriptions and explanations.** Aim for the most impactful information in the fewest words.

You will be provided with:
1.  **User Prompt:** The original prompt given to both models.
2.  **Model A Name:** The identifier for Model A.
3.  **Model A Response:** The response from Model A.
4.  **Model B Name:** The identifier for Model B.
5.  **Model B Response:** The response from Model B.

**Your Goal:**
Produce a JSON list of objects. Each object will represent a single distinct property observed in one model's response that is notably absent or different in the other's. Focus on identifying key areas of distinction, and the individual property observations in the output list (e.g., Model A's formal tone would be one entry, Model B's casual tone would be another related entry). As these are very common and easy to measure with heuristics, please do not include properties like "Model A is more concise than Model B". If applicatble, make sure to also include properties revolving around the models reasoning, interpretation of the prompt/intent, and potential reason for errors if they exist.

**Focus on Meaningful Properties:**
Prioritize properties that would actually influence a user's model choice or could impact the model's performance. This could include but is not limited to:
* **Capabilities:** Accuracy, completeness, technical correctness, reasoning quality, domain expertise
* **Style:** Tone, approach, presentation style, personality, engagement with the user, and other subjective properties that someone may care about for their own use
* **Error patterns:** Hallucinations, factual errors, logical inconsistencies, safety issues
* **User experience:** Clarity, helpfulness, accessibility, practical utility, response to feedback
* **Safety/alignment:** Bias, harmful content, inappropriate responses, and other safety-related properties
* **Tool use:** Use of tools to complete tasks and how appropriate the tool use is for the task
* **Thought Process:** Chain of reasoning, backtracking, interpretation of the prompt, self-reflection, etc.

**Avoid trivial differences** like minor length variations, basic formatting, or properties that don't meaningfully impact the models capability or the user's experience.

**Definitions:**
*   **General Trait:** Reflects a model's pattern of behavior across a distribution of prompts.
    *   *Think:* Could a model have this property in a different prompt from the one provided? If so, then it is general. If not, then it is context-specific.
*   **Context-Specific Difference:** If the property is a direct reaction to *this current prompt*, then it is context-specific.
    *   *Think:* Is this property a direct reaction to *this current prompt*? If so, then it is context specific. If not, then it is general.
*   **Impact:** How much does this property impact the user's experience?
    *   *Think:* Is this property a major factor in the user's experience? Would the average user care to know that this property exists?
    *   **Low:** Minor stylistic differences that most users wouldn't notice or care about
    *   **Medium:** Noticeable differences that might influence preference but aren't deal-breakers
    *   **High:** Significant differences that could strongly influence model choice (e.g., errors, major capability gaps, strong stylistic preferences)
*   **User Preference Direction:** Which type of user might prefer this property?
    *   *Think:* Does this property appeal to specific user types or use cases?
    *   **Capability-focused:** Users who prioritize accuracy, completeness, technical correctness
    *   **Experience-focused:** Users who prioritize style, tone, presentation, ease of use, or users who focus on very open-ended tasks
    *   **Neutral:** Property doesn't clearly favor one user type over another
    *   **Negative:** Property that most users would find undesirable (errors, poor quality, etc.)
*   **Contains Errors:** Does either model response contain errors?
    *   *Think:* Are there factual errors, hallucinations, or other strange or unwanted behavior?
*   **Unexpected Behavior:** Does the model's response contain highly unusual or concerning behavior? If true then a developer will analyze these responses manually.
    *   *Think:* Does this involve offensive language, gibberish, bias, factual hallucinations, or other strange or unwanted behavior?

**JSON Output Structure for each property (if no notable properties exist, return empty list. Phrase the properties such that a user can understand what they mean without reading the prompt or responses.):**```json
[
  {
    "model": "Model A|Model B",
    "property_description": "Brief description of the unique property observed in this model response (max 2 sentences, only give the property itself - remove any beginning or ending phrases like 'The response is...', 'The model has...', etc.)",
    "category": "1-4 word category",
    "evidence": "Direct quote or evidence from the specified model",
    "type": "General|Context-Specific",
    "reason": "Brief justification for this property, noting its absence/difference in the other model (max 2 sentences)",
    "impact": "Low|Medium|High",
    "user_preference_direction": "Capability-focused|Experience-focused|Neutral|Negative",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  }
]
```"""


def create_pairwise_analysis_prompt(prompt: str, model_a_name: str, model_a_response: str,
                                  model_b_name: str, model_b_response: str) -> str:
    """Create a prompt for analyzing a single response pair."""
    return f"""{PAIRWISE_SYSTEM_PROMPT}

**Model A Response:** {model_a_response}

**Model B Response:** {model_b_response}

Please analyze these two responses and identify unique properties in each model's response using the JSON format specified above."""


def create_batch_summary_prompt(batch_data: List[Dict[str, Any]], query: str) -> str:
    """Create a prompt for summarizing a batch of differences."""
    return f"""Analyze this batch of {len(batch_data)} response pairs and identify the most common behavioral patterns between the models.

Data:
{json.dumps(batch_data, indent=2)}

Query: {query}

Provide a concise summary of the prominent differences in this batch that would answer the given query. For each pattern, include:
- Pattern name
- Brief description
- Rough frequency (e.g., "seen in 20% of examples")
- 1-2 representative examples

Format your response as JSON:
{{"patterns": [
  {{
    "name": "Pattern name",
    "description": "Brief description",
    "frequency": "Approximate frequency",
    "examples": ["Brief example 1", "Brief example 2"]
  }}
]}}"""


def create_query_prompt(query: str, summary_data: Union[List[Dict], str],
                       num_pairs: int, is_batch_summary: bool = False, num_hypotheses: int = 5) -> str:
    """Create a prompt for querying differences based on summaries or raw data."""

    intro = f"""You are an expert AI researcher analyzing behavioral differences between two language models.
You have been given {"summaries of behavioral patterns" if is_batch_summary else "a dataset of analyzed response pairs"} from {num_pairs} analyzed response pairs.

Query: {query}

{"Batch summaries:" if is_batch_summary else "Dataset of analyzed differences:"}
{summary_data if isinstance(summary_data, str) else json.dumps(summary_data, indent=2)}

Based on {"these summaries" if is_batch_summary else "the provided data"}, identify at most {num_hypotheses} significant differences that respond to the query. I'm looking for differences of the format Model A/B is more X than Model B/A, where X is the difference. For each difference, provide:

1. **Description**: Describe a response that would validly have property X. Start with "This response .." Use 1-2 sentences to clearly and specifically describe the property, such that using this description could be used to identify the property on its own. Do not mention the model names.
2. **Detailed Description**: A detailed explanation of what the difference is and why it's significant
3. **Model A/B**: The model that exhibits this property more
4. **Percentage Difference**: An estimate of how much more frequently Model A exhibits this behavior compared to Model B. If the property is more frequent in Model A, the percentage difference should be positive. If the property is more frequent in Model B, the percentage difference should be negative.
5. **Examples**: 2-3 specific examples that demonstrate this difference

Make hypotheses specific and clear. Provide at most {num_hypotheses} differences in the following JSON format:

{{"differences": [
  {{
    "description": "Clear description of the property",
    "detailed_description": "Detailed explanation of the difference and why it's significant",
    "model_a_b": "Model A|Model B",
    "percentage_difference": "X% more present in Model A",
    "examples": [
      {{
        "prompt": "Original prompt text or description",
        "explanation": "Why this example demonstrates the difference"
      }}
    ]
  }}
]}}"""

    if not is_batch_summary:
        intro += "\n\nFocus on meaningful differences that provide insights into model capabilities, reasoning patterns, or behavioral tendencies."

    return intro


def extract_json_from_response(response: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract JSON from a model's response, handling various formats.

    Args:
        response: The model's response containing JSON

    Returns:
        Extracted JSON as dict or list

    Raises:
        ValueError: If no valid JSON could be extracted
    """
    if not response:
        raise ValueError("Empty response")

    try:
        # Try direct JSON parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Look for JSON in code blocks
    code_block_markers = [
        ("```json", "```"),
        ("```JSON", "```"),
        ("```", "```")
    ]

    for start_marker, end_marker in code_block_markers:
        if start_marker in response:
            start_idx = response.find(start_marker) + len(start_marker)
            end_idx = response.find(end_marker, start_idx)
            if end_idx != -1:
                json_str = response[start_idx:end_idx].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

    # Try to find JSON-like content (for arrays)
    lines = response.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    bracket_count = 0

    for line in lines:
        # Check for start of JSON array or object
        if not in_json:
            if '[' in line or '{' in line:
                in_json = True

        if in_json:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            bracket_count += line.count('[') - line.count(']')

            # Check if we've closed all brackets/braces
            if brace_count == 0 and bracket_count == 0 and ('{' in ''.join(json_lines) or '[' in ''.join(json_lines)):
                json_str = '\n'.join(json_lines).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Reset and continue looking
                    json_lines = []
                    in_json = False

    # Last attempt: find the largest JSON-like substring
    import re

    # Try to find JSON object
    obj_pattern = r'\{[^{}]*\}'
    matches = re.findall(obj_pattern, response, re.DOTALL)
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to find JSON array
    arr_pattern = r'\[[^\[\]]*\]'
    matches = re.findall(arr_pattern, response, re.DOTALL)
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract valid JSON from response. Response preview: {response[:500]}...")


def format_statistics_output(stats: Dict[str, Any]) -> str:
    """Format statistics dictionary for pretty printing."""
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.2%}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
            lines.append(f"{key}:")
            for item, count in value[:5]:  # Show top 5
                lines.append(f"  {item}: {count}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
