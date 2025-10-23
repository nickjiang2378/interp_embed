from typing import List

# TODO: implement sample type
def build_scoring_prompt(feature_description: str, sample: str, explanation: bool = True, sample_type: str = "positive") -> str:
    """Build prompt for single sample scoring, comparing positive and negative samples."""

    prompt = f"""You are an expert at evaluating sparse autoencoder feature descriptions. You will be scoring how accurate the feature description is. Some descriptions are poor; some are good.

You are given a feature description and a sample. Tokens that are surrounded by << >> markers are where the feature activated on. Your job is to determine whether the feature description accurately describes where the feature activates, considering BOTH the context before the << >> markers AND the marked tokens themselves.

IMPORTANT NOTES:
1. The << >> markers indicate where the feature activated, but you should NOT restrict your understanding to just those marked tokens. Look at the context BEFORE the marked tokens as well - the preceding tokens often provide crucial information about what the feature is detecting.
2. The feature may be responding to a pattern or concept that spans both the context tokens AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
4. You shouldn't be trying to infer what the feature description should be from the marked tokens; rather, you should use the feature description to score the sample.
5. If there are no tokens marked with << >> markers, the feature did not activate, and you should score whether the feature SHOULD have activated based on the feature description.

FEATURE DESCRIPTION:
"{feature_description}"

SAMPLE:
{sample}

Your task:
- Evaluate if the feature description accurately describes whether or not the feature activates, considering BOTH the context before the << >> markers AND the marked tokens themselves to understand what triggered the feature
- If there are marked tokens, score 1 if the property described by the feature description is clearly present in the sample at the marked tokens (considering both context and marked tokens). If there are no marked tokens, score 1 if there are no tokens in the sample that align with the feature description.
- If there are marked tokens, score 0 if the property described by the feature description is not clearly present in the sample at the marked tokens. If the feature description is not even a valid semantic or linguistic property (ex. "feature_#"), mark 0. If there are no marked tokens, score 0 if there ARE tokens in the sample that align with the feature description.

Return your answer as a JSON object with exactly these fields:
{"- 'explanation': '<brief explanation for the score, focusing on how the context and marked tokens together show the difference between samples>'" if explanation else ""}
- "score": <0 or 1>

Make sure your response is valid JSON that can be parsed directly. Keep the explanation brief (1-2 sentences)."""
    return prompt

def build_labeling_prompt(positive_samples: List[str], negative_samples: List[str], label_and_score = None, explanation:bool = True) -> str:
  """Build prompt for relabeling a feature."""
  refinement_context = ""
  if label_and_score is not None:
    refinement_context = f"""
    REFINEMENT CONTEXT:
    The current label and score for this feature is: {label_and_score}. The score refers to the accuracy of the label on twenty other samples not shown here. Please refine the label based on the samples below.
    """

  prompt = f"""You are an expert at interpreting features from sparse autoencoders (SAEs) for language models.
Below are {len(positive_samples)} POSITIVE samples (where the feature activated, with tokens surrounded by << and >>) and {len(negative_samples)} NEGATIVE samples (where it did not activate, no << >> markers).

The POSITIVE sample contains tokens that caused the feature to activate (marked with << >>), while the NEGATIVE sample does not.

IMPORTANT NOTES:
1. The << >> markers indicate where the feature activated, but you should NOT restrict your understanding to just those marked tokens. Look at the context BEFORE the marked tokens as well - the preceding tokens often provide crucial information about what the feature is detecting.
2. The feature may be responding to a pattern or concept that spans both the marked tokens AND the tokens before the marked token.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end of text, not a meaningful activation.
{refinement_context}
POSITIVE SAMPLES(given as a list of strings):
{positive_samples}

NEGATIVE SAMPLES(given as a list of strings):
{negative_samples}

Your task:
- Carefully compare the POSITIVE and NEGATIVE samples
- Look at BOTH the tokens before the << >> markers AND the marked tokens themselves to understand what the feature is detecting.
- Identify the most specific and concise property that is present in the POSITIVE samples (considering both context and marked tokens), but absent in the NEGATIVE samples.
- Try to give a unified property that isn't just a list of properties, if possible.
- Summarize the common attribute or property that causes the feature to activate. Be as specific as possible, but keep your description concise and clear.
- Do not reference specific sample numbers; however, you can reference the content in the positive and negative samples

Return your answer as a JSON object with exactly these fields:
- "label": "A concise phrase describing the property present in the positive samples (considering both context and marked tokens) but not in the negative samples."
- "brief_description": "A sentence expanding on the label, explaining what the feature is detecting in more detail. This should be a single sentence, not a list of properties. You can phrase it as, the feature is detecting X, etc."
{"- 'detailed_explanation': 'An extended explanation of what this featudre is detecting, including how the context before the marked tokens contributes to the feature's meaning. The explanation should be sufficient on its own to understand what the feature detects. Keep it to <5 concise sentences.'" if explanation else ""}

Make sure your response is valid JSON that can be parsed directly.
"""
  return prompt