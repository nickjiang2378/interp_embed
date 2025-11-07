#!/usr/bin/env python3
"""
Hypothesis Generation System for Dataset Analysis

This script analyzes differences between two dataset objects and generates hypotheses
about the differences based on user queries. It uses the same analysis pipeline as
analysis.ipynb but with additional functionality for hypothesis generation.
"""

import sys
import os
import json
import asyncio
import time
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_asyncio
from typing import Union
import logging
from dotenv import load_dotenv
from interp_embed.llm.utils import call_async_llm, get_llm_client
from interp_embed import Dataset

from datetime import datetime

# Setup path
cwd = os.getcwd()
APP_ROOT = os.path.join(cwd, "..")

# Load environment variables
load_dotenv()

from hypothesis_verifier import HypothesisVerifier

# Import helper functions
from sae_utils import (
    extract_json_from_response,
    ensure_prompts_list,
    get_average_score,
    build_gpt4_labeling_prompt,
    build_single_sample_prompt,
    build_hypotheses_prompt,
    build_middle_out_batch_prompt,
    build_middle_out_final_prompt,
    diff_features_multi,
    limit_feature_differences,
    FeatureLabelingResponse,
    SingleSampleScoringResponse,
)


# Default directories for saving precomputed and results
PRECOMPUTED_DIR = os.path.join("sae", "precomputed_results")
RESULTS_DIR = os.path.join("sae", "generated_hypotheses")

class HypothesisGenerator:
    """Generates hypotheses about differences between two datasets."""

    def __init__(self, model: str = "google/gemini-2.5-flash", max_concurrency: int = 8):
        self.model = model
        self.async_client = get_llm_client(is_openai_model = model.startswith("openai/"), is_async=True)
        self.sync_client = get_llm_client(is_openai_model = model.startswith("openai/"), is_async=False)
        self.sem = asyncio.Semaphore(max_concurrency)
        self.tokens_used = 0

    async def call_llm_async(self, model, messages, max_tokens = None):
        """Make async LLM API call with semaphore."""
        async with self.sem:
            response = await call_async_llm(self.async_client, model, messages, max_tokens = max_tokens)

        return response

    async def label_feature(self, positive_samples: List[str], negative_samples: List[str],
                          prompts: List[str] = None,
                          model: str = None, feature_score: float = None, current_label: str = None) -> Optional[FeatureLabelingResponse]:
        """Label a feature using LLM API."""
        if model is None:
            model = self.model

        prompt = build_gpt4_labeling_prompt(positive_samples, negative_samples, prompts,
                                          feature_score=feature_score, current_label=current_label)

        content = None
        try:
            response = await self.call_llm_async(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing sparse autoencoder features. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens = 10000
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            json_str = extract_json_from_response(content)
            response_data = json.loads(json_str)
            return FeatureLabelingResponse(**response_data)

        except Exception as e:
            print(f"Error calling model for labeling: {e}")
            print(content)
            return None

    async def score_single_sample(self, feature_description: str, positive_sample: str, negative_sample:str,
                                prompt: str = "",
                                model: str = None) -> Optional[SingleSampleScoringResponse]:
        """Score a single sample against a feature description."""
        if model is None:
            model = self.model

        prompt_text = build_single_sample_prompt(feature_description, positive_sample, negative_sample, prompt)
        content = None
        try:
            response = await self.call_llm_async(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating sparse autoencoder feature descriptions. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens = 1000
            )
            content = response.choices[0].message.content

            # Extract JSON from response
            json_str = extract_json_from_response(content)
            response_data = json.loads(json_str)
            return SingleSampleScoringResponse(**response_data)
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            print(content)
            return None

    async def score_feature_samples(self, feature_description: str, positive_samples: list, negative_samples: list,
                                  prompts: list = None,
                                  model: str = None) -> dict:
        """Score multiple samples for a feature."""
        if model is None:
            model = self.model

        # Default to empty prompts if not provided
        prompts = ensure_prompts_list(prompts, len(positive_samples))

        # Pair positive and negative samples
        n_pairs = min(len(positive_samples), len(negative_samples))

        async def sem_scoring(pos_sample, neg_sample, prompt):
            return await self.score_single_sample(feature_description, pos_sample, neg_sample, prompt, model=model)

        # Create tasks for paired samples
        tasks = []
        for i in range(n_pairs):
            prompt = prompts[i] if i < len(prompts) else ""
            tasks.append(sem_scoring(positive_samples[i], negative_samples[i], prompt))

        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]

        # Calculate scores
        total_score = sum(r.score for r in results)

        return {
            "feature_description": feature_description,
            "total_score": total_score,
            "total_count": len(results),
            "positive_sample_explanations": [r.explanation for r in results],
            "negative_sample_explanations": [],
            "all_responses": results,
        }

    async def process_feature(self, diff_activations, all_other_activations, difference, feature_id, target_dataset: Dataset, other_datasets: List[Dataset],
                            threshold: float = 0.75) -> Optional[Tuple]:
        """Process a single feature for analysis."""

        top_indices = np.argsort(diff_activations)[::-1][:40]

        positive_samples = [row.token_activations(feature_id) for row in target_dataset[top_indices]]

        # Get negative samples from all other datasets
        negative_samples = []
        for ind in top_indices:
            # Shuffle the activations for this index (across datasets)
            shuffled_activations = all_other_activations[:, ind].copy()
            other_indices = np.argsort(np.random.shuffle(shuffled_activations)).tolist() # Choose the sample with the lowest activation. If there are multiple at the lowest activation, randomly select one.
            negative_samples.append(other_datasets[other_indices[0]][ind.item()].token_activations(feature_id))


        # Get prompts from target dataset only
        df_target = target_dataset.pandas()
        if "prompt" in df_target.columns:
            prompts = [df_target.iloc[idx]["prompt"] for idx in top_indices]
        else:
            prompts = [""] * len(positive_samples)

        result = await self.score_feature_samples(
            target_dataset.feature_labels()[feature_id],
            positive_samples[:10],
            negative_samples[:10],
            prompts[:10],
        )

        current_score = get_average_score(result)

        # Relabel the feature using the dataset samples
        labeled_result = await self.label_feature(
            positive_samples[10:30],
            negative_samples[10:30],
            prompts[10:30],
            feature_score=current_score,
            current_label=target_dataset.feature_labels()[feature_id],
        )

        if labeled_result is None:
            return None

        new_result_score = await self.score_feature_samples(
            labeled_result.brief_description,
            positive_samples[30:],
            negative_samples[30:],
            prompts[30:],
        )
        new_score = get_average_score(new_result_score)

        # Use the better scoring description
        if new_score >= current_score and new_score >= threshold:
            return (feature_id, labeled_result.brief_description,
                    positive_samples[0], negative_samples[0],
                    prompts[0] if prompts else "",
                    new_score,
                    difference)
        elif new_score < current_score and current_score >= threshold:
            # Return original if no improvement
            return (feature_id, target_dataset.feature_labels()[feature_id],
                    positive_samples[0], negative_samples[0],
                    prompts[0] if prompts else "",
                    current_score,
                    difference)

    async def analyze_feature_differences(self, target_dataset: Dataset, other_datasets: Union[Dataset, List[Dataset]],
                                        threshold: float = 0.75, min_difference: float = 0.03, batch_size: int = 100, both_directions: bool = True, max_feature_diffs: int = None) -> List[Dict]:
        """Analyze differences between target dataset and one or more other datasets."""
        # Handle single dataset case
        if isinstance(other_datasets, Dataset):
            other_datasets = [other_datasets]

        print("Computing feature differences...")
        target_activations = target_dataset.latents()
        all_other_activations = []
        for other_dataset in other_datasets:
            all_other_activations.append(other_dataset.latents())
        all_other_activations = np.stack(all_other_activations, axis=0)
        diffs = diff_features_multi(target_activations, all_other_activations)

        print(diffs.head())

        # Filter for significant differences
        if both_directions:
            target_diff_others = diffs[diffs["target_diff_others"] > min_difference]
            others_diff_target = diffs[diffs["others_diff_target"] > min_difference]
            # Combine significant differences from both directions into a single DataFrame
            significant_diffs = pd.DataFrame({
                "feature_id": pd.concat([target_diff_others["feature_id"], others_diff_target["feature_id"]], ignore_index=True),
                "diff_activation": pd.concat([target_diff_others["target_diff_others"], -others_diff_target["others_diff_target"]], ignore_index=True)
            })
        else:
            # Only consider target_diff_others (features that target has but others don't)
            target_diff_others = diffs[diffs["target_diff_others"] > min_difference]
            significant_diffs = pd.DataFrame({
                "feature_id": target_diff_others["feature_id"],
                "diff_activation": target_diff_others["target_diff_others"]
            })
        significant_diffs = significant_diffs.sort_values("diff_activation", ascending = False)
        print(significant_diffs.head())

        print(f"Found {len(significant_diffs)} features with significant differences")

        # Apply max_feature_diffs limit if specified
        significant_diffs = limit_feature_differences(significant_diffs, max_feature_diffs, both_directions)

        # Process features in batches to control memory usage
        print(f"Processing {len(significant_diffs)} features in batches of {batch_size}...")

        other_activations = all_other_activations.max(axis=0)
        diff_activations = target_activations - other_activations

        semaphore = asyncio.Semaphore(batch_size)
        results = [None] * len(significant_diffs)

        async def process_feature_bounded(i):
            async with semaphore:
                feature_id = significant_diffs["feature_id"].iloc[i].item()
                difference = significant_diffs["diff_activation"].iloc[i].item()
                processed_feature = await self.process_feature(diff_activations[:, feature_id], all_other_activations[:, :, feature_id], difference, feature_id, target_dataset, other_datasets, threshold)
                return i, processed_feature

        tasks = [
            process_feature_bounded(i)
            for i in range(len(significant_diffs))
        ]

        # tqdm for async: wrap asyncio.as_completed with tqdm
        total = len(tasks)
        for fut in tqdm(asyncio.as_completed(tasks), total=total, desc="Analyzing features"):
            i, feature_analysis = await fut
            results[i] = feature_analysis

        results = [r for r in results if r is not None]

        print(f"Successfully processed {len(results)} / {total} features")

        # Convert to dictionary format
        descriptions_dict = []
        for entity in results:
            # New format: (feature_id, description, positive_sample, negative_sample, prompt, score, difference)
            if len(entity) == 7:
                descriptions_dict.append({
                    "feature_id": entity[0],
                    "description": entity[1],
                    "example_positive": entity[2],
                    "example_negative": entity[3],
                    "prompt": entity[4],
                    "feature_confidence": entity[5],
                    "difference": entity[6],
                })
            else:
                print(f"Warning: Unexpected entity format with {len(entity)} items: {entity}")

        # Sort by difference strength
        descriptions_dict.sort(key=lambda x: x["difference"], reverse=True)

        return descriptions_dict

    async def summarize_hypotheses_from_features(self, significant_features: List[Dict], query: str,
                                        hypothesis_format: str = "hypotheses", num_hypotheses: int = 10,
                                        model: str = None, both_directions: bool = True, max_retries: int = 5,
                                        use_middle_out: bool = True) -> Any:
        """
        Summarize hypotheses using LLM based on the query.

        Args:
            significant_features: List of significant feature differences
            query: User's query about the differences
            format: Either "paragraph" for a summary or "hypotheses" for structured hypotheses
            num_hypotheses: Number of hypotheses to generate (only used when format="hypotheses")
            model: Model to use (defaults to instance model)
            max_retries: Maximum number of retry attempts (default 5)
            use_middle_out: Whether to use middle-out approach for large feature sets

        Returns:
            Either a string summary or a list of Hypothesis objects depending on format
        """
        if model is None:
            model = self.model

        # Check if we need middle-out approach
        if use_middle_out and len(significant_features) > 50:
            # Step 1: Create batches of features
            batch_size = 200
            batches = [significant_features[i:i + batch_size]
                      for i in range(0, len(significant_features), batch_size)]

            # Step 2: Summarize each batch
            batch_summaries = []
            for i, batch in enumerate(batches):
                batch_prompt = build_middle_out_batch_prompt(batch, query, i, len(batches))

                try:
                    response = await self.call_llm_async(model, [{"role": "user", "content": batch_prompt}])
                    self.tokens_used += response.usage.total_tokens
                    batch_summaries.append({
                        "batch_index": i,
                        "batch_size": len(batch),
                        "summary": response.choices[0].message.content
                    })
                    print(f"Summarized batch {i+1}/{len(batches)}")
                except Exception as e:
                    continue

            # Step 3: Generate hypotheses from summaries
            prompt = build_middle_out_final_prompt(batch_summaries, query, num_hypotheses, len(significant_features), both_directions)
        else:
            # Use direct approach for smaller feature sets
            prompt = build_hypotheses_prompt(significant_features, query, hypothesis_format, num_hypotheses, both_directions)

        # Retry logic
        for attempt in range(max_retries):
            content = None
            try:
                # Prepare parameters for logging
                api_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": None,
                    "temperature": 0.1,
                }

                response = await self.call_llm_async(model, [{"role": "user", "content": prompt}])

                self.tokens_used += response.usage.total_tokens

                content = response.choices[0].message.content

                if hypothesis_format == "paragraph":
                    return content
                else:  # hypothesis_format == "hypotheses"
                    # Extract JSON from response
                    json_str = extract_json_from_response(content)

                    hypotheses_data = json.loads(json_str)

                    # Convert to list of dicts for JSON output
                    hypotheses_list = []
                    for hyp_data in hypotheses_data:
                        # If it's already a dict, just append; if it's a Hypothesis object, convert to dict
                        if isinstance(hyp_data, dict):
                            hypotheses_list.append(hyp_data)
                        else:
                            # fallback: convert Hypothesis dataclass to dict
                            hypotheses_list.append({
                                "description": hyp_data.description,
                                "examples": hyp_data.examples,
                                "percentage_difference": hyp_data.percentage_difference,
                                "feature_ids": hyp_data.feature_ids,
                                "confidence": hyp_data.confidence
                            })
                    return hypotheses_list

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed - Error generating {'summary' if hypothesis_format == 'paragraph' else 'hypotheses'}: {str(e)}"
                print(error_msg)
                if content:
                    print(f"Response content: {content}")

                # If this was the last attempt, return the failure value
                if attempt == max_retries - 1:
                    final_error_msg = f"Failed after {max_retries} attempts to generate hypotheses"
                    print(final_error_msg)
                    return None if hypothesis_format == "paragraph" else []

                # Wait a bit before retrying (exponential backoff)
                wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

    def generate_save_path(self, dataset1_path: str, dataset2_paths: Union[str, List[str]], output_type: str = "results") -> str:
        """Generate a save path based on input parameters.

        Args:
            dataset1_path: Path to target dataset
            dataset2_paths: Path(s) to other dataset(s)
            output_type: Either "precomputed" or "results"

        Returns:
            Generated save path with timestamp
        """
        # Create directory if it doesn't exist
        if output_type == "precomputed":
            Path(PRECOMPUTED_DIR).mkdir(parents=True, exist_ok=True)
            base_dir = PRECOMPUTED_DIR
        else:
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            base_dir = RESULTS_DIR

        # Generate filename based on input datasets
        dataset1_basename = Path(dataset1_path).stem

        # Handle multiple dataset2 paths
        if isinstance(dataset2_paths, list):
            # Concatenate basenames with underscore
            dataset2_basenames = [Path(p).stem for p in dataset2_paths]
            dataset2_combined = "_".join(dataset2_basenames)  # Limit to first 3 to avoid too long filenames
        else:
            dataset2_combined = Path(dataset2_paths).stem

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_type == "precomputed":
            filename = f"features_sae_{dataset1_basename}_vs_{dataset2_combined}_{timestamp}.csv"
        else:
            filename = f"hypotheses_sae_{dataset1_basename}_vs_{dataset2_combined}_{timestamp}.json"

        # Clean filename (remove spaces and special chars)
        filename = filename.replace(" ", "_").replace("/", "_")

        return os.path.join(base_dir, filename)

    async def generate_hypotheses(self, dataset1_path: str, dataset2_paths: Union[str, List[str]], query: str,
                                threshold: float = 0.75, min_difference: float = 0.03,
                                output_file: str = None, precomputed_features_path: str = None, num_hypotheses: int = 5,
                                batch_size: int = 100, both_directions: bool = True, use_middle_out: bool = True,
                                max_feature_diffs: int = None) -> Dict[str, Any]:
        """Main function to generate hypotheses from dataset files or precomputed features.

        Args:
            dataset1_path: Path to target dataset
            dataset2_paths: Path(s) to other dataset(s) to compare against
            query: Query about the differences
            threshold: Minimum score threshold for features
            min_difference: Minimum difference threshold
            output_file: Path to output JSON file
            precomputed_features_path: Path to precomputed features CSV
            num_hypotheses: Number of hypotheses to generate
        """
        # Convert single path to list
        if isinstance(dataset2_paths, str):
            dataset2_paths = [dataset2_paths]

        if precomputed_features_path:
            # Load precomputed features from CSV
            significant_features = self.load_precomputed_features(precomputed_features_path)
        else:
            # Original flow: analyze datasets
            print(f"Loading datasets...")
            print(f"Target dataset: {dataset1_path}")
            print(f"Other datasets: {dataset2_paths}")

            # Load target dataset
            dataset1 = Dataset.load_from_file(dataset1_path).filter_na_rows()

            # Load other datasets
            other_datasets = []
            for path in dataset2_paths:
                ds = Dataset.load_from_file(path).filter_na_rows()
                other_datasets.append(ds)
                print(f"Loaded dataset from {path}: size {len(ds)}")

            print(f"Target dataset size: {len(dataset1)}")
            print(f"Number of comparison datasets: {len(other_datasets)}")

            # Analyze differences
            significant_features = await self.analyze_feature_differences(dataset1, other_datasets, threshold, min_difference, batch_size=batch_size, both_directions=both_directions, max_feature_diffs=max_feature_diffs)

        print(f"Found {len(significant_features)} significant features")

        # Generate hypotheses
        if both_directions:
            print("Generating hypotheses for both directions")
            # When both=True, run query twice: once for target->other, once for other->target
            # First filter features for target->other (positive differences)
            target_features = [f for f in significant_features if f.get('difference', 0) > 0]
            # Then filter features for other->target (negative differences)
            other_features = [f for f in significant_features if f.get('difference', 0) < 0]

            hypotheses = []

            # Generate hypotheses for target features if there are any
            if target_features:
                target_hypotheses = await self.summarize_hypotheses_from_features(
                    target_features, query, hypothesis_format="hypotheses",
                    num_hypotheses=num_hypotheses // 2, both_directions=False,
                    use_middle_out=use_middle_out
                )
                if target_hypotheses:
                    hypotheses.extend(target_hypotheses)

            # Generate hypotheses for other features if there are any
            if other_features:
                other_hypotheses = await self.summarize_hypotheses_from_features(
                    other_features, query, hypothesis_format="hypotheses",
                    num_hypotheses=num_hypotheses // 2, both_directions=False,
                    use_middle_out=use_middle_out
                )
                if other_hypotheses:
                    hypotheses.extend(other_hypotheses)
        else:
            print("Generating hypotheses for target > other differences")
            # When both=False, only look at target->other differences
            hypotheses = await self.summarize_hypotheses_from_features(
                significant_features, query, hypothesis_format="hypotheses",
                num_hypotheses=num_hypotheses, both_directions=False,
                use_middle_out=use_middle_out
            )

        # Save features to CSV (only if not loading from precomputed)
        features_csv_path = None
        if not precomputed_features_path:
            # Generate default path for features
            features_csv_path = self.generate_save_path(dataset1_path, dataset2_paths, "precomputed")
            features_csv_path = self.save_features_to_csv(significant_features, features_csv_path)

        # Generate output path if not provided
        if not output_file:
            output_file = self.generate_save_path(dataset1_path, dataset2_paths, "results")
            print(f"Auto-generated output path: {output_file}")

        result = {
            "query": query,
            "differences": hypotheses,  # Now a list of dicts, ready for JSON serialization
            "significant_features": significant_features,
            "dataset1_path": dataset1_path,
            "dataset2_paths": dataset2_paths,  # Now a list
            "target_model": dataset1_path,  # Target model filename
            "other_models": dataset2_paths,  # Other model filenames
            "features_csv_path": str(features_csv_path) if features_csv_path else None,
            "precomputed_features_path": precomputed_features_path,
            "output_file": output_file
        }

        # Save results to output file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to {output_file}")

        return result

    def save_features_to_csv(self, significant_features: List[Dict], features_path: str):
        """Save significant features to a CSV file."""
        # Ensure directory exists
        Path(features_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(significant_features)

        # Reorder columns for better readability
        column_order = [
            'feature_id', 'description', 'prompt',
            'example_positive', 'example_negative',
            'feature_confidence', 'difference'
        ]

        # Only include columns that exist
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]

        # Save to CSV
        df.to_csv(features_path, index=False)
        print(f"Features saved to: {features_path}")

        return features_path

    def load_precomputed_features(self, csv_path: str) -> List[Dict]:
        """Load precomputed features from a CSV file."""
        print(f"Loading precomputed features from: {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Convert to list of dictionaries
        features = df.to_dict('records')

        # Ensure all required fields are present with correct types
        for feature in features:
            # Convert feature_id to int if it's stored as string
            if 'feature_id' in feature:
                feature['feature_id'] = int(feature['feature_id'])

            # Set defaults for missing fields
            feature.setdefault('feature_confidence', 0.8)
            feature.setdefault('difference', 0.1)
            feature.setdefault('example_positive', '')
            feature.setdefault('example_negative', '')
            feature.setdefault('description', '')
            feature.setdefault('prompt', '')

            # Handle legacy format with separate positive/negative prompts
            if 'prompt_positive' in feature and 'prompt' not in feature:
                feature['prompt'] = feature.get('prompt_positive', '')

        print(f"Loaded {len(features)} features from CSV")
        return features


async def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate hypotheses about dataset differences")
    parser.add_argument("--dataset1", required=True, help="Path to target dataset file (.pkl)")
    parser.add_argument("--dataset2", required=True, nargs='+', help="Path(s) to other dataset file(s) (.pkl) to compare against")
    parser.add_argument("--query", type=str, default = "What are the most significant, interesting differences?", help="Query about the differences (e.g., 'what are the most interesting differences?')")
    parser.add_argument("--output", help="Path to output JSON file (auto-generated if not provided)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum score threshold for feature label to be considered accurate")
    parser.add_argument("--min-difference", type=float, default=0.03, help="Minimum difference threshold (absolute value)")
    parser.add_argument("--load-precomputed", help="Path to precomputed features CSV file (skips feature analysis)")
    parser.add_argument("--num-hypotheses", type=int, default=10, help="Number of hypotheses to generate")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Maximum number of concurrent tasks")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash", help="LLM model to use for hypothesis generation")
    parser.add_argument("--sae-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="SAE model for tokenization")
    parser.add_argument("--batch-size", type=int, default=20, help="Process features in batches to control memory usage")
    parser.add_argument("--both", action="store_true", default=False,  help="Analyze differences in both directions (target->other and other->target)")
    parser.add_argument("--max-feature-diffs", type=int, default=None, help="Maximum number of feature differences to analyze (None for no limit)")
    parser.add_argument("--no-middle-out", action="store_true", help="Disable middle-out approach for large feature sets (may hit token limits)")
    parser.add_argument("--verify-file", type=str, required=False, help="Path to CSV/JSON file containing data to verify hypotheses against (required when --verify is used)")
    parser.add_argument("--verify-judge-model", default="google/gemini-2.5-flash", help="Model to use as judge for verification")
    parser.add_argument("--verify-max-concurrent", type=int, default=100, help="Maximum concurrent verifications")

    args = parser.parse_args()

    # Initialize generator
    generator = HypothesisGenerator(
        model=args.model,
        max_concurrency=args.max_concurrency
    )

    # Generate hypotheses
    use_middle_out = not args.no_middle_out
    result = await generator.generate_hypotheses(
        args.dataset1,
        args.dataset2,
        args.query,
        args.threshold,
        args.min_difference,
        args.output,
        args.load_precomputed,
        args.num_hypotheses,
        args.batch_size,
        args.both,
        use_middle_out,
        args.max_feature_diffs
    )

    # Print number of hypotheses (now a list of dicts in 'differences')
    print(f"Generated {len(result['differences'])} hypotheses")
    print(f"Tokens used: {generator.tokens_used}")
    # Run verification if requested
    if args.verify_file and len(result["differences"]) > 0:

        print(f"\nRunning hypothesis verification with {args.verify_file}")

        df_verify = pd.read_csv(args.verify_file)

        # Look for text field
        dataset1 = Dataset.load_from_file(args.dataset1)
        text_fields = [dataset1.field]
        for path in args.dataset2:
            dataset = Dataset.load_from_file(path)
            text_fields.append(dataset.field)

        # Run verification
        verification_dir = os.path.join("sae", "verification")
        Path(verification_dir).mkdir(parents=True, exist_ok=True)

        verifier = HypothesisVerifier(judge_model=args.verify_judge_model)
        hypotheses = verifier.load_hypotheses(result["output_file"])

        if hypotheses:
            responses_by_field = verifier.load_responses_for_fields(df_verify, text_fields)
            matrices, results = await verifier.verify_multiple_fields(
                hypotheses, responses_by_field, max_concurrent=args.verify_max_concurrent)
            df_results = verifier.compute_multi_field_results(
                hypotheses, matrices, responses_by_field, results, df_verify)
            verifier.save_multi_field_results(
                verification_dir, hypotheses, df_results, matrices, text_fields, hypothesis_file_path=result["output_file"])
            print(f"Verification complete. Results saved to {verification_dir}")

    # Log final token count at the very end
    print(f"\nFINAL TOTAL TOKENS USED: {generator.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
