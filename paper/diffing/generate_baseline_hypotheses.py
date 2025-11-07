#!/usr/bin/env python3
"""
Multi-model B version: Compares one model against multiple baseline models using Gemini 2.5 Pro.

This analyzer precomputes differences between model A and multiple model B columns:
- Finds differences present in model A but none of the model B columns
- Finds differences present in all model B columns but not in model A
- Saves/loads precomputed differences to/from JSON files
- Allows querying differences at test-time with arbitrary questions

Usage:
# Precompute differences and save to file (with multiple model B columns)
python generate_llm_hypotheses_multi.py data.csv --model-b-cols model_b1 model_b2 model_b3 --output precomputed.json --max-samples 100

# Load precomputed differences and query
python generate_llm_hypotheses_multi.py data.csv --load-precomputed precomputed.json --query "What are the major differences?"
"""

import asyncio
import csv
from tqdm import tqdm
import json
import os
import sys
from typing import Dict, List, Any, Set
from openai import AsyncOpenAI
import argparse
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from baseline_utils import (
    create_query_prompt,
    extract_json_from_response,
)
from interp_embed.llm.utils import get_llm_client, call_async_llm
from hypothesis_verifier import HypothesisVerifier

load_dotenv()

# Setup logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)

# Suppress httpx HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Default directories for saving precomputed and results
PRECOMPUTED_DIR = os.path.join("baselines", "precomputed_results")
RESULTS_DIR = os.path.join("baselines", "generated_hypotheses")

class MultiModelDiffAnalyzer:
    """Analyzer that compares model A against multiple model B columns."""

    def __init__(self, model: str = "google/gemini-2.5-flash", max_concurrency: int = 200):
        """Initialize the analyzer with OpenRouter API key."""
        self.client = get_llm_client(is_openai_model=model.startswith("openai/"), is_async=True)
        self.model = model
        self.precomputed_differences = []
        self.max_concurrency = max_concurrency
        self.tokens_used = 0
        self.model_a_col = None
        self.model_b_cols = None

    def read_csv_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """Read CSV file and return list of dictionaries with prompts and responses."""
        data = []
        try:
            data = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Successfully read {len(data)} rows from {csv_path}")
            return data.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

    def create_multi_model_prompt(self, prompt: str, model_a_response: str,
                                 model_b_responses: List[str], both_directions: bool = True) -> str:
        """Create a prompt for analyzing Dataset A against multiple Dataset B responses."""

        # Format the model B responses
        model_b_section = "\n\n".join([
            f"**Dataset B{i+1} Response:**\n{response}"
            for i, response in enumerate(model_b_responses)
        ])

        analysis_prompt = f"""Analyze the differences between Dataset A and multiple Dataset B responses.\n"""

        analysis_prompt += f'**User Prompt:**\n{prompt}' if prompt.strip() != '' else ''

        analysis_prompt += f"""
**Dataset A Response:**
{model_a_response}

{model_b_section}

{'Identify two types of differences:' if both_directions else 'Identify the following type of differences:'}
1. Properties/capabilities that Dataset A has but NONE of the Dataset B responses have{'''
2. Properties/capabilities that ALL Dataset B responses have but Dataset A does not have''' if both_directions else ''}

For each difference, provide a JSON object with:
- "category": The type of difference (e.g., "Style", "Content", "Technical", "Reasoning", "Accuracy")
- "property": Specific property being compared
- "difference_type": Either "unique_to_a" (present in A but none of B models) or "common_to_all_b" (present in all B models but not A)
- "impact": "Low", "Medium", or "High"
- "description": Brief explanation of the difference

Return your analysis as a JSON array of difference objects.
"""
        return analysis_prompt

    async def analyze_multi_comparison(self, prompt: str, model_a_response: str,
                                      model_b_responses: List[str], both_directions: bool = True) -> Dict[str, Any]:
        """Analyze model A against multiple model B responses."""
        try:
            analysis_prompt = self.create_multi_model_prompt(
                prompt, model_a_response, model_b_responses, both_directions
            )

            response = await call_async_llm(self.client, self.model, [{"role": "user", "content": analysis_prompt}])

            self.tokens_used += response.usage.total_tokens

            return {
                "prompt": prompt,
                "model_a_response": model_a_response,
                "model_b_responses": model_b_responses,
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing multi-model comparison: {e}")
            return {
                "prompt": prompt,
                "model_a_response": model_a_response,
                "model_b_responses": model_b_responses,
                "analysis": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON from the model's response."""
        try:
            result = extract_json_from_response(response)
            # Ensure we always return a list
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return []

    async def precompute_differences(self, csv_path: str, model_a_col: str, model_b_cols: List[str],
                                   max_samples: int = None, save_path: str = None, both_directions: bool = True) -> List[Dict[str, Any]]:
        """Precompute differences for model A against multiple model B columns."""
        logger.info(f"Starting multi-model analysis of {csv_path}")

        # Store model names for later use
        self.model_a_col = model_a_col
        self.model_b_cols = model_b_cols

        # Read data
        data = self.read_csv_data(csv_path)

        # Limit samples if specified
        if max_samples:
            data = data[:max_samples]

        # Process each row
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = [None] * len(data)

        async def process_row(i, row):
            async with semaphore:
                prompt_text = row.get('prompt', row.get('question', ''))
                model_a_response = row.get(model_a_col, '')

                # Get all model B responses
                model_b_responses = []
                for col in model_b_cols:
                    response = row.get(col, '')
                    if not response:
                        logger.warning(f"Skipping row {i}: missing data in column {col}")
                        return (i, None)
                    model_b_responses.append(response)

                if not model_a_response:
                    logger.warning(f"Skipping row {i}: missing data in column {model_a_col}")
                    return (i, None)

                multi_analysis = await self.analyze_multi_comparison(
                    prompt_text, model_a_response, model_b_responses, both_directions
                )

                # Extract structured properties from the analysis
                if multi_analysis.get('analysis'):
                    properties = self.extract_json_from_response(multi_analysis['analysis'])
                    # Filter for the two types of differences we care about
                    filtered_properties = []
                    for prop in properties:
                        diff_type = prop.get('difference_type', '')
                        if both_directions:
                            if diff_type in ['unique_to_a', 'common_to_all_b']:
                                filtered_properties.append(prop)
                        else:
                            if diff_type == 'unique_to_a':
                                filtered_properties.append(prop)
                    multi_analysis['properties'] = filtered_properties

                return (i, multi_analysis)

        tasks = [
            process_row(i, row)
            for i, row in enumerate(data)
        ]

        # Process with progress bar
        completed = 0
        total = len(tasks)
        print(f"Analyzing {total} multi-model comparisons")
        for fut in tqdm(asyncio.as_completed(tasks), total=total, desc="Analyzing multi-model comparisons"):
            i, multi_analysis = await fut
            if multi_analysis is not None:
                results[i] = multi_analysis
            completed += 1

        # Remove any None results (skipped rows)
        results = [r for r in results if r is not None]

        self.precomputed_differences = results
        logger.info(f"Completed analysis of {len(results)} multi-model comparisons")

        # Save results if path provided
        if save_path:
            self.save_precomputed_differences(save_path)

        return results

    def generate_save_path(self, csv_path: str, precomputed: bool = True) -> str:
        """Generate a save path based on input parameters."""
        # Create precomputed directory if it doesn't exist
        if precomputed:
            Path(PRECOMPUTED_DIR).mkdir(parents=True, exist_ok=True)
        else:
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

        # Generate filename based on input CSV
        csv_basename = Path(csv_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model}_{csv_basename}_multi_model_{timestamp}.json"

        # Clean filename (remove spaces and special chars)
        filename = filename.replace(" ", "_").replace("/", "_")

        return os.path.join(PRECOMPUTED_DIR if precomputed else RESULTS_DIR, filename)

    def save_precomputed_differences(self, output_path: str):
        """Save precomputed differences to a JSON file."""
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Add model names to the saved data
        data_to_save = {
            'model_a_col': self.model_a_col,
            'model_b_cols': self.model_b_cols,
            'differences': self.precomputed_differences
        }

        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        logger.info(f"Saved precomputed differences to {output_path}")

    def load_precomputed_differences(self, input_path: str):
        """Load precomputed differences from a JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
            # Handle both old and new format
            if isinstance(data, list):
                # Old format - just the differences
                self.precomputed_differences = data
            else:
                # New format - includes model names
                self.model_a_col = data.get('model_a_col')
                self.model_b_cols = data.get('model_b_cols')
                self.precomputed_differences = data.get('differences', [])
        logger.info(f"Loaded {len(self.precomputed_differences)} precomputed differences from {input_path}")

    async def query_differences(self, query: str, use_middle_out: bool = True, num_hypotheses: int = 5) -> Dict[str, Any]:
        """Query the precomputed multi-model differences with an arbitrary question."""
        if not self.precomputed_differences:
            return {"error": "No precomputed differences found. Please run precompute_differences first."}
        print("Querying differences")

        if use_middle_out:
            # Middle-out transform: First create a compressed summary of patterns
            logger.info("Using middle-out transform to handle large dataset")

            # Step 1: Create batches of differences to summarize
            batch_size = 100
            batches = [self.precomputed_differences[i:i + batch_size]
                      for i in range(0, len(self.precomputed_differences), batch_size)]

            logger.info(f"Processing {len(batches)} batches of differences")

            # Step 2: Summarize each batch focusing on multi-model patterns
            batch_summaries = []
            for i, batch in enumerate(batches):
                logger.info(f"Summarizing batch {i+1}/{len(batches)}")

                batch_data = []
                for comparison in batch:
                    if comparison.get('properties'):
                        # Focus on the two types of differences we care about
                        unique_to_a = [p for p in comparison['properties'] if p.get('difference_type') == 'unique_to_a']
                        common_to_all_b = [p for p in comparison['properties'] if p.get('difference_type') == 'common_to_all_b']

                        if unique_to_a or common_to_all_b:
                            batch_data.append({
                                "prompt": comparison['prompt'][:200] + "..." if len(comparison['prompt']) > 200 else comparison['prompt'],
                                "unique_to_a": unique_to_a,
                                "common_to_all_b": common_to_all_b
                            })

                if not batch_data:
                    continue

                # Create a summary prompt for this batch
                summary_prompt = f"""Summarize the following multi-model comparison patterns for the query: "{query}"
Batch data:
{json.dumps(batch_data, indent=2)}

Provide a concise summary of the key patterns relevant to the query."""

                try:
                    response = await call_async_llm(self.client, self.model, [{"role": "user", "content": summary_prompt}])

                    self.tokens_used += response.usage.total_tokens

                    summary = response.choices[0].message.content
                    batch_summaries.append({
                        "batch_index": i,
                        "batch_size": len(batch_data),
                        "summary": summary
                    })
                except Exception as e:
                    logger.error(f"Error summarizing batch {i}: {e}")
                    continue

            # Step 3: Use the summaries to answer the query
            logger.info("Using batch summaries to answer query")

            query_prompt = create_query_prompt(
                query,
                json.dumps(batch_summaries, indent=2),
                len(self.precomputed_differences),
                is_batch_summary=True,
                num_hypotheses=num_hypotheses
            )

        else:
            # Direct approach - may hit token limits on large datasets
            logger.warning("Using direct approach without middle-out transform - may hit token limits")

            # Create a summary of all differences for the query
            summary_data = []
            for comparison in self.precomputed_differences:
                if comparison.get('properties'):
                    unique_to_a = [p for p in comparison['properties'] if p.get('difference_type') == 'unique_to_a']
                    common_to_all_b = [p for p in comparison['properties'] if p.get('difference_type') == 'common_to_all_b']

                    if unique_to_a or common_to_all_b:
                        summary_data.append({
                            "prompt": comparison['prompt'],
                            "unique_to_a": unique_to_a,
                            "common_to_all_b": common_to_all_b
                        })

            # Create a query prompt
            query_prompt = create_query_prompt(
                query,
                summary_data,
                len(self.precomputed_differences),
                is_batch_summary=False,
                num_hypotheses=num_hypotheses
            )

        try:
            response = await call_async_llm(self.client, self.model, [{"role": "user", "content": query_prompt}])

            self.tokens_used += response.usage.total_tokens

            # Extract structured results
            try:
                results = extract_json_from_response(response.choices[0].message.content)
                results["query"] = query
                return results
            except Exception as e:
                # If JSON extraction fails, return the raw response
                logger.warning(f"Could not extract JSON from query response: {e}")
                return {"raw_response": response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Error querying differences: {e}")
            return {"error": f"Error processing query: {e}"}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the precomputed multi-model differences."""
        if not self.precomputed_differences:
            return {"error": "No precomputed differences found"}

        total_comparisons = len(self.precomputed_differences)
        successful_analyses = sum(1 for comp in self.precomputed_differences if comp.get('properties'))

        # Count properties by type
        unique_to_a_count = 0
        common_to_all_b_count = 0
        category_counts = {}
        impact_counts = {"Low": 0, "Medium": 0, "High": 0}

        for comp in self.precomputed_differences:
            if comp.get('properties'):
                for prop in comp['properties']:
                    diff_type = prop.get('difference_type', '')

                    if diff_type == 'unique_to_a':
                        unique_to_a_count += 1
                    elif diff_type == 'common_to_all_b':
                        common_to_all_b_count += 1

                    # Category counts
                    category = prop.get('category', 'Unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1

                    # Impact counts
                    impact = prop.get('impact', 'Unknown')
                    if impact in impact_counts:
                        impact_counts[impact] += 1

        return {
            "total_comparisons": total_comparisons,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / total_comparisons if total_comparisons > 0 else 0,
            "unique_to_a_count": unique_to_a_count,
            "common_to_all_b_count": common_to_all_b_count,
            "category_counts": category_counts,
            "impact_counts": impact_counts
        }

    def print_results(self, results: Dict[str, Any]):
        """Print the analysis results in a formatted way."""
        if "raw_response" in results:
            print("\n" + "="*80)
            print("ANALYSIS RESULTS (Raw Response)")
            print("="*80)
            print(results["raw_response"])
            return

        print("\n" + "="*80)
        print("MULTI-MODEL DIFFERENCE ANALYSIS RESULTS")
        print("="*80)

        if "differences" in results:
            for i, diff in enumerate(results["differences"], 1):
                print(f"\n{i}. {diff.get('title', 'Untitled Difference')}")
                print("-" * 60)
                print(f"Description: {diff.get('description', 'No description')}")
                print(f"Percentage Difference: {diff.get('percentage_difference', 'Not specified')}")

                examples = diff.get('examples', [])
                if examples:
                    print(f"\nExamples ({len(examples)}):")
                    for j, example in enumerate(examples, 1):
                        print(f"  {j}. Prompt: {example.get('prompt', 'No prompt')}")
                        print(f"     Explanation: {example.get('explanation', 'No explanation')}")
                        print()
        else:
            print("No structured differences found in the response.")


async def main():
    parser = argparse.ArgumentParser(description="Quickstart: 'python3 generate_baseline_hypotheses.py <path/to/csv> --model-a-col <column1> --model-b-cols <column2>'")
    parser.add_argument("csv_path", help="Path to CSV file containing prompts and model responses")
    parser.add_argument("--model-a-col", default="model_a_response", help="Column name for Model A responses")
    parser.add_argument("--model-b-cols", nargs='+', required=True, help="List of column names for Model B responses")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to analyze")
    parser.add_argument("--save-results", help="Output file path for analysis results")
    parser.add_argument("--save-precomputed", help="Save precomputed differences to file")
    parser.add_argument("--load-precomputed", help="Load precomputed differences from file")
    parser.add_argument("--query", default = "What are the most significant, interesting differences?", help="Query to run on precomputed differences")
    parser.add_argument("--model", default="google/gemini-2.5-flash", help="Model to use for analysis")
    parser.add_argument("--no-middle-out", action="store_true", help="Disable middle-out transform for queries")
    parser.add_argument("--max-concurrency", type=int, default=200, help="Maximum number of concurrent requests")
    parser.add_argument("--num-hypotheses", type=int, default=10, help="Number of hypotheses to generate")
    parser.add_argument("--both", action="store_true", help="Analyze differences in both directions (A->B and B->A)")
    parser.add_argument("--verify", action="store_true", help="Run hypothesis verification after generating hypotheses")
    parser.add_argument("--verify-judge-model", default="google/gemini-2.5-flash", help="Model to use as judge for verification")
    parser.add_argument("--verify-max-concurrent", type=int, default=100, help="Maximum concurrent verifications")


    args = parser.parse_args()

    print(f"args.both: {args.both}")

    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} not found")
        sys.exit(1)

    # model_b_cols is now directly a list from argparse
    model_b_cols = args.model_b_cols

    # Initialize analyzer
    analyzer = MultiModelDiffAnalyzer(model=args.model, max_concurrency=args.max_concurrency)

    if args.load_precomputed:
        # Load precomputed differences
        analyzer.load_precomputed_differences(args.load_precomputed)
    else:
        # Generate save path if not provided
        if not args.save_precomputed:
            args.save_precomputed = analyzer.generate_save_path(args.csv_path)
            print(f"Auto-generated save path: {args.save_precomputed}")

        # Precompute differences
        results = await analyzer.precompute_differences(
            args.csv_path,
            args.model_a_col,
            model_b_cols,
            args.max_samples,
            args.save_precomputed,
            args.both
        )

    # Print statistics
    stats = analyzer.get_statistics()
    #print("\nMulti-Model Analysis Statistics:")
    #print(f"Total comparisons: {stats['total_comparisons']}")
    #print(f"Successful analyses: {stats['successful_analyses']}")
    #print(f"Success rate: {stats['success_rate']:.2%}")
    #print(f"Properties unique to Model A: {stats['unique_to_a_count']}")
    #print(f"Properties common to all comparison models: {stats['common_to_all_b_count']}")
    #print(f"Category counts: {stats['category_counts']}")
    #print(f"Impact counts: {stats['impact_counts']}")

    # Handle query if provided
    if args.query:
        #print(f"\nQuery: {args.query}")
        #print("="*50)
        use_middle_out = not args.no_middle_out
        answer = await analyzer.query_differences(args.query, use_middle_out=use_middle_out, num_hypotheses=args.num_hypotheses)
        #print(answer)
        if not args.save_results:
            args.save_results = analyzer.generate_save_path(args.csv_path, precomputed=False)
            print(f"Auto-generated output path: {args.save_results}")

        # Add link to precomputed file in results
        if args.save_precomputed:
            answer['precomputed_file'] = args.save_precomputed
        elif args.load_precomputed:
            answer['precomputed_file'] = args.load_precomputed

        # Add model names to results
        answer['model_a_col'] = analyzer.model_a_col
        answer['model_b_cols'] = analyzer.model_b_cols

        json.dump(answer, open(args.save_results, "w"), indent=2)

        # Run verification if requested
        if args.verify and args.save_results:
            print("\n" + "="*80)
            print("Running hypothesis verification...")
            print("="*80)

            # Parse model B columns
            verification_fields = [args.model_a_col] + model_b_cols

            # Create verification output directory
            verification_dir = os.path.join("baselines", "verification")
            Path(verification_dir).mkdir(parents=True, exist_ok=True)

            # Initialize verifier
            verifier = HypothesisVerifier(judge_model=args.verify_judge_model)

            # Load the hypotheses we just saved
            hypotheses = verifier.load_hypotheses(args.save_results)

            if hypotheses:
                # Load CSV data
                df = pd.read_csv(args.csv_path)

                # If max_samples was specified, use the same subset
                if args.max_samples:
                    df = df[:args.max_samples]

                # Load responses for all fields
                responses_by_field = verifier.load_responses_for_fields(df, verification_fields)

                # Run verification
                matrices, results = await verifier.verify_multiple_fields(
                    hypotheses,
                    responses_by_field,
                    max_concurrent=args.verify_max_concurrent
                )

                # Compute and save results
                df_results = verifier.compute_multi_field_results(
                    hypotheses, matrices, responses_by_field, results, df
                )

                # Save verification results with hypothesis file path
                verifier.save_multi_field_results(
                    verification_dir,
                    hypotheses,
                    df_results,
                    matrices,
                    verification_fields,
                    hypothesis_file_path=args.save_results
                )

                print(f"\nVerification results saved to {verification_dir}")
            else:
                print("No hypotheses found to verify")

    print("\n" + "="*80)
    print(f"Tokens used: {analyzer.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
