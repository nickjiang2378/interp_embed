#!/usr/bin/env python3
"""
Hypothesis Verifier for LLM Response Analysis

This script takes a series of hypotheses and a CSV file with multiple response fields,
then verifies whether each hypothesis applies to each response across all specified fields.
It identifies which hypotheses are unique to specific fields.

Usage:
    # Single field verification
    python hypothesis_verifier.py --hypotheses hypotheses.txt --input dataset.csv --fields response_column --output output_dir/

    # Multiple fields verification
    python hypothesis_verifier.py --hypotheses hypotheses.txt --input dataset.csv --fields field1 field2 field3 --output output_dir/
"""

import argparse
import json
import pandas as pd
import numpy as np
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os
from tqdm.asyncio import tqdm
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from interp_embed.llm.utils import get_llm_client, call_async_llm


class HypothesisVerifier:
    """Verifies whether hypotheses apply to LLM responses using an LLM judge."""

    def __init__(self, judge_model="openai/gpt-4o", api_key=None):
        self.judge_model = judge_model
        self.client = get_llm_client(is_openai_model = judge_model.startswith("openai/"), is_async=True)
        self.verification_results = []

    def load_hypotheses(self, hypotheses_file: str) -> List[Dict[str, Any]]:
        """Load hypotheses from a JSON file.

        Expects JSON format from pairwise baseline analysis with structure:
        {
            "differences": [
                {
                    "title": "Hypothesis title",
                    "description": "Detailed description",
                    "percentage_difference": "X% more/less frequent",
                    "examples": [
                        {
                            "prompt": "Example prompt",
                            "explanation": "Why this demonstrates the difference"
                        }
                    ]
                }
            ]
        }

        Returns list of hypothesis dictionaries with title, description, and examples.
        """
        with open(hypotheses_file, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            data = json.loads(content)

            # Handle the standard output format from pairwise baseline
            if isinstance(data, dict) and 'differences' in data:
                hypotheses = []
                for diff in data['differences']:
                    hypothesis = {
                        'title': diff.get('title', ''),
                        'description': diff.get('description', ''),
                        'percentage_difference': diff.get('percentage_difference', ''),
                        'examples': diff.get('examples', [])
                    }
                    hypotheses.append(hypothesis)
                print(f"Loaded {len(hypotheses)} hypotheses from pairwise baseline JSON format")
                return hypotheses

            # Handle direct list of hypotheses
            elif isinstance(data, list):
                hypotheses = []
                for item in data:
                    if isinstance(item, str):
                        # Simple string hypothesis
                        hypotheses.append({'title': item, 'description': '', 'examples': []})
                    elif isinstance(item, dict):
                        # Ensure required fields exist
                        hypothesis = {
                            'title': item.get('title', item.get('hypothesis', item.get('property_description', ''))),
                            'description': item.get('description', ''),
                            'examples': item.get('examples', [])
                        }
                        hypotheses.append(hypothesis)
                print(f"Loaded {len(hypotheses)} hypotheses from JSON list format")
                return hypotheses

            else:
                raise ValueError(f"Unexpected JSON format. Expected dict with 'differences' key or list of hypotheses.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in hypotheses file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading hypotheses: {e}")

    def load_responses_dataframe(self, responses_file: str) -> pd.DataFrame:
        """Load the full dataframe from CSV file."""
        try:
            df = pd.read_csv(responses_file)
            print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def load_responses_for_fields(self, df: pd.DataFrame, fields: List[str]) -> Dict[str, List[str]]:
        """Load responses for multiple fields from a dataframe."""
        responses_by_field = {}

        for field in fields:
            if field not in df.columns:
                raise ValueError(f"Field '{field}' not found in CSV. Available columns: {list(df.columns)}")

            responses = df[field].fillna("").astype(str).tolist()
            responses_by_field[field] = responses
            print(f"Loaded {len(responses)} responses from field '{field}'")

        return responses_by_field

    async def verify_hypothesis_response(self, hypothesis: Dict[str, Any], response: str,
                                       hypothesis_idx: int, response_idx: int) -> Dict[str, Any]:
        """Verify whether a single hypothesis applies to a single response."""

        # Extract hypothesis components
        hypothesis_description = hypothesis.get('description', '')

        # Build the prompt with all available information
        prompt = f"""You are an expert at analyzing whether text exhibits specific properties or characteristics.

HYPOTHESIS: {hypothesis_description}

RESPONSE TEXT TO ANALYZE:
{response}

TASK: Determine whether the response text exhibits the property described in the hypothesis.

INSTRUCTIONS:
1. Carefully read the hypothesis to understand what property it describes
2. Analyze the response text to see if it clearly embodies that property.
3. Consider both explicit and implicit manifestations of the property
4. Be consistent and objective in your evaluation
5. If you are unsure, answer "NO"
6. If the response text is close but not quite embodying the property, give an alternative version of the response that would've satisfied the property in your reasoning.

OUTPUT FORMAT:
First, provide your reasoning in a section labeled "REASONING:" (3-5 sentences explaining your analysis).
Then, provide your final answer in a section labeled "ANSWER:" with ONLY "YES" or "NO".

Example format:
REASONING: [Your analysis here explaining why the text does or doesn't exhibit the property, as well as an alternative version of the response that would've satisfied the property in your reasoning.]
ANSWER: YES

Your response:"""

        try:
            start_time = time.time()

            response_obj = await call_async_llm(self.client, self.judge_model, [{"role": "user", "content": prompt}], max_tokens=3000, temperature=0.0)

            end_time = time.time()

            # Parse the response to extract reasoning and answer
            full_response = response_obj.choices[0].message.content.strip()

            # Extract reasoning and answer
            reasoning = ""
            answer = ""
            failed_to_parse = False

            if "REASONING:" in full_response and "ANSWER:" in full_response:
                parts = full_response.split("ANSWER:")
                if len(parts) == 2:
                    reasoning_part = parts[0]
                    answer_part = parts[1]

                    # Extract reasoning
                    if "REASONING:" in reasoning_part:
                        reasoning = reasoning_part.split("REASONING:")[1].strip()

                    # Extract answer
                    answer = answer_part.strip().upper()
            else:
                # Fallback if format is not as expected
                answer = full_response.strip().upper()
                if answer not in ["YES", "NO"]:
                    answer = "NO"  # Default to NO if unclear
                reasoning = "Format error: Could not parse reasoning"
                failed_to_parse = True

            is_verified = answer == "YES"

            result = {
                'hypothesis_idx': hypothesis_idx,
                'response_idx': response_idx,
                'hypothesis': hypothesis_description,  # Keep full hypothesis dict for reference
                'response': response,  # Full response text, no truncation
                'verification': is_verified,
                'judge_response': answer,
                'reasoning': reasoning,
                'response_time': end_time - start_time,

            }

            # Mark format errors as failures
            if failed_to_parse:
                result["error"] = reasoning

            return result

        except Exception as e:
            return {
                'hypothesis_idx': hypothesis_idx,
                'response_idx': response_idx,
                'hypothesis': hypothesis_description,
                'response': response,  # Full response text, no truncation
                'verification': False,
                'judge_response': 'ERROR',
                'reasoning': f'Error occurred: {str(e)}',
                'response_time': 0,
                'tokens_used': 0,
                'error': str(e)
            }

    async def verify_all(self, hypotheses: List[Dict[str, Any]], responses: List[str],
                        max_concurrent: int = 5, dataset_name: str = "Dataset") -> np.ndarray:
        """Verify all hypothesis-response pairs and return a matrix."""

        print(f"Starting verification of {len(hypotheses)} hypotheses against {len(responses)} responses from {dataset_name}...")
        print(f"Total verification tasks: {len(hypotheses) * len(responses)}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_verify(hypothesis: Dict[str, Any], response: str, h_idx: int, r_idx: int, task_idx: int):
            """Verify with semaphore and preserve task index."""
            async with semaphore:
                result = await self.verify_hypothesis_response(hypothesis, response, h_idx, r_idx)
                return task_idx, result

        # Create all tasks with their indices
        tasks = []
        task_idx = 0
        for h_idx, hypothesis in enumerate(hypotheses):
            for r_idx, response in enumerate(responses):
                task = bounded_verify(hypothesis, response, h_idx, r_idx, task_idx)
                tasks.append(task)
                task_idx += 1

        # Execute tasks concurrently and collect results with preserved order
        results = [None] * len(tasks)

        # Track failures
        failure_count = 0

        # Use as_completed with tqdm progress bar
        with tqdm(total=len(tasks), desc=f"Verifying hypotheses for {dataset_name}") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    task_idx, result = await coro
                    results[task_idx] = result


                    # Check if this was a failure/error (either exception or format error)
                    if result and 'error' in result:
                        print(result)
                        failure_count += 1

                    # Update progress bar with failure count
                    pbar.set_postfix({'failures': failure_count})
                    pbar.update(1)

                except Exception as e:
                    failure_count += 1
                    # Update progress bar with failure count
                    pbar.set_postfix({'failures': failure_count})
                    pbar.update(1)

        # Filter out None results (in case of errors) and store for detailed output
        self.verification_results = [r for r in results if r is not None]

        # Convert results to matrix
        matrix = np.zeros((len(hypotheses), len(responses)), dtype=int)

        for result in self.verification_results:
            if result and 'error' not in result:
                h_idx = result['hypothesis_idx']
                r_idx = result['response_idx']
                matrix[h_idx, r_idx] = 1 if result['verification'] else 0

        return matrix

    async def verify_multiple_fields(self, hypotheses: List[Dict[str, Any]],
                                   responses_by_field: Dict[str, List[str]],
                                   max_concurrent: int = 5) -> tuple:
        """Verify hypotheses on multiple fields and return results."""

        fields = list(responses_by_field.keys())
        num_responses = len(next(iter(responses_by_field.values())))

        # Verify all fields have same number of responses
        for field, responses in responses_by_field.items():
            if len(responses) != num_responses:
                raise ValueError(f"All fields must have same number of responses. Field '{field}' has {len(responses)}, expected {num_responses}")

        print(f"\nVerifying hypotheses on {len(fields)} fields:")
        for field in fields:
            print(f"- {field}: {len(responses_by_field[field])} responses")
        print(f"- Number of hypotheses: {len(hypotheses)}")

        # Run verification on all fields
        matrices = {}
        results = {}

        for field in fields:
            print(f"\nProcessing field: {field}")
            matrix = await self.verify_all(hypotheses, responses_by_field[field], max_concurrent, field)
            matrices[field] = matrix
            results[field] = self.verification_results.copy()

        return matrices, results

    def compute_multi_field_results(self, hypotheses: List[Dict[str, Any]],
                                  matrices: Dict[str, np.ndarray],
                                  responses_by_field: Dict[str, List[str]],
                                  results_by_field: Dict[str, List[Dict[str, Any]]],
                                  df: pd.DataFrame = None) -> pd.DataFrame:
        """Create a comprehensive results dataframe for multiple fields."""

        fields = list(matrices.keys())
        num_responses = matrices[fields[0]].shape[1]

        # Build row-by-row results
        all_rows = []

        for r_idx in range(num_responses):
            for h_idx, hypothesis in enumerate(hypotheses):
                row_data = {
                    'response_idx': r_idx,
                    'hypothesis_idx': h_idx,
                    'hypothesis': hypothesis.get('description', '')
                }

                # Add prompt if available in the dataframe
                if df is not None and 'prompt' in df.columns:
                    row_data['prompt'] = df.iloc[r_idx]['prompt']

                # Add verification results for each field
                unique_count = 0
                total_verified = 0

                for field in fields:
                    is_verified = bool(matrices[field][h_idx, r_idx])
                    row_data[f'{field}_verified'] = is_verified
                    row_data[f'{field}_response'] = responses_by_field[field][r_idx]

                    # Find the corresponding result to get reasoning
                    reasoning = ""
                    for result in results_by_field[field]:
                        if result['hypothesis_idx'] == h_idx and result['response_idx'] == r_idx:
                            reasoning = result.get('reasoning', '')
                            break

                    row_data[f'{field}_reasoning'] = reasoning

                    if is_verified:
                        total_verified += 1

                # Check if uniquely verified in exactly one field
                if total_verified == 1:
                    for field in fields:
                        if row_data[f'{field}_verified']:
                            row_data['unique_to_field'] = field
                            break
                else:
                    row_data['unique_to_field'] = 'None' if total_verified == 0 else 'Multiple'

                row_data['num_fields_verified'] = total_verified
                all_rows.append(row_data)

        return pd.DataFrame(all_rows)


    def save_multi_field_results(self, output_dir: str,
                               hypotheses: List[Dict[str, Any]],
                               df_results: pd.DataFrame,
                               matrices: Dict[str, np.ndarray],
                               fields: List[str],
                               hypothesis_file_path: str = None):
        """Save multi-field verification results."""

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"multi_field_verification_{self.judge_model}_{timestamp}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # 1. Save the comprehensive results (main output requested by user)
        comprehensive_file = os.path.join(output_path, 'verification_results.csv')

        # Reorder columns for better readability
        col_order = ['response_idx', 'hypothesis_idx', 'hypothesis']

        # Add prompt column if it exists
        if 'prompt' in df_results.columns:
            col_order.append('prompt')

        # Add verification columns
        for field in fields:
            col_order.append(f'{field}_verified')

        # Add reasoning columns
        for field in fields:
            col_order.append(f'{field}_reasoning')

        # Add response columns
        for field in fields:
            col_order.append(f'{field}_response')

        # Add aggregate columns
        col_order.extend(['num_fields_verified', 'unique_to_field'])

        df_results = df_results[col_order]
        df_results.to_csv(comprehensive_file, index=False)
        print(f"Comprehensive results saved to {comprehensive_file}")

        # 2. Save summary statistics per hypothesis
        summary_data = []
        num_responses = matrices[fields[0]].shape[1]

        for h_idx, hypothesis in enumerate(hypotheses):
            row = {
                'hypothesis_idx': h_idx,
                'hypothesis': hypothesis.get('description', '')
            }

            # Count verifications per field
            for field in fields:
                verifications = matrices[field][h_idx, :]
                count = np.sum(verifications)
                pct = (count / num_responses) * 100
                row[f'{field}_count'] = count
                row[f'{field}_pct'] = pct

            # Count unique verifications
            unique_counts = {field: 0 for field in fields}

            for r_idx in range(num_responses):
                verified_fields = [field for field in fields if matrices[field][h_idx, r_idx] == 1]
                if len(verified_fields) == 1:
                    unique_counts[verified_fields[0]] += 1

            for field in fields:
                row[f'{field}_unique_count'] = unique_counts[field]
                row[f'{field}_unique_pct'] = (unique_counts[field] / num_responses) * 100

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_path, 'hypothesis_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to {summary_file}")

        # 3. Create JSON report
        report_data = {
            "metadata": {
                "timestamp": timestamp,
                "judge_model": self.judge_model,
                "hypothesis_file": hypothesis_file_path if hypothesis_file_path else "Not specified",
                "fields_analyzed": fields,
                "num_responses": num_responses,
                "num_hypotheses": len(hypotheses)
            },
            "hypotheses": hypotheses,
            "summary_by_hypothesis": []
        }

        # Add detailed summary for each hypothesis
        for _, row in summary_df.iterrows():
            hypothesis_summary = {
                "hypothesis_idx": int(row['hypothesis_idx']),
                "hypothesis": row['hypothesis'],
                "verification_rates": {},
                "unique_verification_counts": {}
            }

            for field in fields:
                hypothesis_summary["verification_rates"][field] = {
                    "percentage": float(row[f'{field}_pct']),
                    "count": int(row[f'{field}_count']),
                    "total": num_responses
                }
                hypothesis_summary["unique_verification_counts"][field] = {
                    "percentage": float(row[f'{field}_unique_pct']),
                    "count": int(row[f'{field}_unique_count']),
                    "total": num_responses
                }

            report_data["summary_by_hypothesis"].append(hypothesis_summary)

        # Save JSON report
        report_file = os.path.join(output_path, 'verification_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"JSON report saved to {report_file}")



async def main():
    """Main function to run the hypothesis verification."""

    parser = argparse.ArgumentParser(
        description="Verify hypotheses against LLM responses across multiple fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single field verification:
  python hypothesis_verifier.py -p hypotheses.txt -i dataset.csv --fields response_column -o output_dir/

  # Multiple fields verification:
  python hypothesis_verifier.py -p hypotheses.txt -i dataset.csv --fields field1 field2 field3 -o output_dir/
  """
    )

    parser.add_argument("-p", "--hypotheses", required=True,
                       help="Path to hypotheses file (text file with one per line, or JSON)")
    parser.add_argument("-i", "--input", required=True,
                       help="Path to CSV file containing responses")
    parser.add_argument("--fields", nargs='+', required=True,
                       help="Field names in CSV for responses (can specify multiple)")
    parser.add_argument("-o", "--output", required=True,
                       help="Path to output directory (ex. `sae/verification/`)")

    # Other options
    parser.add_argument("--judge-model", default="google/gemini-2.5-flash", help="Model to use as judge (default: google/gemini-2.5-flash)")
    parser.add_argument("--max-concurrent", type=int, default=100, help="Maximum concurrent verifications (default: 5)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.hypotheses):
        print(f"Error: Hypotheses file '{args.hypotheses}' not found")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return

    # Initialize verifier
    verifier = HypothesisVerifier(judge_model=args.judge_model, api_key=args.api_key)

    # Load hypotheses
    hypotheses = verifier.load_hypotheses(args.hypotheses)
    if not hypotheses:
        print("Error: No hypotheses loaded")
        return

    start_time = time.time()

    # Load the CSV dataframe
    df = verifier.load_responses_dataframe(args.input)

    # Load responses for all specified fields
    responses_by_field = verifier.load_responses_for_fields(df, args.fields)

    print(f"\n=== Running hypothesis verification on {len(args.fields)} fields ===")

    # Run multi-field verification
    matrices, results = await verifier.verify_multiple_fields(
        hypotheses,
        responses_by_field,
        max_concurrent=args.max_concurrent
    )

    # Compute comprehensive results
    df_results = verifier.compute_multi_field_results(hypotheses, matrices, responses_by_field, results, df)

    # Save results
    verifier.save_multi_field_results(
        args.output,
        hypotheses,
        df_results,
        matrices,
        args.fields,
        hypothesis_file_path=args.hypotheses
    )

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Fields analyzed: {', '.join(args.fields)}")
    print(f"Number of responses: {len(df)}")
    print(f"Number of hypotheses: {len(hypotheses)}")

    # Show unique verification counts per hypothesis
    for h_idx, hypothesis in enumerate(hypotheses[:5]):  # Show first 5
        print(f"\nHypothesis {h_idx}: {hypothesis.get('description', '')[:60]}...")
        for field in args.fields:
            unique_count = len(df_results[(df_results['hypothesis_idx'] == h_idx) &
                                        (df_results['unique_to_field'] == field)])
            print(f"  - Unique to {field}: {unique_count} responses")

    if len(hypotheses) > 5:
        print(f"\n... and {len(hypotheses) - 5} more hypotheses")

    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())