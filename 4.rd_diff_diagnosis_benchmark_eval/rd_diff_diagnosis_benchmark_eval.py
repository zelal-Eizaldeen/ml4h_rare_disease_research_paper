import argparse
import sys
import os
import json
import re
import difflib
import pronto

from pathlib import Path

from datetime import datetime
from typing import List, Dict, Tuple, Any,  Optional, Union

sys.path.append("/projects/illinois/eng/cs/user/scratch/RDMA")

from rdma.utils.data import read_json_file
from rdma.utils.llm_client import LocalLLMClient
def parse_diseases_json(response: str) -> Tuple[List[str], str]:
    """
    Robust function to parse LLM response into explanation and diseases list

    Returns:
        Tuple of (diseases_list, explanation)
    """
    try:
        # First, try direct JSON parsing
        cleaned = response.strip()
        response_json = json.loads(cleaned)

        # Validate it has the expected structure
        if isinstance(response_json, dict):
            explanation = response_json.get("explanation", "")
            diseases_list = response_json.get("rare_diseases", [])

            # Ensure diseases_list is actually a list
            if isinstance(diseases_list, list):
                return diseases_list, explanation
            else:
                return [], explanation
        else:
            return [], ""

    except json.JSONDecodeError:
        # Try to extract JSON object using regex
        json_pattern = r"\{.*?\}"
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                json_content = match.group(0)
                response_json = json.loads(json_content)
                explanation = response_json.get("explanation", "")
                diseases_list = response_json.get("rare_diseases", [])

                if isinstance(diseases_list, list):
                    return diseases_list, explanation
                else:
                    return [], explanation
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract from the response manually
        try:
            # Look for explanation and diseases patterns
            explanation_match = re.search(
                r'"explanation":\s*"([^"]*)"', response, re.DOTALL
            )
            diseases_match = re.search(
                r'"rare_diseases":\s*\[(.*?)\]', response, re.DOTALL
            )

            explanation = explanation_match.group(1) if explanation_match else ""

            if diseases_match:
                diseases_str = diseases_match.group(1)
                # Parse the diseases list
                diseases_list = []
                for item in diseases_str.split(","):
                    item = item.strip().strip("\"'")
                    if item:
                        diseases_list.append(item)
                return diseases_list, explanation
            else:
                return [], explanation

        except Exception:
            return [], ""

def is_soft_match(pred: str, gold: str, threshold: float = 0.8) -> bool:
    """
    Soft string match using difflib (normalized similarity).
    Returns True if similarity >= threshold.
    """
    sim = difflib.SequenceMatcher(None, pred.lower(), gold.lower()).ratio()
    return sim >= threshold

####################################################
# Extract rare diseases from MONDO OBO
####################################################
def build_rare_disease_set(obo_path: str, out_path: str = "/projects/illinois/eng/cs//scratch/RDMA/contribution/ontology_data/rare_diseases.json"):
    ontology = pronto.Ontology(obo_path)
    rare_diseases = set()

    for term in ontology.terms():
        if not term.name:
            continue
        # Check ORDO (Orphanet) cross refs
        xrefs = [xref.id.lower() for xref in term.xrefs]
        if any("ordo" in x for x in xrefs):
            rare_diseases.add(term.name.lower())
            continue
        # Check subsets
        subsets = [s.lower() for s in term.subsets]
        if any("rare" in s for s in subsets):
            rare_diseases.add(term.name.lower())

    # Save once to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(rare_diseases)), f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(rare_diseases)} rare diseases from MONDO")
    return rare_diseases

####################################################
# Load rare diseases (use pre-saved JSON if available)
####################################################
def load_rare_diseases(obo_file="/projects/illinois/eng/cs/scratch/RDMA/contribution/ontology_data/mondo.obo", json_file="/projects/illinois/eng/cs/scratch/RDMA/contribution/ontology_data/rare_diseases.json"):
    if Path(json_file).exists():
        print("Zilal json files YES")
        with open(json_file, "r") as f:
            return set(json.load(f))
    else:
        return build_rare_disease_set(obo_file, json_file)

#Added 
def clean_llm_predictions(predictions: List[str], rare_disease_set: Optional[set] = None) -> List[str]:
    """
    Filter and clean LLM predictions to remove non-diseases or common conditions.
    If rare_disease_set is provided, only keep items that match it.
    """
    cleaned = []
    seen = set()

    for item in predictions:
        # Strip reasoning text like "analysis..." or "we need..."
        item = re.sub(r'analysis.*|We need.*', '', item, flags=re.IGNORECASE).strip()

        if not item or len(item) < 3:
            continue

        item_norm = item.lower()

        # Example: skip obvious non-rare/common terms
        blacklist = {"diverticulosis", "hyperlipidemia", "hypertension", "splenomegaly", 
                     "anemia", "mediastinal lymphadenopathy"}  
        if item_norm in blacklist:
            continue

        # If ontology set provided, only keep matches
        if rare_disease_set and item_norm not in rare_disease_set:
            continue

        if item not in seen:
            seen.add(item)
            cleaned.append(item)

    return cleaned
#Ended 

def parse_diseases_list(response: str) -> List[str]:
    """
    Robust function to parse LLM response into Python list
    """
    try:
        # First, try direct JSON parsing
        cleaned = response.strip()
        diseases_list = json.loads(cleaned)

        # Validate it's a list
        if isinstance(diseases_list, list):
            return diseases_list
        else:
            return []

    except json.JSONDecodeError:
        # Try to extract JSON array using regex
        json_pattern = r'\[(.*?)\]'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                json_content = '[' + match.group(1) + ']'
                diseases_list = json.loads(json_content)
                return diseases_list
            except json.JSONDecodeError:
                pass

        # Last resort: manual parsing for common formats
        try:
            # Remove brackets and split by comma
            content = response.strip()
            content = re.sub(r'^\[|\]$', '', content)  # Remove outer brackets

            # Split by comma and clean each item
            items = [item.strip().strip('"\'') for item in content.split(',')]
            items = [item for item in items if item]  # Remove empty items

            return items

        except Exception:
            return []

def normalize_disease_name(disease: str) -> str:
    """
    Normalize disease name for comparison: lowercase and strip whitespace
    """
    return disease.lower().strip()
####################################################
# Integrate rare-disease filtering here
####################################################
def benchmark_rare_disease_diagnosis(
    data: Dict[str, Any],
    llm_client: Any,
    rare_diseases: set, 
    num_samples: int = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Benchmark rare disease diagnosis performance

    Args:
        data: Dictionary containing patient data with phenotypes and disease entities
        llm_client: LLM client with query method
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed results for each case

    Returns:
        Dictionary with benchmark metrics
    """

    # System prompt for LLM
    diff_diag_sys_prompt = """Given the following phenotypes, identify the top 10 most likely rare diseases and provide your reasoning.

CRITICAL: Your response must be EXACTLY in this JSON format with no additional text:

{
  "explanation": "Brief explanation of your diagnostic reasoning based on the phenotypes presented",
  "rare_diseases": ["Disease Name 1", "Disease Name 2", "Disease Name 3", "Disease Name 4", "Disease Name 5", "Disease Name 6", "Disease Name 7", "Disease Name 8", "Disease Name 9", "Disease Name 10"]
}

Rules:
1. Return ONLY the JSON object - no explanations outside the JSON, no additional text
2. Use double quotes around all strings
3. The "explanation" should be a concise summary of your diagnostic reasoning
4. The "rare_diseases" array should list diseases in order of likelihood (most likely first)

Example of correct format:
{
  "explanation": "The combination of tall stature, arachnodactyly, and lens dislocation strongly suggests connective tissue disorders, with Marfan syndrome being most likely.",
  "rare_diseases": ["Marfan Syndrome", "Ehlers-Danlos Syndrome", "Homocystinuria"]
}

Your response:"""
# Initialize counters
    total_diseases = 0
    hits = 0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_patients = 0
    patients_with_hits = 0
    parsing_failures = 0

    # Dictionary to store patient-level results
    patient_results = {}

    # Get samples to process
    items_to_process = list(data.items())
    if num_samples is not None:
        items_to_process = items_to_process[:num_samples]

    if verbose:
        print(f"Evaluating {len(items_to_process)} patients...")
        print("=" * 60)

    for patient_id, patient_data in items_to_process:
        if "matched_phenotypes" not in patient_data:
            continue

        total_patients += 1
        patient_hits = 0

        # Build phenotypes string
        phenotypes = ""
        for phenotype in patient_data["matched_phenotypes"]:
            phenotypes += phenotype["phenotype"] + ", "

        # Query LLM
        phenotypes_prompt = f"Phenotypes: {phenotypes}"
        print(phenotypes_prompt)
        llm_response = llm_client.query(
            system_message=diff_diag_sys_prompt, user_input=phenotypes_prompt
        )
        print("Test:", llm_response)
        # Parse response
        predicted_diseases, explanation = parse_diseases_json(llm_response)

        if not predicted_diseases and not explanation:
            parsing_failures += 1
            if verbose:
                print(f"Patient {patient_id}: Failed to parse LLM response")
                print(f"Raw response: '{llm_response}'")

            # Still store the failed case
            patient_results[patient_id] = {
                "phenotypes": phenotypes.strip(", "),
                "predicted_diseases": [],
                "explanation": "",
                "observed_diseases": patient_data.get("disease_entities", []),
                "raw_llm_response": llm_response,
                "parsing_failed": True,
                "hits": 0,
                "hit_at_1": False,
                "hit_at_5": False,
                "hit_at_10": False,
            }
            continue

        # Normalize predicted diseases for comparison
        predicted_normalized = [normalize_disease_name(d) for d in predicted_diseases]

        # Get ground truth diseases
        observed_diseases = patient_data.get("disease_entities", [])

        if verbose:
            print(f"Patient ID: {patient_id}")
            print(f"Phenotypes: {phenotypes.strip(', ')}")
            print(f"Explanation: {explanation}")
            print(f"Predicted diseases: {predicted_diseases}")
            print(f"Observed diseases: {observed_diseases}")

        # Calculate hits for this patient
        patient_hit_details = []
        patient_hit_at_1 = False
        patient_hit_at_5 = False
        patient_hit_at_10 = False

        for observed_disease in observed_diseases:
            observed_normalized = normalize_disease_name(observed_disease)
            total_diseases += 1

            # Check if disease is in top-10 predictions
            hit_in_top10 = observed_normalized in predicted_normalized
            hit_rank = None

            if hit_in_top10:
                hits += 1
                patient_hits += 1
                hit_rank = predicted_normalized.index(observed_normalized) + 1
                hits_at_10 += 1  # If it's in the list, it's automatically in top-10

                patient_hit_details.append(
                    {"disease": observed_disease, "hit": True, "rank": hit_rank}
                )

                if verbose:
                    print(f"  ✓ Hit: '{observed_disease}' found at rank {hit_rank}")

                # Check hit@1
                if hit_rank == 1:
                    hits_at_1 += 1
                    patient_hit_at_1 = True
                    if verbose:
                        print(f"  ✓ Hit@1: '{observed_disease}' is top prediction")

                # Check hit@5
                if hit_rank <= 5:
                    hits_at_5 += 1
                    patient_hit_at_5 = True
                    if verbose and hit_rank <= 5:
                        print(
                            f"  ✓ Hit@5: '{observed_disease}' is in top 5 (rank {hit_rank})"
                        )

                # Patient hit@10 is already handled above
                patient_hit_at_10 = True
                if verbose:
                    print(
                        f"  ✓ Hit@10: '{observed_disease}' is in top 10 (rank {hit_rank})"
                    )

            else:
                patient_hit_details.append(
                    {"disease": observed_disease, "hit": False, "rank": None}
                )
                if verbose:
                    print(f"  ✗ Miss: '{observed_disease}' not found in predictions")

        if patient_hits > 0:
            patients_with_hits += 1

        # Store patient-level results
        patient_results[patient_id] = {
            "phenotypes": phenotypes.strip(", "),
            "predicted_diseases": predicted_diseases,
            "explanation": explanation,
            "observed_diseases": observed_diseases,
            "raw_llm_response": llm_response,
            "parsing_failed": False,
            "hits": patient_hits,
            "hit_at_1": patient_hit_at_1,
            "hit_at_5": patient_hit_at_5,
            "hit_at_10": patient_hit_at_10,
            "hit_details": patient_hit_details,
            "total_observed_diseases": len(observed_diseases),
        }

        if verbose:
            print(f"  Patient hits: {patient_hits}/{len(observed_diseases)}")
            print("-" * 60)

    # Calculate final metrics
    hit_rate = hits / total_diseases if total_diseases > 0 else 0
    hit_at_1_rate = hits_at_1 / total_diseases if total_diseases > 0 else 0
    hit_at_5_rate = hits_at_5 / total_diseases if total_diseases > 0 else 0
    hit_at_10_rate = hits_at_10 / total_diseases if total_diseases > 0 else 0
    patient_hit_rate = patients_with_hits / total_patients if total_patients > 0 else 0
    parsing_success_rate = (
        1 - (parsing_failures / total_patients) if total_patients > 0 else 0
    )

    results = {
        "hit_rate": hit_rate,  # Same as hit_at_10_rate for backwards compatibility
        "hit_at_1_rate": hit_at_1_rate,
        "hit_at_5_rate": hit_at_5_rate,
        "hit_at_10_rate": hit_at_10_rate,
        "patient_hit_rate": patient_hit_rate,
        "parsing_success_rate": parsing_success_rate,
        "total_diseases": total_diseases,
        "total_patients": total_patients,
        "hits": hits,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "patients_with_hits": patients_with_hits,
        "parsing_failures": parsing_failures,
    }

    return results, patient_results

   

def print_benchmark_results(results: Dict[str, float], model_type: str) -> None:
    """
    Print benchmark results in a formatted way
    """
    print("\n" + "=" * 60)
    print(f"RARE DISEASE DIAGNOSIS BENCHMARK RESULTS - {model_type.upper()}")
    print("=" * 60)
    print(f"Total patients evaluated: {results['total_patients']}")
    print(f"Total diseases to predict: {results['total_diseases']}")
    print(f"LLM parsing success rate: {results['parsing_success_rate']:.2%}")
    print("-" * 60)
    print(
        f"Hit@1 Rate: {results['hit_at_1_rate']:.2%} ({results['hits_at_1']}/{results['total_diseases']})"
    )
    print(
        f"Hit@5 Rate: {results['hit_at_5_rate']:.2%} ({results['hits_at_5']}/{results['total_diseases']})"
    )
    print(
        f"Hit@10 Rate: {results['hit_at_10_rate']:.2%} ({results['hits_at_10']}/{results['total_diseases']})"
    )
    print(
        f"Patient Hit Rate: {results['patient_hit_rate']:.2%} ({results['patients_with_hits']}/{results['total_patients']})"
    )
    print("=" * 60)
#Save computed results of llm inference into json
def build_pretty_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Add human-readable percentage strings alongside the raw numbers."""
    d = dict(results)  # raw (floats/ints)
    pretty = {
        "hit_rate": f"{results.get('hit_rate', 0):.2%}",
        "hit_at_1_rate": f"{results.get('hit_at_1_rate', 0):.2%}",
        "patient_hit_rate": f"{results.get('patient_hit_rate', 0):.2%}",
        "parsing_success_rate": f"{results.get('parsing_success_rate', 0):.2%}",
        "counts": {
            "total_patients": results.get("total_patients", 0),
            "total_diseases": results.get("total_diseases", 0),
            "hits": results.get("hits", 0),
            "hits_at_1": results.get("hits_at_1", 0),
            "patients_with_hits": results.get("patients_with_hits", 0),
            "parsing_failures": results.get("parsing_failures", 0),
        }
    }
    return {"raw": d, "pretty": pretty}

def save_results_to_file(
    results: Dict[str, float],
    patient_results: Dict[str, Dict[str, Any]],
    model_type: str,
    output_dir: str = "benchmark_results",
) -> str:
    """
    Save benchmark results and patient-level results to a JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add metadata to results
    results_with_metadata = {
        "model_type": model_type,
        "timestamp": str(pd.Timestamp.now()),
        "summary_metrics": results,
        "patient_results": patient_results,
    }

    # Create filename with timestamp
    filename = (
        f"{model_type}_benchmark_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    return filepath

def save_benchmark_results_json(
    results: Dict[str, Any],
    out_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Save benchmark results to JSON (raw numbers + pretty strings)."""
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **(metadata or {})
        },
        "results": build_pretty_results(results)
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out_path)


if __name__ == "__main__":
    """
    Main function to run the benchmarking script
    """

    parser = argparse.ArgumentParser(
        description="Benchmark LLM performance on rare disease diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common model types (any string accepted, will be passed to LocalLLMClient):
  llama3_70b    - Llama 3 70B Chat (70B parameters)
  llama3_8b     - Llama 3 8B Chat (8B parameters)  
  llama3_70b_2b - Llama 3 70B Chat (70B parameters)
  mistral_24b   - Mixtral 8x7B Instruct (24B parameters)
  llama3_70b_r1 - Llama 3 70B Chat R1 (70B parameters)
  qwen_70b      - Qwen2 72B Instruct (72B parameters)
  mixtral_70b   - Mixtral 8x7B Instruct (70B parameters)

Examples:
  python benchmark_llm.py --model_type llama3_8b --num_samples 10 --verbose
  python benchmark_llm.py --model_type mistral_24b --data_file custom_data.json
  python benchmark_llm.py --model_type qwen_70b --device cuda:1 --save_results
  python benchmark_llm.py --model_type my_custom_model --verbose
        """,
    )
    # Required arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of LLM model to benchmark (any string, passed to LocalLLMClient)",
    )
    # Data arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/medical_students_data/high_agreement_with_phenotypes.json",
        help="Path to the JSON data file containing patient data",
    )
    # Model configuration arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu', 'auto')",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/shared/rsaas/jw3/rare_disease/model_cache",
        help="Directory to cache models",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Temperature for model sampling (default: 0.001 for deterministic results)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of patient samples to evaluate (default: all)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each patient case",
    )

    # Output arguments
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to a JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
     # List models option
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List common model types and exit (for reference only)",
    )

    # parser.add_argument("--model", type=str, default="qwen3_32b",
    #                     help="Model type (e.g., mistral_24b, llama3_70b, qwen3_32b, gpt_oss_20b)")
    

    # parser.add_argument("--dataset", type=str, required=True,
    #                     help="Path to dataset JSON file")
    # parser.add_argument("--out", type=str, default=None,
    #                     help="Path to save benchmark results JSON")
    parser.add_argument("--mondo", type=str, default="mondo.obo", help="Path to MONDO OBO file")


    args = parser.parse_args()

    # Load dataset
    # data = read_json_file(args.dataset)
    # Handle list models option
    # Validate data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' not found.")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.data_file}...")
    try:
        with open(args.data_file, "r") as f:
            data = json.load(f)
        print(f"Loaded data for {len(data)} patients")
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
   
    # Load rare diseases from MONDO
    rare_diseases = load_rare_diseases(args.mondo)

    # Load client dynamically
    # Initialize LLM client
    print(f"Initializing {args.model_type} model on {args.device}...")
    try:
        llm_client = LocalLLMClient(
            model_type=args.model_type,
            device=args.device,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print(
            f"Make sure '{args.model_type}' is a valid model type supported by LocalLLMClient"
        )
        sys.exit(1)
    # llm_client = LocalLLMClient(
    #     model_type=args.model,
    #     device=args.device,
    #     temperature=args.temperature
    # )
    
    # Run benchmark
    print(f"\nStarting benchmark for {args.model_type}...")
    try:
        results, patient_results = benchmark_rare_disease_diagnosis(
            data=data,
            llm_client=llm_client,
            rare_diseases=rare_diseases, #By 
            num_samples=args.num_samples,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)
        
    # # Run benchmark
    # results = benchmark_rare_disease_diagnosis(
    #     data=data,
    #     llm_client=llm_client,
    #     rare_diseases=rare_diseases,
    #     verbose=True
    # )

    # Prepare metadata
    # Print results
    print_benchmark_results(results, args.model_type)
    
    # meta = {
    #     "model": args.model,
    #     "device": args.device,
    #     "temperature": args.temperature,
    #     "dataset": args.dataset,
    #     "mondo": args.mondo
    # }

    # # Output file path
    # out_path = args.out or f"hits_{args.model}.json"
    # out = save_benchmark_results_json(results, out_path, metadata=meta)
    # print("Saved:", out)
    
    
    # Save results if requested
    if args.save_results:
        try:
            import pandas as pd  # For timestamp

            filepath = save_results_to_file(
                results, patient_results, args.model_type, args.output_dir
            )
            print(f"\nResults saved to: {filepath}")
            print(f"Patient-level results included for {len(patient_results)} patients")
        except ImportError:
            # Fallback without pandas
            import datetime

            os.makedirs(args.output_dir, exist_ok=True)
            results_with_metadata = {
                "model_type": args.model_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "summary_metrics": results,
                "patient_results": patient_results,
            }
            filename = f"{args.model_type}_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(args.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(results_with_metadata, f, indent=2)
            print(f"\nResults saved to: {filepath}")
            print(f"Patient-level results included for {len(patient_results)} patients")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
