import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from src.qc.pipeline import QCPipeline
from src.utils.logger import setup_logger

logger = setup_logger()


def run_qc_workflow(
    input_file: Optional[str] = "data/input_samples/qc.json",
    output_file: str = "data/outputs/qc_results.json",
    review_queue_file: str = "data/outputs/review_queue.json",
    enable_bt: bool = True
) -> Dict[str, Any]:
    """
    Executes QC gates across dataset samples, generates review queue,
    and computes precision/recall when fault labels are present.
    """
    qc = QCPipeline(enable_back_translation=enable_bt)

    if input_file and Path(input_file).exists():
        with open(input_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        logger.info(f"Loaded {len(samples)} QC samples from {input_file}")
    else:
        logger.error(f"Input file non-existent or not provided: {input_file}")
        return {}

    results = []
    review_queue = []

    has_fault_labels = any("is_fault" in item for item in samples)
    tp, fp, tn, fn = 0, 0, 0, 0
    passed_count = 0

    for item in samples:
        src_text = item.get("source") or item.get("src") or item.get("original_src") or ""

        # Flexibly parse target text across key variations
        target_text = (
            item.get("target_text")
            or item.get("translated_text")
            or item.get("translation")
            or item.get("target")
            or item.get("ref")
            or ""
        )

        src_lang = item.get("src_lang") or item.get("source_lang") or "en"
        target_lang = item.get("tgt_lang") or item.get("target_lang") or "hi"
        dnt_terms = item.get("dnt_terms") or item.get("glossary") or []
        is_fault = item.get("is_fault", False)

        entity_mapping = {term: term for term in dnt_terms} if isinstance(dnt_terms, list) else dnt_terms

        qc_eval = qc.evaluate_segment(
            original_src=src_text,
            translated_text=target_text,
            target_lang=target_lang,
            source_lang=src_lang,
            entity_mapping=entity_mapping
        )

        passed = qc_eval["overall_passed"]
        flagged = not passed

        if passed:
            passed_count += 1

        if has_fault_labels:
            if is_fault and flagged:
                tp += 1
            elif not is_fault and flagged:
                fp += 1
            elif not is_fault and not flagged:
                tn += 1
            elif is_fault and not flagged:
                fn += 1

        record = {
            "id": item.get("id", "sample"),
            "source": src_text,
            "target_text": target_text,
            "qc": qc_eval
        }

        results.append(record)

        if flagged:
            review_queue.append(record)

    total = len(samples)
    pass_rate = f"{(passed_count / total) * 100:.1f}%" if total > 0 else "0.0%"

    qc_metrics = {}
    if has_fault_labels:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        qc_metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output = {
        "total_samples": total,
        "passed_samples": passed_count,
        "pass_rate": pass_rate,
        "flagged_for_review": len(review_queue),
        "qc_metrics": qc_metrics if has_fault_labels else "No fault-injection labels in dataset",
        "details": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)

    rq_path = Path(review_queue_file)
    rq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rq_path, "w", encoding="utf-8") as f:
        json.dump(review_queue, f, ensure_ascii=False, indent=2)

    metrics_log = (
        f" Precision: {qc_metrics['precision']}, Recall: {qc_metrics['recall']}, F1: {qc_metrics['f1_score']}."
        if has_fault_labels
        else " No fault-injection labels."
    )

    logger.info(
        f"QC Evaluation complete. Passed: {passed_count}/{total} ({pass_rate}). "
        f"Flagged for human review: {len(review_queue)}.{metrics_log} Exported to {rq_path}"
    )

    return summary_output


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "data/input_samples/qc.json"
    run_qc_workflow(input_file=file_arg)