# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

TEST_TOPICS_PATH = os.getenv("TEST_TOPICS_PATH")


def _get_ks(topk: int) -> list[int]:
	return [a for a in [5, 10, 20, 50, 100, 200, 300, 500, 1000] if a <= topk]


def _load_scores_df(results: str | Path | pd.DataFrame, topk: int) -> pd.DataFrame:
	if isinstance(results, (str, Path)):
		scores_df = pd.read_csv(
			results,
			sep=",",
			header=None,
			skiprows=1,
			names=["q_id", "doc_id", "score"],
		)
	elif isinstance(results, pd.DataFrame):
		scores_df = results.copy()
		scores_df.columns = ["q_id", "doc_id", "score"]
	else:
		raise ValueError("results must be a string, Path object, or DataFrame.")

	scores_df["q_id"] = scores_df["q_id"].astype(str)
	scores_df["doc_id"] = scores_df["doc_id"].astype(str)
	scores_df["score"] = pd.to_numeric(scores_df["score"], errors="coerce")
	scores_df = scores_df.dropna(subset=["score"])
	return scores_df.groupby("q_id").head(topk).reset_index(drop=True)


def _load_qrels_dict(test_topics_path: Path) -> dict[str, set[str]]:
	return (
		pd.read_csv(
			test_topics_path,
			names=["topic_number", "candidate_number", "score"],
			sep="\t",
		)
		.groupby("topic_number")["candidate_number"]
		.apply(lambda s: set(s.astype(str).tolist()))
		.to_dict()
	)


def calculate_per_topic_metrics(
	results: str | Path | pd.DataFrame,
	topk: int,
	test_topics_path: Path = TEST_TOPICS_PATH,
) -> pd.DataFrame:
	test_topics_path = Path(test_topics_path)
	scores_df = _load_scores_df(results=results, topk=topk)
	true_dict = _load_qrels_dict(test_topics_path=test_topics_path)
	ks = _get_ks(topk=topk)

	rows = []
	for q_id, group in scores_df.groupby("q_id"):
		ranked_docs = group.sort_values("score", ascending=False)["doc_id"].tolist()
		true_rels = true_dict.get(q_id, set())
		row = {"q_id": str(q_id)}

		for k in ks:
			k_int = int(k)
			top_k_docs = ranked_docs[:k]
			relevance_flags = [1 if doc_id in true_rels else 0 for doc_id in top_k_docs]

			tp = int(sum(relevance_flags))
			fp = int(len(top_k_docs) - tp)
			fn = int(len(true_rels - set(top_k_docs)))
			accuracy = tp / k if k > 0 else 0.0
			precision = tp / (tp + fp) if tp + fp > 0 else 0.0
			recall = tp / (tp + fn) if tp + fn > 0 else 0.0
			f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

			dcg = float(sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_flags)))
			ideal_relevance_flags = sorted(relevance_flags, reverse=True)
			idcg = float(sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance_flags)))
			ndcg = dcg / idcg if idcg > 0 else 0.0

			row[f"accuracy@{k_int}"] = float(accuracy)
			row[f"precision@{k_int}"] = float(precision)
			row[f"recall@{k_int}"] = float(recall)
			row[f"f1@{k_int}"] = float(f1)
			row[f"nDCG@{k_int}"] = float(ndcg)

		rows.append(row)

	return pd.DataFrame(rows)


def bootstrap_recall_ndcg(
	per_topic_metrics_df: pd.DataFrame,
	confidence_level: float = 0.95,
	seed: int = 42,
	n_bootstrap: int | None = None,
) -> dict:
	if per_topic_metrics_df is None or per_topic_metrics_df.empty:
		return {
			"confidence_level": float(confidence_level),
			"seed": int(seed),
			"n_bootstrap": 0,
			"topic_count": 0,
			"by_k": {},
			"note": "No per-topic metrics available for bootstrap.",
		}
	if "topic" in per_topic_metrics_df.columns:
		per_topic_metrics_df = per_topic_metrics_df.rename(columns={"topic": "q_id"})
		
	unique_topics = per_topic_metrics_df["q_id"].nunique()
	n_resamples = int(n_bootstrap) if n_bootstrap is not None else 1000

	def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
		mean_value = float(np.mean(values))
		if values.size < 2:
			return mean_value, mean_value, mean_value
		result = bootstrap(
			(values,),
			np.mean,
			confidence_level=float(confidence_level),
			n_resamples=int(n_resamples),
			method="percentile",
			random_state=int(seed),
		)
		return (
			mean_value,
			float(result.confidence_interval.low),
			float(result.confidence_interval.high),
		)

	by_k = {}
	ks = sorted(
		int(col.split("@", 1)[1])
		for col in per_topic_metrics_df.columns
		if col.startswith("recall@")
	)

	for k in ks:
		recall_col = f"recall@{k}"
		ndcg_col = f"nDCG@{k}"
		if ndcg_col not in per_topic_metrics_df.columns:
			continue

		k_df = per_topic_metrics_df[[recall_col, ndcg_col]].apply(pd.to_numeric, errors="coerce").dropna()
		n_topics_k = int(len(k_df))
		if n_topics_k == 0:
			by_k[str(k)] = {
				"topic_count": 0,
				"recall": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
				"nDCG": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
			}
			continue

		recall_values = k_df[recall_col].to_numpy(dtype=float)
		ndcg_values = k_df[ndcg_col].to_numpy(dtype=float)
		recall_mean, recall_ci_low, recall_ci_high = _bootstrap_mean_ci(recall_values)
		ndcg_mean, ndcg_ci_low, ndcg_ci_high = _bootstrap_mean_ci(ndcg_values)

		entry = {
			"topic_count": n_topics_k,
			"recall": {"mean": recall_mean, "ci_low": recall_ci_low, "ci_high": recall_ci_high},
			"nDCG": {"mean": ndcg_mean, "ci_low": ndcg_ci_low, "ci_high": ndcg_ci_high},
		}
		if n_topics_k < 2:
			entry["note"] = "Degenerate CI: fewer than 2 topics."
		by_k[str(k)] = entry

	return {
		"confidence_level": float(confidence_level),
		"seed": int(seed),
		"n_bootstrap": int(n_resamples),
		"topic_count": int(unique_topics),
		"metric_names": ["recall", "nDCG"],
		"by_k": by_k,
	}

def calculate_metrics(
	results: str | Path | pd.DataFrame,
	topk: int ,
	test_topics_path: Path = TEST_TOPICS_PATH,
) -> dict:
	test_topics_path = Path(test_topics_path)
	per_topic_df = calculate_per_topic_metrics(
		results=results,
		topk=topk,
		test_topics_path=test_topics_path,
	)
	metrics = {}
	for k in _get_ks(topk=topk):
		metrics.update(
			{
				f"accuracy@{int(k)}": float(pd.to_numeric(per_topic_df[f"accuracy@{int(k)}"], errors="coerce").mean()),
				f"precision@{int(k)}": float(pd.to_numeric(per_topic_df[f"precision@{int(k)}"], errors="coerce").mean()),
				f"recall@{int(k)}": float(pd.to_numeric(per_topic_df[f"recall@{int(k)}"], errors="coerce").mean()),
				f"f1@{int(k)}": float(pd.to_numeric(per_topic_df[f"f1@{int(k)}"], errors="coerce").mean()),
				f"nDCG@{int(k)}": float(pd.to_numeric(per_topic_df[f"nDCG@{int(k)}"], errors="coerce").mean()),
			}
		)


	return metrics
