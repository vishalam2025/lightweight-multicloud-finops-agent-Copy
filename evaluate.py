import pandas as pd
from detectors import rule_based_detect, statistical_detect, ml_based_detect
def get_truth_map(df):
    return df.drop_duplicates("resource_id").set_index("resource_id")["is_truly_idle"].to_dict()
def score_detector(flags, truth_map):
    tp = fp = tn = fn = 0

    for resource_id, flagged in flags.items():
        actual = truth_map[resource_id]
        if flagged and actual:
            tp += 1
        elif flagged and not actual:
            fp += 1
        elif not flagged and not actual:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "false_positive_rate": round(fpr, 3),
        "false_negative_rate": round(fnr, 3),
    }
if __name__ == "__main__":
    df = pd.read_csv("synthetic_usage_dataset.csv")
    truth_map = get_truth_map(df)

    strategies = {
        "Rule-Based": rule_based_detect(df),
        "Statistical": statistical_detect(df),
        "ML-Based": ml_based_detect(df),
    }

    rows = []
    for name, flags in strategies.items():
        metrics = score_detector(flags, truth_map)
        metrics["strategy"] = name
        rows.append(metrics)

    result_df = pd.DataFrame(rows).set_index("strategy")
    print(result_df)
    combined_flags = {rid: (strategies["Rule-Based"][rid] or strategies["Statistical"][rid] or strategies["ML-Based"][rid]) for rid in strategies["Rule-Based"]}
    print(score_detector(combined_flags, truth_map))
