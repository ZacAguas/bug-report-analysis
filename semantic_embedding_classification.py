import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import wilcoxon
import json
import os

files = [
    "datasets/caffe.csv",
    "datasets/incubator-mxnet.csv",
    "datasets/keras.csv",
    "datasets/pytorch.csv",
    "datasets/tensorflow.csv",
]

REPEAT = 10
st_model = SentenceTransformer("all-MiniLM-L6-v2")
final_results = []

for file_path in files:
    project_name = os.path.basename(file_path).replace(".csv", "")

    df = pd.read_csv(file_path).sample(frac=1, random_state=999)
    df["text"] = df.apply(
        lambda row: str(row["Title"]) + ". " + str(row["Body"])
        if pd.notna(row["Body"])
        else str(row["Title"]),
        axis=1,
    )

    embeddings = st_model.encode(df["text"].tolist(), show_progress_bar=False)
    y = df["class"].values
    indices = np.arange(len(y))

    f1_scores = []
    precisions = []
    recalls = []

    for repeated_time in range(REPEAT):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=repeated_time,
        )

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(embeddings[train_idx], y[train_idx])
        y_pred = clf.predict(embeddings[test_idx])

        f1_scores.append(f1_score(y[test_idx], y_pred, average="macro"))
        precisions.append(precision_score(y[test_idx], y_pred, average="macro"))
        recalls.append(recall_score(y[test_idx], y_pred, average="macro"))

    # Load baseline scores for Wilcoxon
    with open(f"results/{project_name}_scores.json") as f:
        baseline = json.load(f)

    stat, p_val = wilcoxon(f1_scores, baseline["f1"])

    final_results.append(
        {
            "Project": project_name,
            "Baseline_F1": np.mean(baseline["f1"]),
            "Baseline_P": np.mean(baseline["precision"]),
            "Baseline_R": np.mean(baseline["recall"]),
            "Proposed_F1": np.mean(f1_scores),
            "Proposed_P": np.mean(precisions),
            "Proposed_R": np.mean(recalls),
            "p-value": p_val,
        }
    )

    os.makedirs("results", exist_ok=True)
    with open(f"results/{project_name}_proposed_scores.json", "w") as f:
        json.dump({"f1": f1_scores, "precision": precisions, "recall": recalls}, f)

print(pd.DataFrame(final_results).to_markdown(index=False))
