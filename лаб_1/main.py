from cluster_analyzer import ClusteringExplorer


def main():
    explorer = ClusteringExplorer(k_candidates=range(2, 16), seed=42)

    df, X = explorer.read_dataset("dataset.csv")

    explorer.evaluate_candidates(X)
    labels = explorer.build_clusters(X)

    df["cluster"] = labels
    df.to_csv("dataset_clustered.csv", index=False, encoding="utf-8")
    print("Размеченный датасет сохранён: dataset_clustered.csv")
    explorer.dump_model("kmeans_model.joblib")

    explorer.render_plots(X, prefix="cluster_report")


if __name__ == "__main__":
    main()