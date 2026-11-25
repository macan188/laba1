import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ClusteringExplorer:

    def __init__(self, k_candidates=range(2, 16), seed=42, target_k=5):
        self.k_candidates = list(k_candidates)
        self.seed = seed
        self.target_k = target_k
        self.chosen_k = None
        self.clusterer = None
        self.cluster_labels = None
        self.scores = None

    def read_dataset(self, path="dataset.csv"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV-файл отсутствует: {os.path.abspath(path)}")

        frame = pd.read_csv(path)
        numeric_cols = [col for col in frame.columns if col.startswith("feature_")]
        if not numeric_cols:
            numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()

        matrix = frame[numeric_cols].to_numpy(dtype=float)
        print(f"Данные загружены: {matrix.shape[0]} строк, "
              f"{matrix.shape[1]} столбцов: {numeric_cols}")
        return frame, matrix

    def evaluate_candidates(self, X):
        rows = []
        n_points = len(X)
        sil_sample = min(10_000, n_points)

        for k in self.k_candidates:
            model_k = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=5,
                random_state=self.seed
            )
            predicted = model_k.fit_predict(X)

            distortion = model_k.inertia_ / n_points
            sil_val = silhouette_score(
                X, predicted, sample_size=sil_sample, random_state=self.seed
            )
            db_val = davies_bouldin_score(X, predicted)

            rows.append({
                "k": k,
                "distortion": distortion,
                "silhouette": sil_val,
                "davies_bouldin": db_val
            })

        self.scores = pd.DataFrame(rows)
        best_row = self.scores.loc[self.scores["silhouette"].idxmax()]
        selected_row = (
            self.scores.loc[self.scores["k"] == self.target_k].iloc[0]
            if self.target_k in self.scores["k"].values
            else best_row
        )
        self.chosen_k = int(selected_row["k"])

        print("\nПеребор количества кластеров:")
        print(self.scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n>>> Выбрано k = {self.chosen_k} "
              f"(Silhouette = {selected_row['silhouette']:.4f}, "
              f"Davies-Bouldin = {selected_row['davies_bouldin']:.4f}; "
              f"максимум Silhouette при k = {int(best_row['k'])})")
        return self.chosen_k

    def build_clusters(self, X, k=None):
        n_clusters = k if k is not None else self.chosen_k
        if n_clusters is None:
            n_clusters = self.evaluate_candidates(X)
        self.chosen_k = int(n_clusters)

        self.clusterer = KMeans(
            n_clusters=self.chosen_k,
            init="k-means++",
            n_init=30,
            random_state=self.seed
        )
        self.cluster_labels = self.clusterer.fit_predict(X)
        return self.cluster_labels

    def dump_model(self, path="kmeans_model.joblib"):
        if self.clusterer is None:
            return
        joblib.dump(self.clusterer, path)
        print(f"Модель k-средних сохранена: {path}")

    def render_plots(self, X, prefix="cluster_report"):
        plt.style.use("default")
        plt.rcParams['axes.facecolor'] = '#f8f9fa'

        fig_sel, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(self.scores["k"], self.scores["distortion"],
                     marker="o", lw=2, color="#8e44ad")
        axes[0].axvline(self.chosen_k, color="#e74c3c", ls="-.", lw=1.5,
                        label=f"k = {self.chosen_k}")
        axes[0].set_title("Distortion (Mean Squared Distance)")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Distortion")
        axes[0].set_xticks(self.k_candidates)
        axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].plot(self.scores["k"], self.scores["silhouette"],
                     marker="s", lw=2, color="#16a085")
        axes[1].axvline(self.chosen_k, color="#e74c3c", ls="-.", lw=1.5,
                        label=f"k = {self.chosen_k}")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silhouette")
        axes[1].set_xticks(self.k_candidates)
        axes[1].legend()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        axes[2].plot(self.scores["k"], self.scores["davies_bouldin"],
                     marker="D", lw=2, color="#d35400")
        axes[2].axvline(self.chosen_k, color="#e74c3c", ls="-.", lw=1.5,
                        label=f"k = {self.chosen_k}")
        axes[2].set_title("Davies-Bouldin Index (lower is better)")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("DB Index")
        axes[2].set_xticks(self.k_candidates)
        axes[2].legend()
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        fig_sel.suptitle("Searching for the Best k", fontsize=14, fontweight="bold", y=1.02)
        fig_sel.tight_layout()
        fig_sel.savefig(f"{prefix}_metrics.png", dpi=200, bbox_inches='tight')

        dim = X.shape[1]
        pca2 = PCA(n_components=2, random_state=self.seed)
        X_p2 = pca2.fit_transform(X)
        c_p2 = pca2.transform(self.clusterer.cluster_centers_)
        r2 = pca2.explained_variance_ratio_ * 100

        fig_cl = plt.figure(figsize=(15, 6.5))

        custom_colors = ["#4B0082", "#00CED1", "#FF1493", "#FFD700", "#32CD32",
                         "#8A2BE2", "#FF4500", "#00FA9A", "#DC143C", "#1E90FF"]

        ax_a = fig_cl.add_subplot(1, 2, 1)
        for cid in range(self.chosen_k):
            sel = self.cluster_labels == cid
            color = custom_colors[cid % len(custom_colors)]
            ax_a.scatter(
                X_p2[sel, 0], X_p2[sel, 1],
                color=color, label=f"Кластер {cid} (n={sel.sum()})",
                alpha=0.6, s=12, edgecolors='white', linewidths=0.3, rasterized=True
            )

        ax_a.scatter(
            c_p2[:, 0], c_p2[:, 1],
            c="black", marker="*", s=350, edgecolors="white", linewidth=1.0,
            label="Центроиды", zorder=10
        )
        ax_a.set_xlabel(f"Главная компонента 1 ({r2[0]:.1f}%)", fontsize=11)
        ax_a.set_ylabel(f"Главная компонента 2 ({r2[1]:.1f}%)", fontsize=11)
        ax_a.set_title(f"2D проекция (PCA {dim}D → 2D)", fontsize=12, pad=10)

        ax_a.grid(True, linestyle='--', alpha=0.5)
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)

        ax_a.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, frameon=True,
                    facecolor='white', edgecolor='lightgray', markerscale=2)

        if dim >= 3:
            pca3 = PCA(n_components=3, random_state=self.seed)
            X_p3 = pca3.fit_transform(X)
            c_p3 = pca3.transform(self.clusterer.cluster_centers_)
            r3 = pca3.explained_variance_ratio_ * 100

            ax_b = fig_cl.add_subplot(1, 2, 2, projection="3d")
            for cid in range(self.chosen_k):
                sel = self.cluster_labels == cid
                color = custom_colors[cid % len(custom_colors)]
                ax_b.scatter(
                    X_p3[sel, 0], X_p3[sel, 1], X_p3[sel, 2],
                    color=color, alpha=0.4, s=8, rasterized=True
                )

            ax_b.scatter(
                c_p3[:, 0], c_p3[:, 1], c_p3[:, 2],
                c="black", marker="*", s=300, edgecolors="white", linewidth=0.8, zorder=10
            )

            ax_b.set_xlabel(f"PC1 ({r3[0]:.1f}%)", fontsize=10)
            ax_b.set_ylabel(f"PC2 ({r3[1]:.1f}%)", fontsize=10)
            ax_b.set_zlabel(f"PC3 ({r3[2]:.1f}%)", fontsize=10)
            ax_b.set_title(f"3D проекция (PCA {dim}D → 3D)", fontsize=12, pad=10)

            ax_b.xaxis.pane.fill = False
            ax_b.yaxis.pane.fill = False
            ax_b.zaxis.pane.fill = False
            ax_b.grid(True, linestyle=':', alpha=0.6)

        fig_cl.suptitle(f"Результаты кластеризации (k = {self.chosen_k})",
                        fontsize=15, fontweight="bold", y=1.02)
        fig_cl.tight_layout()
        fig_cl.savefig(f"{prefix}_clusters.png", dpi=200, bbox_inches='tight')
        print(f"Сохранены файлы: {prefix}_metrics.png, {prefix}_clusters.png")
        plt.show()