import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 原始 centroid 表
centroids = pd.DataFrame({
    "miR-141": [-1.33, 0.10, -0.23, 0.67, 0.82],
    "miR-155": [-1.18, -0.50, -0.10, 0.63, 1.17],
    "miR-21":  [-1.61, -0.19,  0.33, 0.77, 0.74],
    "miR-221": [-1.63, -0.23,  0.54, 0.69, 0.67],
    "miR-222": [-0.97, -1.05,  0.53, 0.83, 0.69]
}, index=["LX-2", "Hep3B", "HepG2", "Huh-7", "MHCC97H"])

# 2. t-SNE（2维）
tsne = TSNE(n_components=2, random_state=42, perplexity=2.5)
tsne_emb = tsne.fit_transform(centroids)

# 3. 画图
tsne_df = pd.DataFrame(tsne_emb, columns=["t-SNE-1", "t-SNE-2"], index=centroids.index)
plt.figure(figsize=(4,3))
sns.scatterplot(data=tsne_df, x="t-SNE-1", y="t-SNE-2", hue=tsne_df.index, s=120)
plt.title("t-SNE of miRNA centroids")
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()