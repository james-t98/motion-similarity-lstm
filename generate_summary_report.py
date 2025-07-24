import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Configuration ----------
BASE_DIR = "/content/drive/MyDrive/sdc_msc_data_analytics_project"
LOG_DIR = os.path.join(BASE_DIR, "logs", "inference")
SUMMARY_PATH = os.path.join(LOG_DIR, "inference_summary.csv")
VIS_DIR = os.path.join(LOG_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# ---------- Scan Subdirectories ----------
records = []

print("🔍 Scanning inference logs...")
for root, dirs, files in os.walk(LOG_DIR):
    for file in files:
        if file.startswith("inference_") and file.endswith(".csv"):
            file_path = os.path.join(root, file)
            method = file.replace("inference_", "").replace(".csv", "")
            run_id = os.path.basename(root)

            try:
                df = pd.read_csv(file_path)
                summary_idx = df[df.iloc[:, 0] == "summary"].index[0]
                mse = float(df.iloc[summary_idx + 2, 0])
                r2 = float(df.iloc[summary_idx + 2, 1])
                
                records.append({
                    "run_id": run_id,
                    "method": method,
                    "mse": mse,
                    "r2": r2,
                    "log_path": file_path
                })
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

# ---------- Build Summary ----------
if not records:
    print("❌ No valid inference logs found.")
    exit(1)

df = pd.DataFrame(records)
df.sort_values(by=["run_id", "method"], inplace=True)
df.to_csv(SUMMARY_PATH, index=False)
print(f"✅ Summary CSV saved: {SUMMARY_PATH}")

# ---------- Generate Visualizations ----------
pivot_r2 = df.pivot(index="run_id", columns="method", values="r2")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_r2, annot=True, cmap="YlGnBu", fmt=".4f", cbar_kws={"label": "R² Score"})
plt.title("📊 R² Score Heatmap by Run and Method")
plt.ylabel("Run ID")
plt.xlabel("Similarity Method")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "heatmap_r2.png"))
plt.close()
print(f"✅ Heatmap saved: {os.path.join(VIS_DIR, 'heatmap_r2.png')}")

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="r2", estimator=np.mean, errorbar="sd", palette="muted")
plt.title("🔍 Mean R² Score per Similarity Method")
plt.ylabel("Mean R² Score")
plt.xlabel("Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "barplot_r2_mean.png"))
plt.close()
print(f"✅ Barplot saved: {os.path.join(VIS_DIR, 'barplot_r2_mean.png')}")

plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="method", y="mse", marker='o', ci=None, estimator=np.mean)
plt.title("📈 Mean MSE per Method")
plt.ylabel("Mean MSE")
plt.xlabel("Method")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "lineplot_mse.png"))
plt.close()
print(f"✅ Line plot saved: {os.path.join(VIS_DIR, 'lineplot_mse.png')}")
