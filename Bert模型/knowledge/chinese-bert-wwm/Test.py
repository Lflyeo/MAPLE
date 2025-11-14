import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

# ==================================
# 📂 读取 Excel 文件
# ==================================
# ⚠️ 请修改为你的实际文件路径
df = pd.read_excel("fold_2_best_model_test_results.xlsx", sheet_name="Predictions")

# ==================================
# 🧩 数据预处理：将标签字符串拆分成列表
# ==================================
df["true_label"] = df["true_labels"].astype(str).apply(lambda x: [i.strip() for i in x.split(",")])
df["pred_label"] = df["pred_labels"].astype(str).apply(lambda x: [i.strip() for i in x.split(",")])

# ==================================
# 🧮 标签二值化（多标签任务所需）
# ==================================
mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(df["true_label"])
y_pred = mlb.transform(df["pred_label"])

# ==================================
# ⚙️ 定义计算函数
# ==================================
def calc_metrics(y_true, y_pred, average_type):
    return {
        "Precision": precision_score(y_true, y_pred, average=average_type, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average_type, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average_type, zero_division=0)
    }

# ==================================
# 📈 计算 micro / macro / weighted 三种平均
# ==================================
results = {
    "micro": calc_metrics(y_true, y_pred, "micro"),
    "macro": calc_metrics(y_true, y_pred, "macro"),
    "weighted": calc_metrics(y_true, y_pred, "weighted")
}

# ==================================
# 🖨️ 打印结果
# ==================================
print("=== 📊 Evaluation Results ===")
for avg_type, metrics in results.items():
    print(f"\n[{avg_type.upper()} Average]")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1']:.4f}")

# ==================================
# ✅ 输出模型预测结果对照表
# ==================================
df["Pred_Correct"] = df.apply(lambda x: set(x["true_label"]) == set(x["pred_label"]), axis=1)
print("\n=== 🧾 Sample Predictions ===")
print(df.head(10)[["true_label", "pred_label", "Pred_Correct"]])
