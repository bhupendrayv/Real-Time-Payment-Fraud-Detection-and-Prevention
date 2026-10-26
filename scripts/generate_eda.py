"""Generate the Phase 1 EDA Jupyter notebook programmatically.

Uses nbformat so we get a valid .ipynb without hand-writing JSON.

Usage:
    python scripts/generate_eda.py
"""
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"


def make_notebook() -> None:
    """Build and save the EDA notebook."""
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        }
    )

    cells: list = []

    # ── Title ──────────────────────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "# FraudShield — Exploratory Data Analysis\n"
            "\n"
            "Dataset: [Kaggle Credit Card Fraud Detection (ULB)]"
            "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  \n"
            "284,807 transactions · 492 frauds · V1–V28 PCA features + Time + Amount\n"
            "\n"
            "**Questions:**\n"
            "1. How imbalanced is the dataset?\n"
            "2. How do fraud and legitimate transactions differ in amount?\n"
            "3. Are there temporal patterns?\n"
            "4. Any missing values?\n"
            "5. Which features correlate most with fraud?"
        )
    )

    # ── Imports ────────────────────────────────────────────────────
    cells.append(
        nbf.v4.new_code_cell(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')  # non-interactive backend for nbconvert\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "from ml.config import RAW_DATA_PATH, ALL_FEATURES, TARGET, DOCS_DIR\n"
            "\n"
            "sns.set_theme(style='whitegrid', palette='husl')\n"
            "DOCS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "print('Setup complete ✓')"
        )
    )

    # ── Load data ─────────────────────────────────────────────────
    cells.append(
        nbf.v4.new_code_cell(
            "df = pd.read_csv(RAW_DATA_PATH)\n"
            "print(f'Shape: {df.shape}')\n"
            "print(f'Columns: {list(df.columns)}')\n"
            "df.head()"
        )
    )

    # ── 1. Class balance ──────────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Class Balance\n"
            "\n"
            "The dataset is extremely imbalanced — fraud is about 0.17 % of transactions."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fraud_counts = df[TARGET].value_counts().sort_index()\n"
            "print(f'Legitimate (0): {fraud_counts[0]:>7,}  '\n"
            "      f'({fraud_counts[0] / len(df) * 100:.3f}%)')\n"
            "print(f'Fraud      (1): {fraud_counts[1]:>7,}  '\n"
            "      f'({fraud_counts[1] / len(df) * 100:.3f}%)')\n"
            "print(f'Imbalance ratio: 1 : {fraud_counts[0] // fraud_counts[1]}')\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "\n"
            "colors = ['#2ecc71', '#e74c3c']\n"
            "bars = axes[0].bar(['Legitimate', 'Fraud'],\n"
            "                   fraud_counts.values, color=colors)\n"
            "axes[0].set_ylabel('Count')\n"
            "axes[0].set_title('Class Distribution (absolute)')\n"
            "for bar, cnt in zip(bars, fraud_counts.values):\n"
            "    axes[0].text(bar.get_x() + bar.get_width() / 2,\n"
            "                bar.get_height(), f'{cnt:,}',\n"
            "                ha='center', va='bottom', fontweight='bold')\n"
            "\n"
            "axes[1].pie(fraud_counts.values,\n"
            "            labels=['Legitimate', 'Fraud'],\n"
            "            colors=colors, autopct='%1.3f%%', startangle=90)\n"
            "axes[1].set_title('Class Distribution (proportion)')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(DOCS_DIR / 'class_distribution.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "print('Saved → docs/class_distribution.png')"
        )
    )

    # ── 2. Amount distribution ────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell("## 2. Amount Distribution by Class")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "\n"
            "axes[0].hist(df[df[TARGET] == 0]['Amount'], bins=100,\n"
            "             color='#2ecc71', alpha=0.7, log=True)\n"
            "axes[0].set_title('Amount — Legitimate')\n"
            "axes[0].set_xlabel('Amount')\n"
            "axes[0].set_ylabel('Count (log scale)')\n"
            "\n"
            "axes[1].hist(df[df[TARGET] == 1]['Amount'], bins=50,\n"
            "             color='#e74c3c', alpha=0.7)\n"
            "axes[1].set_title('Amount — Fraud')\n"
            "axes[1].set_xlabel('Amount')\n"
            "axes[1].set_ylabel('Count')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(DOCS_DIR / 'amount_distribution.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('\\nAmount statistics by class:')\n"
            "print(df.groupby(TARGET)['Amount'].describe().round(2))"
        )
    )

    # ── 3. Time patterns ─────────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Time Patterns\n"
            "\n"
            "`Time` = seconds elapsed from the first transaction "
            "in the dataset (~48 hours)."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "hours = df['Time'] / 3600\n"
            "\n"
            "fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)\n"
            "\n"
            "axes[0].hist(hours, bins=100, color='#3498db', alpha=0.7)\n"
            "axes[0].set_title('All Transactions Over Time')\n"
            "axes[0].set_ylabel('Count')\n"
            "\n"
            "axes[1].hist(hours[df[TARGET] == 1], bins=48,\n"
            "             color='#e74c3c', alpha=0.7)\n"
            "axes[1].set_title('Fraud Transactions Over Time')\n"
            "axes[1].set_xlabel('Time (hours)')\n"
            "axes[1].set_ylabel('Count')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.savefig(DOCS_DIR / 'time_patterns.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    )

    # ── 4. Missing values ─────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("## 4. Missing Values"))

    cells.append(
        nbf.v4.new_code_cell(
            "missing = df.isnull().sum()\n"
            "total_missing = missing.sum()\n"
            "print(f'Total missing values across all columns: {total_missing}')\n"
            "\n"
            "if total_missing > 0:\n"
            "    print('\\nColumns with missing values:')\n"
            "    print(missing[missing > 0])\n"
            "else:\n"
            "    print('No missing values found ✓')"
        )
    )

    # ── 5. Feature correlations ───────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Feature Correlations with Fraud"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "corr_with_target = (\n"
            "    df[ALL_FEATURES + [TARGET]]\n"
            "    .corr()[TARGET]\n"
            "    .drop(TARGET)\n"
            "    .sort_values()\n"
            ")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 12))\n"
            "colors = ['#e74c3c' if v < 0 else '#2ecc71'\n"
            "          for v in corr_with_target.values]\n"
            "corr_with_target.plot.barh(ax=ax, color=colors)\n"
            "ax.set_title('Feature Correlation with Fraud (Pearson)')\n"
            "ax.set_xlabel('Correlation coefficient')\n"
            "ax.axvline(x=0, color='black', linewidth=0.5)\n"
            "plt.tight_layout()\n"
            "plt.savefig(DOCS_DIR / 'feature_correlations.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()\n"
            "\n"
            "print('\\nTop 10 features most correlated with fraud (abs):')\n"
            "print(corr_with_target.abs().sort_values(ascending=False)"
            ".head(10).round(4))"
        )
    )

    # ── 6. V-feature heatmap ──────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. V-Feature Correlation Heatmap\n"
            "\n"
            "V1–V28 are PCA-transformed, so they should be largely uncorrelated."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(14, 12))\n"
            "pca_cols = [f'V{i}' for i in range(1, 29)]\n"
            "corr_matrix = df[pca_cols].corr()\n"
            "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n"
            "sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r',\n"
            "            center=0, square=True, linewidths=0.5,\n"
            "            ax=ax, cbar_kws={'shrink': 0.8})\n"
            "ax.set_title('V-Feature Pairwise Correlations')\n"
            "plt.tight_layout()\n"
            "plt.savefig(DOCS_DIR / 'v_feature_heatmap.png',\n"
            "            dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    )

    # ── Summary ───────────────────────────────────────────────────
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Summary\n"
            "\n"
            "| Finding | Detail |\n"
            "|---|---|\n"
            "| Class imbalance | ~0.17 % fraud — extreme imbalance |\n"
            "| Amount | Fraud amounts tend lower; legitimate has a long tail |\n"
            "| Time | Two visible daily cycles; fraud spread throughout |\n"
            "| Missing values | None |\n"
            "| Key features | V14, V17, V12, V10 most correlated with fraud |\n"
            "| PCA features | Largely uncorrelated as expected |\n"
            "\n"
            "**Next step →** Phase 2: baseline models with proper class-imbalance handling."
        )
    )

    nb.cells = cells

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook created at {NOTEBOOK_PATH}")


if __name__ == "__main__":
    make_notebook()
