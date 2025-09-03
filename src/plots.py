import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ Style Settings ------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ------------------ Data ------------------

# Table 3 - CSQA and GSM8K results (with Claude Sonnet 3.7 where available)
table3 = pd.DataFrame({
    "Dataset": ["CSQA", "CSQA", "GSM8K", "GSM8K"],
    "Prompting": ["Chain of Thought", "Baseline", "Chain of Thought", "Baseline"],
    "InstructGPT": [73.5, 79.5, 46.9, 15.6],
    "PaLM-540B": [79.9, 78.1, 56.9, 17.9],
    "GPT-4.1": [93.77, 93.16, 94.88, 94.81],
    "GPT-4o": [93.07, 92.72, 94.43, 94.13],
    "Claude-Sonnet-3.7": [np.nan, np.nan, 96.09, 96.16]  # only GSM8K data available
})

# Table 4 - Ablation results
table4 = pd.DataFrame({
    "Ablation": ["Variable Compute", "Equation Only", "Reasoning Post Answer"],
    "LaMDA 137B": [6.4, 5.4, 6.1],
    "GPT-4.1": [47.10, 79.01, 48.76],
    "Claude-Sonnet-3.7": [68.16, 96.45, 64.03]
})

# Table 5 - Out-of-Distribution results (CoT, Baseline + new PaLM benchmark)
table5 = pd.DataFrame({
    "Problem": ["Last Name Concatenation", "Coin Flip"],
    "CoT": [63.0, 100.0],
    "Baseline": [0.0, 98.0],
})

# ------------------ Colors ------------------
soft_colors = ['#FFB3D1', '#B3E5D1', '#D1C4E9', '#FFCCCB', '#FFF2B3', '#B3D9FF']
accent_colors = ['#FF8FA3', '#4ECDC4', '#9575CD', '#FF6B6B', '#FFE066', '#66B2FF']

# ------------------ Plots Directory ------------------
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plots_dir, exist_ok=True)

# =====================================================
# Plot 1: CSQA Results
# =====================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.patch.set_facecolor('#FEFEFE')

csqa_data = table3[table3["Dataset"] == "CSQA"]
x_csqa = np.arange(len(csqa_data["Prompting"]))
width = 0.13

model_cols_csqa = [col for col in table3.columns if col not in ["Dataset", "Prompting"]]
bars_csqa = []
for i, model in enumerate(model_cols_csqa):
    bars = ax1.bar(
        x_csqa + width * (i - len(model_cols_csqa) / 2),
        csqa_data[model],
        width,
        label=model,
        color=soft_colors[i % len(soft_colors)],
        edgecolor=accent_colors[i % len(accent_colors)],
        linewidth=1.5,
        alpha=0.9
    )
    bars_csqa.append(bars)

ax1.set_xlabel('Prompting Method', fontsize=13, color='#4A4A4A')
ax1.set_ylabel('Accuracy (%)', fontsize=13, color='#4A4A4A')
ax1.set_title('CSQA Results Comparison', fontsize=16, weight='bold', color='#2E2E2E', pad=20)
ax1.set_xticks(x_csqa)
ax1.set_xticklabels(csqa_data["Prompting"], color='#4A4A4A')
ax1.set_ylim(0, 100)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#E0E0E0')
ax1.spines['bottom'].set_color('#E0E0E0')
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#FDFDFD')

# Legend below
ax1.legend(
    fontsize=11, framealpha=0.9, fancybox=True, shadow=True,
    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_cols_csqa)
)

# Add value labels
for bars in bars_csqa:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                     color='#4A4A4A', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'csqa_results_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# Plot 2: GSM8K Results
# =====================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor('#FEFEFE')

gsm8k_data = table3[table3["Dataset"] == "GSM8K"]
x_gsm8k = np.arange(len(gsm8k_data["Prompting"]))
bars_gsm8k = []
for i, model in enumerate(model_cols_csqa):
    bars = ax2.bar(
        x_gsm8k + width * (i - len(model_cols_csqa) / 2),
        gsm8k_data[model],
        width,
        label=model,
        color=soft_colors[i % len(soft_colors)],
        edgecolor=accent_colors[i % len(accent_colors)],
        linewidth=1.5,
        alpha=0.9
    )
    bars_gsm8k.append(bars)

ax2.set_xlabel('Prompting Method', fontsize=13, color='#4A4A4A')
ax2.set_ylabel('Accuracy (%)', fontsize=13, color='#4A4A4A')
ax2.set_title('GSM8K Results Comparison', fontsize=16, weight='bold', color='#2E2E2E', pad=20)
ax2.set_xticks(x_gsm8k)
ax2.set_xticklabels(gsm8k_data["Prompting"], color='#4A4A4A')
ax2.set_ylim(0, 100)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#E0E0E0')
ax2.spines['bottom'].set_color('#E0E0E0')
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#FDFDFD')

# Legend below
ax2.legend(
    fontsize=11, framealpha=0.9, fancybox=True, shadow=True,
    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_cols_csqa)
)

# Add value labels
for bars in bars_gsm8k:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                     color='#4A4A4A', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'gsm8k_results_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# Plot 3: Ablation Results
# =====================================================
fig3, ax_ablation = plt.subplots(figsize=(12, 7))
fig3.patch.set_facecolor('#FEFEFE')

x = np.arange(len(table4))
model_cols_ablation = [col for col in table4.columns if col != "Ablation"]
bars_ablation = []
for i, model in enumerate(model_cols_ablation):
    bars = ax_ablation.bar(
        x + width * (i - len(model_cols_ablation) / 2),
        table4[model],
        width,
        label=model,
        color=soft_colors[i % len(soft_colors)],
        edgecolor=accent_colors[i % len(accent_colors)],
        linewidth=2,
        alpha=0.9
    )
    bars_ablation.append(bars)

ax_ablation.set_xlabel('Ablation Method', fontsize=13, color='#4A4A4A')
ax_ablation.set_ylabel("Accuracy (%)", fontsize=13, color='#4A4A4A')
ax_ablation.set_title("Ablation Results Comparison", fontsize=16, weight="bold", color='#2E2E2E', pad=20)
ax_ablation.set_xticks(x)
ax_ablation.set_xticklabels(table4["Ablation"], fontsize=11, color='#4A4A4A')
ax_ablation.set_ylim(0, 105)
ax_ablation.spines['top'].set_visible(False)
ax_ablation.spines['right'].set_visible(False)
ax_ablation.spines['left'].set_color('#E0E0E0')
ax_ablation.spines['bottom'].set_color('#E0E0E0')
ax_ablation.grid(True, alpha=0.3)
ax_ablation.set_facecolor('#FDFDFD')
plt.setp(ax_ablation.get_xticklabels(), rotation=15, ha="right")

# Legend below
ax_ablation.legend(
    fontsize=12, framealpha=0.9, fancybox=True, shadow=True,
    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_cols_ablation)
)

# Add value labels
for bars in bars_ablation:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax_ablation.text(bar.get_x() + bar.get_width() / 2., height + 1,
                             f'{height:.1f}', ha='center', va='bottom', fontsize=11,
                             color='#4A4A4A', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ablation_results_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# Plot 4: Out-of-Distribution Results
# =====================================================
fig4, ax3 = plt.subplots(figsize=(9, 6))
fig4.patch.set_facecolor('#FEFEFE')

x = np.arange(len(table5))
model_cols_ood = [col for col in table5.columns if col != "Problem"]
bars_ood = []
for i, model in enumerate(model_cols_ood):
    bars = ax3.bar(
        x + width * (i - len(model_cols_ood) / 2),
        table5[model],
        width,
        label=model,
        color=soft_colors[i % len(soft_colors)],
        edgecolor=accent_colors[i % len(accent_colors)],
        linewidth=2,
        alpha=0.9
    )
    bars_ood.append(bars)

ax3.set_xlabel('Problem Type', fontsize=13, color='#4A4A4A')
ax3.set_xticks(x)
ax3.set_xticklabels(table5["Problem"], fontsize=11, color='#4A4A4A')
ax3.set_ylabel("Accuracy (%)", fontsize=13, color='#4A4A4A')
ax3.set_title("Out-of-Distribution Accuracy", fontsize=16, weight="bold", color='#2E2E2E', pad=20)
ax3.set_ylim(0, 105)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#E0E0E0')
ax3.spines['bottom'].set_color('#E0E0E0')
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('#FDFDFD')
plt.yticks(color='#4A4A4A')

# Legend below
ax3.legend(
    fontsize=12, framealpha=0.9, fancybox=True, shadow=True,
    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(model_cols_ood)
)

# Add value labels
for bars in bars_ood:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                     f'{height:.0f}%', ha='center', va='bottom', fontsize=11,
                     color='#4A4A4A', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ood_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
