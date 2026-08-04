"""
One-off script to generate images/latency.png and images/pairwise_results.png
from real committed results. Not part of the reproducibility pipeline (doesn't
need to run again unless the underlying numbers change) — kept for provenance,
not wired into reproduce.py.
"""
import matplotlib.pyplot as plt

# Validated categorical palette (dataviz skill, slots 1-3 — all-pairs safe in both modes)
COLORS = {
    "LangChain": "#2a78d6",   # blue
    "LlamaIndex": "#eb6834",  # orange
    "DSPy": "#1baf7a",        # aqua
}
FRAMEWORKS = ["LangChain", "LlamaIndex", "DSPy"]

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e5e4de",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


# ---------------------------------------------------------------------------
# latency.png — retrieval median + generation median, side by side (avoids
# dual-axis: two separate small-multiple panels instead of one chart with
# two y-scales)
# ---------------------------------------------------------------------------
retrieval_median = {"LangChain": 117.2, "LlamaIndex": 386.5, "DSPy": 119.3}
generation_median = {"LangChain": 1634.6, "LlamaIndex": 1130.1, "DSPy": 3580.0}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

for ax, data, title, unit in [
    (ax1, retrieval_median, "Retrieval latency (median)", "ms"),
    (ax2, generation_median, "Generation latency (median)", "ms"),
]:
    vals = [data[fw] for fw in FRAMEWORKS]
    bars = ax.bar(FRAMEWORKS, vals, color=[COLORS[fw] for fw in FRAMEWORKS], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}ms",
                 ha="center", va="bottom", fontsize=9.5, color="#0b0b0b")
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(vals) * 1.18)
    style_ax(ax)

fig.suptitle("Latency by framework — 450-query benchmark, concurrent load (RPS=5, 8 workers)",
             fontsize=11, y=1.03)
fig.text(0.5, -0.02,
          "Measured under shared vLLM endpoint — queue wait included, not isolated per-framework cost. See docs/latency-analysis.md.",
          ha="center", fontsize=8.5, color="#52514e")
fig.tight_layout()
fig.savefig("images/latency.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved images/latency.png")

# ---------------------------------------------------------------------------
# pairwise_results.png — 3 matchups, each a pair of bars colored by framework
# identity (not by "winner"/"loser") so color stays consistent across the
# whole chart
# ---------------------------------------------------------------------------
matchups = [
    ("LangChain vs\nLlamaIndex", "LangChain", 87, "LlamaIndex", 51),
    ("LangChain vs\nDSPy", "LangChain", 85, "DSPy", 51),
    ("LlamaIndex vs\nDSPy", "LlamaIndex", 82, "DSPy", 52),
]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = range(len(matchups))
bar_w = 0.32

for i, (label, fw_a, score_a, fw_b, score_b) in enumerate(matchups):
    xa, xb = i - bar_w / 2, i + bar_w / 2
    ax.bar(xa, score_a, width=bar_w, color=COLORS[fw_a])
    ax.bar(xb, score_b, width=bar_w, color=COLORS[fw_b])
    ax.text(xa, score_a, str(score_a), ha="center", va="bottom", fontsize=10, color="#0b0b0b")
    ax.text(xb, score_b, str(score_b), ha="center", va="bottom", fontsize=10, color="#0b0b0b")

ax.set_xticks(list(x))
ax.set_xticklabels([m[0] for m in matchups], fontsize=10)
ax.set_ylabel("Wins (of 143 questions)")
ax.set_ylim(0, 100)
ax.set_title("Pairwise preference — Qwen3-14B judge picks the better answer directly",
              fontsize=11.5, pad=12)
style_ax(ax)

# Legend by framework identity, fixed order, only for frameworks that appear
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[fw]) for fw in FRAMEWORKS]
ax.legend(handles, FRAMEWORKS, frameon=False, loc="upper right", fontsize=9.5)

fig.text(0.5, -0.03,
          "LangChain wins every head-to-head matchup. Total wins: LangChain 172, LlamaIndex 133, DSPy 103.",
          ha="center", fontsize=8.5, color="#52514e")
fig.tight_layout()
fig.savefig("images/pairwise_results.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved images/pairwise_results.png")
