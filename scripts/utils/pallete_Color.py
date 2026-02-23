import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 1. Define labels and normalized RGB colors (divided by 255.0)
labels = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
# 0 background (transparent if you write RGBA), 1..4 damage classes
PALETTE = {
    0: (0, 0, 0),
    1: (240, 249, 33),   # No Damage  (yellow)
    2: (252, 148, 65),   # Minor      (orange)
    3: (204, 71, 120),   # Major      (magenta)
    4: (126, 3, 168),    # Destroyed  (purple)
}
colors_rgb_norm = [tuple(c / 255.0 for c in PALETTE[i]) for i in range(1, 5)]

# 2. Create a ListedColormap for plotting data
damage_cmap = mcolors.ListedColormap(colors_rgb_norm)

# --- How to create the legend panel visually ---

# Create patches for the legend
patches = [
    mpatches.Patch(color=colors_rgb_norm[0], label=labels[0]),
    mpatches.Patch(color=colors_rgb_norm[1], label=labels[1]),
    mpatches.Patch(color=colors_rgb_norm[2], label=labels[2]),
    mpatches.Patch(color=colors_rgb_norm[3], label=labels[3]),
]

# Example Plot setup to display the palette panel
fig, ax = plt.subplots(figsize=(4, 3))
ax.legend(handles=patches, loc='center', title="Damage Severity Palette", frameon=True, fontsize=12)
ax.axis('off') # Hide axes
plt.savefig("damage_severity_palette.png", bbox_inches='tight', dpi=300)