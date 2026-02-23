import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import numpy as np
import argparse

# --- 1. Define the Custom Colormap ---
def get_flood_colormap():
    """
    Recreates the exact gradient from your water_depth_floodnet script.
    Low Values  -> Dark Blue-Green (R=0.0, G=0.2, B=0.3)
    High Values -> Bright Cyan     (R=0.2, G=1.0, B=1.0)
    """
    colors = [
        (0.0, 0.2, 0.3),  # Color at min value
        (0.2, 1.0, 1.0)   # Color at max value
    ]
    return LinearSegmentedColormap.from_list("flood_blues", colors)

def save_dem_visualization(dem_path, output_path, vmax=None):
    """
    Saves a hillshaded DEM visualization using the flood color scale.
    """
    print(f"[dem_viz] Loading: {dem_path}")
    
    try:
        with rasterio.open(dem_path) as src:
            elevation = src.read(1, masked=True)
            bounds = src.bounds
            
            # --- Auto-Scale Logic ---
            if vmax is None:
                valid_data = elevation.compressed()
                if valid_data.size > 0:
                    vmax = np.percentile(valid_data, 99)
                else:
                    vmax = np.nanmax(elevation)
            
            vmin = np.nanmin(elevation)

    except Exception as e:
        print(f"[dem_viz] Error loading DEM: {e}")
        return

    # 1. Setup Hillshading
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = get_flood_colormap()

    # 2. Render Image
    rgb = ls.shade(
        elevation, 
        cmap=cmap, 
        vert_exag=0.5, 
        blend_mode='soft',
        vmin=vmin,
        vmax=vmax
    )

    # 3. Plot and Save (Headless)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    ax.set_title("DEM Visualization", fontsize=15, fontweight='bold', color='#333333')
    ax.axis('off')
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (m)')

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig) # Clear memory
    print(f"[dem_viz] Saved visualization -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True, help="Path to input DEM (.tif)")
    parser.add_argument("--out", required=True, help="Path to output image (.png)")
    args = parser.parse_args()

    save_dem_visualization(args.dem, args.out)