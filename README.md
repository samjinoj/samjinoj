import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

def rasterize_shapefile(gdf, value_column, bounds, resolution, fill_value=np.nan):
    """
    Rasterize a GeoDataFrame.
    """
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)
    
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]
    
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill_value,
        dtype='float32'
    )
    
    return raster, transform

def estimate_fish_stock_change_spatial(parameters):
    """
    Estimate spatial fish stock change based on environmental parameter rasters.
    """
    # Define optimal ranges
    optimal_ranges = {
        "temperature": (15, 25),
        "salinity": (30, 35),
        "dissolved_oxygen": (5, 8),
        "water_currents": (0.1, 1.0),
        "sea_level_rise": (0, 20),
        "turbidity": (0, 10)
    }
    
    # Weights
    weights = {
        "temperature": 0.25,
        "salinity": 0.20,
        "dissolved_oxygen": 0.25,
        "water_currents": 0.15,
        "sea_level_rise": 0.10,
        "turbidity": 0.05
    }
    
    # Initialize score array
    example_key = list(parameters.keys())[0]
    score = np.zeros_like(parameters[example_key], dtype=float)
    
    for name, array in parameters.items():
        lower, upper = optimal_ranges[name]
        optimal = (array >= lower) & (array <= upper)
        
        deviation = np.minimum(np.abs(array - lower), np.abs(array - upper))
        range_span = upper - lower
        impact = np.where(optimal, 1.0, np.maximum(0, 1 - (deviation / range_span)))
        
        score += impact * weights[name]
    
    fish_stock_change = (score - 0.5) * 200  # -100% to +100%
    
    return fish_stock_change

def save_geotiff(output_path, array, transform, crs):
    """
    Save array as GeoTIFF.
    """
    height, width = array.shape
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(array, 1)

# --------------------------
# Example main workflow
# --------------------------

# Load shapefiles
temperature_gdf = gpd.read_file('temperature.shp')
salinity_gdf = gpd.read_file('salinity.shp')
dissolved_oxygen_gdf = gpd.read_file('dissolved_oxygen.shp')
water_currents_gdf = gpd.read_file('water_currents.shp')
sea_level_rise_gdf = gpd.read_file('sea_level_rise.shp')
turbidity_gdf = gpd.read_file('turbidity.shp')

# Common settings
resolution = 0.001  # spatial resolution in CRS units (degrees/meters)
# Common bounding box (union of all shapefiles)
bounds = temperature_gdf.total_bounds  # (minx, miny, maxx, maxy)

# Rasterize each shapefile
temperature_raster, transform = rasterize_shapefile(temperature_gdf, 'value', bounds, resolution)
salinity_raster, _ = rasterize_shapefile(salinity_gdf, 'value', bounds, resolution)
dissolved_oxygen_raster, _ = rasterize_shapefile(dissolved_oxygen_gdf, 'value', bounds, resolution)
water_currents_raster, _ = rasterize_shapefile(water_currents_gdf, 'value', bounds, resolution)
sea_level_rise_raster, _ = rasterize_shapefile(sea_level_rise_gdf, 'value', bounds, resolution)
turbidity_raster, _ = rasterize_shapefile(turbidity_gdf, 'value', bounds, resolution)

# Stack rasters into parameters dictionary
parameters = {
    "temperature": temperature_raster,
    "salinity": salinity_raster,
    "dissolved_oxygen": dissolved_oxygen_raster,
    "water_currents": water_currents_raster,
    "sea_level_rise": sea_level_rise_raster,
    "turbidity": turbidity_raster
}

# Calculate fish stock change
fish_stock_change = estimate_fish_stock_change_spatial(parameters)

# Save output to GeoTIFF
output_tif_path = 'fish_stock_change.tif'
save_geotiff(output_tif_path, fish_stock_change, transform, temperature_gdf.crs)

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(fish_stock_change, cmap='coolwarm', vmin=-100, vmax=100)
plt.colorbar(label='Fish Stock Change (%)')
plt.title('Spatial Fish Stock Change Estimate')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print(f"Fish stock change map saved to: {output_tif_path}")
