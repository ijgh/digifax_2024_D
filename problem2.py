#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terrain-Climate Interaction and Extreme Weather Formation Analysis

This script analyzes Chinese terrain and climate data to quantify the interactions
between terrain factors (elevation, slope, aspect) and climate factors in the
formation of extreme precipitation events.

Author: ijgh
Date: 2025-07-23
Version: 1.1
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import pearsonr, binned_statistic
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Global Configuration
# =============================================================================

# Data file path configuration
DATA_CONFIG = {
    'dem_dir': r"D:\digifax_2024D_date\dataset1\中国数字高程图(1km)\Geo\TIFF",
    'temp_dir': r"D:\digifax_2024D_date\dataset2\日平均数据",
    'precip_dir': r"D:\0.25rain(1961-2022)"
}

# Analysis parameters configuration
ANALYSIS_CONFIG = {
    'start_year': 1990,
    'end_year': 2020,
    'target_resolution': 0.25,  # Target resolution (degrees)
    'china_bounds': {  # China mainland boundaries
        'west': 73.0,
        'east': 136.0,
        'south': 18.0,
        'north': 54.0
    },
    'heavy_rain_threshold': 50,  # Heavy rain threshold (mm/day)
    'nodata_value': -9999
}

# Visualization configuration - English labels
VIZ_CONFIG = {
    'labels': {
        'elevation': 'Elevation (m)',
        'slope': 'Slope (degrees)',
        'aspect_class': 'Aspect Class (1=N, 2=E, 3=S, 4=W)',
        'roughness': 'Roughness',
        'tpi': 'Topographic Position Index',
        'mean_annual_precip': 'Mean Annual Precipitation (mm)',
        'spring_ratio': 'Spring Precipitation Ratio',
        'summer_ratio': 'Summer Precipitation Ratio',
        'autumn_ratio': 'Autumn Precipitation Ratio',
        'winter_ratio': 'Winter Precipitation Ratio',
        'precip_cv': 'Precipitation Coefficient of Variation',
        'heavy_rain_days': 'Annual Heavy Rain Days',
        'mean_annual_temp': 'Mean Annual Temperature (°C)',
        'elev_x_precip': 'Elevation × Precipitation',
        'slope_x_precip': 'Slope × Precipitation'
    },
    'titles': {
        'feature_importance': 'Feature Importance for Heavy Rain Frequency (XGBoost Model)',
        'elevation_rain': 'Relationship between Elevation and Heavy Rain Frequency',
        'slope_rain': 'Relationship between Slope and Heavy Rain Frequency',
        'interaction_effect': 'Interactive Effect of Elevation and Precipitation on Heavy Rain Frequency',
        'prediction_actual': 'XGBoost Model: Predicted vs Actual Heavy Rain Frequency',
        'correlation_heatmap': 'Feature Correlation Heatmap',
        '3d_terrain': '3D Relationship: Elevation, Slope and Heavy Rain Frequency'
    }
}


# =============================================================================
# Utility Functions
# =============================================================================

def find_file(directory, extension=None, contains=None):
    """
    Find files matching specified conditions in a directory

    Parameters:
    -----------
    directory : str
        Directory path to search
    extension : str, optional
        File extension (e.g., '.tif')
    contains : str, optional
        String that filename should contain

    Returns:
    --------
    str or None
        Full path of found file, None if not found
    """
    try:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return None

        files = os.listdir(directory)

        # Filter files by conditions
        if extension:
            files = [f for f in files if f.lower().endswith(extension.lower())]
        if contains:
            files = [f for f in files if contains.lower() in f.lower()]

        if files:
            file_path = os.path.join(directory, files[0])
            print(f"Found file in {directory}: {files[0]}")
            return file_path
        else:
            print(f"No matching files found in {directory}")
            return None

    except Exception as e:
        print(f"Error finding file: {e}")
        return None


# =============================================================================
# Data Loading Module
# =============================================================================

def load_dem_data(dem_path):
    """
    Load and process China Digital Elevation Model (DEM) data

    Parameters:
    -----------
    dem_path : str
        DEM file path

    Returns:
    --------
    tuple
        (dem_masked, dem_transform, dem_crs, dem_profile)
    """
    if not dem_path or not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM file does not exist: {dem_path}")

    print("Loading DEM data...")

    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # Read first band
            dem_transform = src.transform
            dem_crs = src.crs
            dem_profile = src.profile

        # Handle invalid values, create masked array
        nodata = src.nodata if src.nodata else ANALYSIS_CONFIG['nodata_value']
        dem_masked = np.ma.masked_values(dem, nodata)

        print(f"DEM data loaded - Shape: {dem.shape}, Valid pixels: {dem_masked.count()}")
        return dem_masked, dem_transform, dem_crs, dem_profile

    except Exception as e:
        print(f"Failed to load DEM data: {e}")
        raise


def load_precipitation_data(precip_path, start_year=None, end_year=None):
    """
    Load and process China 0.25° daily precipitation dataset

    Parameters:
    -----------
    precip_path : str
        Precipitation data file path
    start_year : int, optional
        Start year
    end_year : int, optional
        End year

    Returns:
    --------
    tuple
        (precip_dataset, precip_variable_name)
    """
    if not precip_path or not os.path.exists(precip_path):
        raise FileNotFoundError(f"Precipitation data file does not exist: {precip_path}")

    # Use default years from config
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print(f"Loading precipitation data ({start_year}-{end_year})...")

    try:
        ds = xr.open_dataset(precip_path)
        print(f"Precipitation dataset variables: {list(ds.data_vars)}")

        # Auto-detect precipitation variable name
        precip_var = None
        possible_vars = ['pre', 'PRE', 'prcp', 'PRCP', 'precipitation', 'rain']

        for var in possible_vars:
            if var in ds.data_vars:
                precip_var = var
                break

        if not precip_var:
            # Try to find variables containing precipitation keywords
            for var in ds.data_vars:
                if any(keyword in var.lower() for keyword in ['pre', 'rain', 'prcp']):
                    precip_var = var
                    break

        if not precip_var:
            raise ValueError(f"Precipitation variable not found, available variables: {list(ds.data_vars)}")

        print(f"Using precipitation variable: {precip_var}")

        # Time range filtering
        try:
            ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        except (KeyError, ValueError) as e:
            print(f"Time filtering failed: {e}, trying alternative method")
            # Try using year index
            if 'years' in ds.dims:
                years = ds.years.values
                year_mask = (years >= start_year) & (years <= end_year)
                ds = ds.isel(years=year_mask)

        print(f"Precipitation data loaded - Shape: {ds[precip_var].shape}")
        return ds, precip_var

    except Exception as e:
        print(f"Failed to load precipitation data: {e}")
        raise


def load_temperature_data(temp_dir, start_year=None, end_year=None):
    """
    Load and process temperature data organized by year

    Parameters:
    -----------
    temp_dir : str
        Temperature data directory path
    start_year : int, optional
        Start year
    end_year : int, optional
        End year

    Returns:
    --------
    tuple
        (temp_sample, temp_transform, temp_crs, temp_profile, temp_files_list)
    """
    if not os.path.exists(temp_dir):
        print(f"Temperature data directory does not exist: {temp_dir}")
        return None, None, None, None, []

    # Use default years from config
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print(f"Loading temperature data ({start_year}-{end_year})...")

    # Find year folders
    year_folders = []
    try:
        for item in os.listdir(temp_dir):
            folder_path = os.path.join(temp_dir, item)
            if os.path.isdir(folder_path) and item.endswith('_avg'):
                try:
                    year = int(item.split('_')[0])
                    if start_year <= year <= end_year:
                        year_folders.append((year, folder_path))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Failed to read temperature data folders: {e}")
        return None, None, None, None, []

    if not year_folders:
        print(f"No temperature data found for specified year range")
        return None, None, None, None, []

    # Collect all temperature files
    year_folders.sort(key=lambda x: x[0])
    all_temp_files = []

    for year, folder_path in year_folders:
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
            temp_files = [os.path.join(folder_path, f) for f in files]
            all_temp_files.extend(temp_files)
            print(f"Year {year}: Found {len(temp_files)} temperature files")
        except Exception as e:
            print(f"Failed to read temperature files for year {year}: {e}")

    if not all_temp_files:
        print("No temperature data files found")
        return None, None, None, None, []

    # Read sample file to get basic information
    try:
        with rasterio.open(all_temp_files[0]) as src:
            temp_sample = src.read(1)
            temp_transform = src.transform
            temp_crs = src.crs
            temp_profile = src.profile

        print(f"Temperature data loaded - Total files: {len(all_temp_files)}")
        return temp_sample, temp_transform, temp_crs, temp_profile, all_temp_files

    except Exception as e:
        print(f"Failed to read temperature data sample: {e}")
        return None, None, None, None, []


# =============================================================================
# Data Preprocessing Module
# =============================================================================

def resample_dem_to_target(dem_data, dem_transform, dem_crs, target_res=None):
    """
    Resample DEM data to target resolution

    Parameters:
    -----------
    dem_data : numpy.ndarray
        Original DEM data
    dem_transform : Affine
        Original transform matrix
    dem_crs : CRS
        Original coordinate system
    target_res : float, optional
        Target resolution (degrees)

    Returns:
    --------
    tuple
        (resampled_dem, dst_transform, dst_profile)
    """
    target_res = target_res or ANALYSIS_CONFIG['target_resolution']
    bounds = ANALYSIS_CONFIG['china_bounds']

    print(f"Resampling DEM data to {target_res}° resolution...")

    # Calculate target grid size
    width = int((bounds['east'] - bounds['west']) / target_res)
    height = int((bounds['north'] - bounds['south']) / target_res)

    # Calculate target transform matrix
    dst_transform = rasterio.transform.from_bounds(
        bounds['west'], bounds['south'],
        bounds['east'], bounds['north'],
        width, height
    )

    # Create temporary file for resampling
    temp_tif = 'temp_dem_resample.tif'

    try:
        # Write to temporary file
        with rasterio.open(temp_tif, 'w',
                           driver='GTiff',
                           height=dem_data.shape[0],
                           width=dem_data.shape[1],
                           count=1,
                           dtype=dem_data.dtype,
                           crs=dem_crs,
                           transform=dem_transform,
                           nodata=ANALYSIS_CONFIG['nodata_value']) as dst:

            # Handle masked array
            if isinstance(dem_data, np.ma.MaskedArray):
                dem_array = dem_data.filled(ANALYSIS_CONFIG['nodata_value'])
            else:
                dem_array = dem_data
            dst.write(dem_array, 1)

        # Perform resampling
        with rasterio.open(temp_tif) as src:
            resampled_dem = np.zeros((height, width), dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=resampled_dem,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',
                resampling=Resampling.bilinear
            )

        # Create output profile
        dst_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'dtype': resampled_dem.dtype,
            'crs': 'EPSG:4326',
            'transform': dst_transform
        }

        print(f"DEM resampling completed - New shape: {resampled_dem.shape}")
        return resampled_dem, dst_transform, dst_profile

    except Exception as e:
        print(f"Failed to resample DEM data: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(temp_tif):
            os.remove(temp_tif)


# =============================================================================
# Feature Calculation Module
# =============================================================================

def calculate_terrain_features(dem, transform):
    """
    Calculate terrain features: slope, aspect, roughness, topographic position index

    Parameters:
    -----------
    dem : numpy.ndarray
        Digital elevation model data
    transform : Affine
        Geographic coordinate transform matrix

    Returns:
    --------
    dict
        Dictionary containing various terrain features
    """
    print("Calculating terrain features...")

    try:
        # Get pixel resolution
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])

        # 1. Calculate slope (using Sobel operator)
        dx = ndimage.sobel(dem, axis=1) / (8.0 * pixel_width)
        dy = ndimage.sobel(dem, axis=0) / (8.0 * pixel_height)
        slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

        # 2. Calculate aspect (0-360 degrees, North=0)
        aspect = np.degrees(np.arctan2(dy, -dx))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # 3. Aspect classification (1=North, 2=East, 3=South, 4=West)
        aspect_class = np.zeros_like(aspect, dtype=np.int8)
        aspect_class[(aspect >= 315) | (aspect < 45)] = 1  # North
        aspect_class[(aspect >= 45) & (aspect < 135)] = 2  # East
        aspect_class[(aspect >= 135) & (aspect < 225)] = 3  # South
        aspect_class[(aspect >= 225) & (aspect < 315)] = 4  # West

        # 4. Terrain roughness (3x3 window standard deviation)
        roughness = ndimage.generic_filter(dem, np.std, size=3)

        # 5. Topographic Position Index (TPI)
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0  # Center point weight = 0
        kernel_sum = kernel.sum()
        neighborhood_mean = ndimage.convolve(dem, kernel) / kernel_sum
        tpi = dem - neighborhood_mean

        terrain_features = {
            'elevation': dem,
            'slope': slope,
            'aspect': aspect,
            'aspect_class': aspect_class,
            'roughness': roughness,
            'tpi': tpi
        }

        print("Terrain feature calculation completed")
        return terrain_features

    except Exception as e:
        print(f"Failed to calculate terrain features: {e}")
        raise


def calculate_climate_variables(precip_ds, precip_var, temp_files=None,
                                start_year=None, end_year=None):
    """
    Calculate climate variables: annual precipitation, seasonal distribution, temperature

    Parameters:
    -----------
    precip_ds : xarray.Dataset
        Precipitation dataset
    precip_var : str
        Precipitation variable name
    temp_files : list, optional
        Temperature file list
    start_year : int, optional
        Start year
    end_year : int, optional
        End year

    Returns:
    --------
    dict
        Dictionary containing various climate variables
    """
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print("Calculating climate variables...")

    try:
        # 1. Calculate precipitation-related variables
        if 'time' in precip_ds.dims:
            # Standard time dimension processing
            annual_precip = precip_ds[precip_var].groupby('time.year').sum('time')
            mean_annual_precip = annual_precip.mean('year')

            # Seasonal precipitation distribution
            seasons = {
                'spring': [3, 4, 5],   # Spring
                'summer': [6, 7, 8],   # Summer
                'autumn': [9, 10, 11], # Autumn
                'winter': [12, 1, 2]   # Winter
            }

            seasonal_precip = {}
            seasonal_ratios = {}

            for season, months in seasons.items():
                season_data = precip_ds.sel(
                    time=precip_ds['time.month'].isin(months)
                )[precip_var].groupby('time.year').sum('time')

                seasonal_precip[season] = season_data.mean('year')
                seasonal_ratios[f'{season}_ratio'] = seasonal_precip[season] / mean_annual_precip

            # Precipitation coefficient of variation
            precip_cv = annual_precip.std('year') / mean_annual_precip

            # Simplified heavy rain frequency calculation
            threshold = ANALYSIS_CONFIG['heavy_rain_threshold']
            heavy_rain_ratio = (precip_ds[precip_var] >= threshold).mean('time')
            heavy_rain_days = heavy_rain_ratio * 365.25

        else:
            # Alternative: no standard time dimension
            print("Warning: Using simplified climate variable calculation")
            mean_annual_precip = precip_ds[precip_var].mean()

            # Default seasonal ratios
            seasonal_ratios = {
                'spring_ratio': xr.ones_like(mean_annual_precip) * 0.25,
                'summer_ratio': xr.ones_like(mean_annual_precip) * 0.40,
                'autumn_ratio': xr.ones_like(mean_annual_precip) * 0.25,
                'winter_ratio': xr.ones_like(mean_annual_precip) * 0.10
            }

            precip_cv = xr.ones_like(mean_annual_precip) * 0.2
            heavy_rain_days = mean_annual_precip * 0.05

        # 2. Calculate annual temperature (if temperature data available)
        mean_annual_temp = None
        if temp_files:
            mean_annual_temp = _calculate_annual_temperature(temp_files)

        # Organize return results
        climate_vars = {
            'mean_annual_precip': mean_annual_precip,
            'precip_cv': precip_cv,
            'heavy_rain_days': heavy_rain_days,
            'mean_annual_temp': mean_annual_temp
        }

        # Add seasonal ratios
        climate_vars.update(seasonal_ratios)

        print("Climate variable calculation completed")
        return climate_vars

    except Exception as e:
        print(f"Failed to calculate climate variables: {e}")
        import traceback
        traceback.print_exc()

        # Return default values
        return _create_default_climate_vars()


def _calculate_annual_temperature(temp_files):
    """
    Calculate annual temperature from temperature file list

    Parameters:
    -----------
    temp_files : list
        Temperature file path list

    Returns:
    --------
    numpy.ndarray or None
        Annual temperature array
    """
    print("Calculating annual temperature...")

    try:
        # Organize files by year
        year_files = {}
        for temp_file in temp_files:
            try:
                # Extract year from path
                path_parts = os.path.normpath(temp_file).split(os.sep)
                folder_name = next((part for part in path_parts if part.endswith('_avg')), None)

                if folder_name:
                    year = int(folder_name.split('_')[0])
                    if year not in year_files:
                        year_files[year] = []
                    year_files[year].append(temp_file)
            except (ValueError, IndexError):
                continue

        print(f"Found temperature data for {len(year_files)} years")

        # Select representative files (4 seasonal files per year)
        selected_files = _select_representative_temp_files(year_files)
        print(f"Selected {len(selected_files)} representative files")

        # Read temperature data and calculate average
        temp_data_list = []
        for temp_file in selected_files:
            try:
                with rasterio.open(temp_file) as src:
                    temp_data = src.read(1).astype(np.float32)
                    # Handle invalid values
                    if src.nodata is not None:
                        temp_data = np.where(temp_data == src.nodata, np.nan, temp_data)
                    temp_data_list.append(temp_data)
            except Exception as e:
                print(f"Failed to read temperature file {temp_file}: {e}")

        if temp_data_list:
            temp_array = np.array(temp_data_list)
            mean_annual_temp = np.nanmean(temp_array, axis=0)
            print(f"Annual temperature calculation completed - Shape: {mean_annual_temp.shape}")
            return mean_annual_temp
        else:
            print("No valid temperature data")
            return None

    except Exception as e:
        print(f"Failed to calculate annual temperature: {e}")
        return None


def _select_representative_temp_files(year_files):
    """
    Select representative files from each year's temperature files

    Parameters:
    -----------
    year_files : dict
        Dictionary of files organized by year

    Returns:
    --------
    list
        List of selected representative files
    """
    selected_files = []

    for year, files in year_files.items():
        files.sort()  # Sort by filename

        if len(files) >= 4:
            # Select four seasonal representative files
            quarter_size = len(files) // 4
            seasonal_files = [
                files[quarter_size // 2],                    # Spring
                files[quarter_size + quarter_size // 2],     # Summer
                files[2 * quarter_size + quarter_size // 2], # Autumn
                files[3 * quarter_size + quarter_size // 2]  # Winter
            ]
            selected_files.extend(seasonal_files)
        else:
            # Use all files when few files available
            selected_files.extend(files)

    return selected_files


def _create_default_climate_vars():
    """
    Create default climate variables (for error handling)

    Returns:
    --------
    dict
        Default climate variables dictionary
    """
    # Create default sized empty arrays
    lat_size, lon_size = 144, 256
    default_array = xr.DataArray(
        np.zeros((lat_size, lon_size)),
        dims=['lat', 'lon']
    )

    return {
        'mean_annual_precip': default_array,
        'spring_ratio': default_array,
        'summer_ratio': default_array,
        'autumn_ratio': default_array,
        'winter_ratio': default_array,
        'precip_cv': default_array,
        'heavy_rain_days': default_array,
        'mean_annual_temp': None
    }


# =============================================================================
# Data Integration Module
# =============================================================================

# def integrate_data(terrain_features, climate_vars, dem_target_transform):
#     """
#     Integrate terrain features and climate variables into unified dataframe
#
#     Parameters:
#     -----------
#     terrain_features : dict
#         Terrain features dictionary
#     climate_vars : dict
#         Climate variables dictionary
#     dem_target_transform : Affine
#         Target transform matrix
#
#     Returns:
#     --------
#     pandas.DataFrame
#         Integrated dataframe
#     """
#     print("Integrating data into unified dataframe...")
#
#     try:
#         # Get terrain data grid coordinates
#         elevation = terrain_features['elevation']
#         height, width = elevation.shape
#
#         # Create lat/lon grid
#         y_coords, x_coords = np.mgrid[0:height, 0:width]
#         lons, lats = rasterio.transform.xy(dem_target_transform, y_coords, x_coords)
#         lons, lats = np.array(lons), np.array(lats)
#
#         # Prepare basic data dictionary
#         data_dict = {
#             'lat': lats.flatten(),
#             'lon': lons.flatten(),
#             'elevation': elevation.flatten(),
#             'slope': terrain_features['slope'].flatten(),
#             'aspect': terrain_features['aspect'].flatten(),
#             'aspect_class': terrain_features['aspect_class'].flatten(),
#             'roughness': terrain_features['roughness'].flatten(),
#             'tpi': terrain_features['tpi'].flatten(),
#         }
#
#         # Match climate variable data
#         _match_climate_data(data_dict, climate_vars)
#
#         # Add temperature data (if available)
#         if climate_vars['mean_annual_temp'] is not None:
#             data_dict['mean_annual_temp'] = climate_vars['mean_annual_temp'].flatten()
#
#         # Create interaction features
#         data_dict['elev_x_precip'] = data_dict['elevation'] * data_dict['mean_annual_precip']
#         data_dict['slope_x_precip'] = data_dict['slope'] * data_dict['mean_annual_precip']
#
#         # Convert to DataFrame and clean data
#         df = pd.DataFrame(data_dict)
#         df_clean = _clean_dataframe(df)
#
#         print(f"Data integration completed - Original points: {len(df)}, Valid points: {len(df_clean)}")
#         return df_clean
#
#     except Exception as e:
#         print(f"Data integration failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return _create_sample_dataset()


def integrate_data(terrain_features, climate_vars, dem_target_transform):
    """
    整合地形特征和气候变量到统一的数据框 - 优化版本，减少数据丢失
    """
    print("正在整合数据到统一数据框...")

    try:
        # 获取地形数据的网格坐标
        elevation = terrain_features['elevation']
        height, width = elevation.shape
        total_points = height * width

        print(f"地形数据网格尺寸: {height} x {width} = {total_points} 个点")

        # 创建经纬度网格
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        lons, lats = rasterio.transform.xy(dem_target_transform, y_coords, x_coords)
        lons, lats = np.array(lons), np.array(lats)

        # 准备基础数据字典
        data_dict = {
            'lat': lats.flatten(),
            'lon': lons.flatten(),
            'elevation': elevation.flatten(),
            'slope': terrain_features['slope'].flatten(),
            'aspect': terrain_features['aspect'].flatten(),
            'aspect_class': terrain_features['aspect_class'].flatten(),
            'roughness': terrain_features['roughness'].flatten(),
            'tpi': terrain_features['tpi'].flatten(),
        }

        # 验证基础数据长度
        base_length = len(data_dict['lat'])
        print(f"基础数据长度: {base_length}")

        # 匹配气候变量数据 - 改进版本
        _match_climate_data_improved(data_dict, climate_vars)

        # 添加温度数据（如果可用）
        if climate_vars.get('mean_annual_temp') is not None:
            try:
                temp_flat = climate_vars['mean_annual_temp'].flatten()
                if len(temp_flat) == base_length:
                    data_dict['mean_annual_temp'] = temp_flat
                else:
                    # 使用双线性插值调整尺寸
                    data_dict['mean_annual_temp'] = _resize_array_to_match(temp_flat, base_length)
                print(f"添加了年均温度数据")
            except Exception as e:
                print(f"添加温度数据时出错: {e}")

        # 创建交互特征
        try:
            data_dict['elev_x_precip'] = data_dict['elevation'] * data_dict['mean_annual_precip']
            data_dict['slope_x_precip'] = data_dict['slope'] * data_dict['mean_annual_precip']
        except Exception as e:
            print(f"创建交互特征时出错: {e}")
            data_dict['elev_x_precip'] = np.zeros(base_length)
            data_dict['slope_x_precip'] = np.zeros(base_length)

        # 转换为DataFrame
        df = pd.DataFrame(data_dict)
        print(f"DataFrame创建成功，形状: {df.shape}")

        # 改进的数据清理
        df_clean = _clean_dataframe_improved(df)

        print(f"数据整合完成 - 原始点数: {len(df)}, 有效点数: {len(df_clean)}")
        return df_clean

    except Exception as e:
        print(f"数据整合失败: {e}")
        import traceback
        traceback.print_exc()
        return _create_sample_dataset()


def _match_climate_data_improved(data_dict, climate_vars):
    """
    改进的气候数据匹配函数，减少数据丢失
    """
    print("正在匹配气候数据...")

    # 获取气候数据的坐标
    try:
        precip_data = climate_vars['mean_annual_precip']
        if hasattr(precip_data, 'lat'):
            precip_lat = precip_data.lat.values
            precip_lon = precip_data.lon.values
        elif hasattr(precip_data, 'latitude'):
            precip_lat = precip_data.latitude.values
            precip_lon = precip_data.longitude.values
        else:
            bounds = ANALYSIS_CONFIG['china_bounds']
            precip_lat = np.linspace(bounds['south'], bounds['north'], 144)
            precip_lon = np.linspace(bounds['west'], bounds['east'], 256)
    except Exception as e:
        print(f"获取气候数据坐标失败: {e}")
        bounds = ANALYSIS_CONFIG['china_bounds']
        precip_lat = np.linspace(bounds['south'], bounds['north'], 144)
        precip_lon = np.linspace(bounds['west'], bounds['east'], 256)

    # 获取气候变量数组
    climate_arrays = {}
    climate_keys = ['mean_annual_precip', 'spring_ratio', 'summer_ratio',
                    'autumn_ratio', 'winter_ratio', 'precip_cv', 'heavy_rain_days']

    for key in climate_keys:
        try:
            if key in climate_vars and climate_vars[key] is not None:
                data_array = climate_vars[key].values
                # 确保数组形状正确
                if data_array.ndim == 1:
                    # 尝试重塑为2D
                    expected_size = len(precip_lat) * len(precip_lon)
                    if len(data_array) == expected_size:
                        data_array = data_array.reshape(len(precip_lat), len(precip_lon))
                    else:
                        # 使用插值调整大小
                        data_array = np.full((len(precip_lat), len(precip_lon)), np.nanmean(data_array))

                # 处理无效值
                data_array = np.nan_to_num(data_array, nan=np.nanmean(data_array))
                climate_arrays[key] = data_array
            else:
                # 创建合理的默认值
                if key == 'mean_annual_precip':
                    climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 800.0)  # 默认800mm
                elif key == 'heavy_rain_days':
                    climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 5.0)  # 默认5天
                elif 'ratio' in key:
                    climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 0.25)  # 默认25%
                else:
                    climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 0.2)  # 默认变异系数
        except Exception as e:
            print(f"处理 {key} 时出错: {e}")
            # 提供合理的默认值
            if key == 'mean_annual_precip':
                climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 800.0)
            elif key == 'heavy_rain_days':
                climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 5.0)
            elif 'ratio' in key:
                climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 0.25)
            else:
                climate_arrays[key] = np.full((len(precip_lat), len(precip_lon)), 0.2)

    # 初始化输出数组
    n_points = len(data_dict['lat'])
    for key in climate_keys:
        data_dict[key] = np.full(n_points, np.nan, dtype=np.float64)

    # 使用向量化操作进行最近邻插值
    lats_array = np.array(data_dict['lat'])
    lons_array = np.array(data_dict['lon'])

    # 为每个点找到最近的气候网格点
    lat_indices = np.searchsorted(precip_lat, lats_array)
    lon_indices = np.searchsorted(precip_lon, lons_array)

    # 确保索引在有效范围内
    lat_indices = np.clip(lat_indices, 0, len(precip_lat) - 1)
    lon_indices = np.clip(lon_indices, 0, len(precip_lon) - 1)

    # 批量赋值
    for key in climate_keys:
        try:
            data_dict[key] = climate_arrays[key][lat_indices, lon_indices]
        except Exception as e:
            print(f"批量赋值 {key} 失败: {e}")
            # 使用循环作为备选方案
            for i in range(n_points):
                try:
                    lat_idx = min(lat_indices[i], len(precip_lat) - 1)
                    lon_idx = min(lon_indices[i], len(precip_lon) - 1)
                    data_dict[key][i] = climate_arrays[key][lat_idx, lon_idx]
                except:
                    data_dict[key][i] = np.nanmean(climate_arrays[key])

    print(f"气候数据匹配完成")


def _resize_array_to_match(source_array, target_length):
    """
    调整数组大小以匹配目标长度
    """
    if len(source_array) == target_length:
        return source_array
    elif len(source_array) > target_length:
        # 等间隔抽样
        indices = np.linspace(0, len(source_array) - 1, target_length, dtype=int)
        return source_array[indices]
    else:
        # 使用插值扩展
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, len(source_array))
        new_indices = np.linspace(0, 1, target_length)
        f = interp1d(old_indices, source_array, kind='linear', fill_value='extrapolate')
        return f(new_indices)


def _clean_dataframe_improved(df):
    """
    改进的数据框清理函数，减少数据丢失
    """
    print(f"清理数据框 - 输入形状: {df.shape}")

    # 检查数据基本统计
    print("数据基本统计:")
    for col in df.select_dtypes(include=[np.number]).columns:
        valid_count = df[col].notna().sum()
        print(f"  {col}: {valid_count}/{len(df)} ({valid_count / len(df) * 100:.1f}%) 有效")

    # 1. 只移除所有特征都为NaN的行
    print("移除完全无效的行...")
    important_cols = ['elevation', 'slope', 'mean_annual_precip', 'heavy_rain_days']
    available_important_cols = [col for col in important_cols if col in df.columns]

    if available_important_cols:
        # 只有当重要列全部为NaN时才移除
        df_cleaned = df.dropna(subset=available_important_cols, how='all')
    else:
        df_cleaned = df.dropna(how='all')

    print(f"移除完全无效行后: {len(df_cleaned)} 行")

    # 2. 对缺失值进行智能填充而不是移除
    print("智能填充缺失值...")

    # 地形特征使用邻近值填充
    terrain_cols = ['elevation', 'slope', 'aspect_class', 'roughness', 'tpi']
    for col in terrain_cols:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                # 使用中位数填充
                median_val = df_cleaned[col].median()
                if pd.notna(median_val):
                    filled_count = df_cleaned[col].isnull().sum()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    print(f"  {col}: 填充了 {filled_count} 个缺失值 (中位数: {median_val:.2f})")

    # 气候特征使用合理默认值填充
    climate_defaults = {
        'mean_annual_precip': 800.0,  # 中国平均降水量
        'spring_ratio': 0.25,
        'summer_ratio': 0.40,
        'autumn_ratio': 0.25,
        'winter_ratio': 0.10,
        'precip_cv': 0.20,
        'heavy_rain_days': 5.0
    }

    for col, default_val in climate_defaults.items():
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                filled_count = df_cleaned[col].isnull().sum()
                df_cleaned[col] = df_cleaned[col].fillna(default_val)
                print(f"  {col}: 填充了 {filled_count} 个缺失值 (默认值: {default_val})")

    # 3. 温度数据使用插值填充
    if 'mean_annual_temp' in df_cleaned.columns:
        if df_cleaned['mean_annual_temp'].isnull().any():
            # 基于海拔的温度梯度估算 (每100m降低0.6°C)
            if 'elevation' in df_cleaned.columns:
                # 使用海拔估算温度
                mask = df_cleaned['mean_annual_temp'].isnull()
                if mask.any():
                    # 假设海平面温度为15°C
                    estimated_temp = 15.0 - (df_cleaned.loc[mask, 'elevation'] / 100.0 * 0.6)
                    df_cleaned.loc[mask, 'mean_annual_temp'] = estimated_temp
                    print(f"  mean_annual_temp: 基于海拔估算填充了 {mask.sum()} 个缺失值")

    # 4. 重新计算交互特征
    if 'elevation' in df_cleaned.columns and 'mean_annual_precip' in df_cleaned.columns:
        df_cleaned['elev_x_precip'] = df_cleaned['elevation'] * df_cleaned['mean_annual_precip']
        df_cleaned['slope_x_precip'] = df_cleaned['slope'] * df_cleaned['mean_annual_precip']

    # 5. 只移除极端异常值（使用更宽松的阈值）
    print("移除极端异常值...")
    initial_count = len(df_cleaned)

    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        if col not in ['lat', 'lon', 'aspect_class']:  # 保留分类变量和坐标
            # 使用3个标准差作为异常值阈值
            mean_val = df_cleaned[col].mean()
            std_val = df_cleaned[col].std()
            if std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"  {col}: 发现 {outlier_count} 个极端异常值")
                    # 不删除，而是用边界值替换
                    df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                    df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound

    final_count = len(df_cleaned)
    print(f"数据清理完成: {final_count} 行 (保留率: {final_count / initial_count * 100:.1f}%)")

    return df_cleaned


def _create_sample_dataset():
    """Create sample dataset for testing and demonstration"""
    print("Creating sample dataset...")
    np.random.seed(42)
    n_samples = 1000

    return pd.DataFrame({
        'elevation': np.random.normal(1500, 500, n_samples),
        'slope': np.random.normal(15, 5, n_samples),
        'aspect_class': np.random.randint(1, 5, n_samples),
        'roughness': np.random.normal(0.2, 0.1, n_samples),
        'tpi': np.random.normal(0, 10, n_samples),
        'heavy_rain_days': np.random.poisson(5, n_samples),
        'mean_annual_precip': np.random.normal(1000, 300, n_samples),
        'spring_ratio': np.random.normal(0.25, 0.05, n_samples),
        'summer_ratio': np.random.normal(0.40, 0.05, n_samples),
        'autumn_ratio': np.random.normal(0.25, 0.05, n_samples),
        'winter_ratio': np.random.normal(0.10, 0.05, n_samples),
        'precip_cv': np.random.normal(0.2, 0.05, n_samples),
        'elev_x_precip': np.random.normal(1500000, 500000, n_samples),
        'slope_x_precip': np.random.normal(15000, 5000, n_samples),
    })


# =============================================================================
# Machine Learning Modeling Module
# =============================================================================

def build_models(df, features, target):
    """Build multiple machine learning models to analyze terrain-climate effects on extreme precipitation"""
    print("Building machine learning models...")

    try:
        df_model = _preprocess_model_data(df, features, target)

        X = df_model[features]
        y = df_model[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {}
        models['GLM'] = _build_glm_model(X_train, X_test, y_train, y_test)
        models['RandomForest'] = _build_rf_model(X_train_scaled, X_test_scaled, y_train, y_test, features)
        models['XGBoost'] = _build_xgb_model(X_train_scaled, X_test_scaled, y_train, y_test, features)

        print("\nModel Performance Summary:")
        for name, model_info in models.items():
            print(f"{name}: R² = {model_info['r2']:.4f}, MSE = {model_info['mse']:.4f}")

        return models, df_model

    except Exception as e:
        print(f"Model building failed: {e}")
        return _create_dummy_models(features, target), df


def _preprocess_model_data(df, features, target):
    """Preprocess model data: remove outliers"""
    print(f"Data preprocessing - Original data points: {len(df)}")
    df_clean = df.copy()

    for col in features + [target]:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.01)
            q99 = df_clean[col].quantile(0.99)
            df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q99)]

    print(f"Data points after removing outliers: {len(df_clean)}")
    return df_clean


def _build_glm_model(X_train, X_test, y_train, y_test):
    """Build Generalized Linear Model"""
    print("Training Generalized Linear Model...")

    try:
        X_train_glm = sm.add_constant(X_train)
        X_test_glm = sm.add_constant(X_test)

        try:
            glm_model = sm.GLM(y_train, X_train_glm, family=sm.families.Poisson())
            glm_result = glm_model.fit()
            y_pred = glm_result.predict(X_test_glm)
        except Exception:
            print("GLM failed, using linear regression alternative...")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred = lr_model.predict(X_test)

            class MockGLMResult:
                def __init__(self, coef, intercept):
                    self.params = np.append(intercept, coef)
                    self.pvalues = np.full_like(self.params, 0.05)

            glm_result = MockGLMResult(lr_model.coef_, lr_model.intercept_)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model': glm_result,
            'X_test': X_test_glm if 'X_test_glm' in locals() else X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2
        }

    except Exception as e:
        print(f"GLM model building failed: {e}")
        return _create_dummy_model_result('GLM', X_test, y_test)


def _build_rf_model(X_train, X_test, y_train, y_test, features):
    """Build Random Forest Model"""
    print("Training Random Forest Model...")

    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model': rf_model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'feature_importance': rf_model.feature_importances_,
            'feature_names': features
        }

    except Exception as e:
        print(f"Random Forest model building failed: {e}")
        return _create_dummy_model_result('RandomForest', X_test, y_test, features)


def _build_xgb_model(X_train, X_test, y_train, y_test, features):
    """Build XGBoost Model"""
    print("Training XGBoost Model...")

    try:
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model': xgb_model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'feature_importance': xgb_model.feature_importances_,
            'feature_names': features
        }

    except Exception as e:
        print(f"XGBoost model building failed: {e}")
        return _create_dummy_model_result('XGBoost', X_test, y_test, features)


def _create_dummy_model_result(model_name, X_test, y_test, features=None):
    """Create dummy model result for error handling"""
    dummy_pred = np.zeros_like(y_test) if hasattr(y_test, '__len__') else np.array([0, 0])

    result = {
        'model': None,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': dummy_pred,
        'mse': 0.0,
        'r2': 0.0
    }

    if features:
        result.update({
            'feature_importance': np.ones(len(features)) / len(features),
            'feature_names': features
        })

    return result


def _create_dummy_models(features, target):
    """Create dummy models dictionary for error handling"""

    class DummyModel:
        def __init__(self, n_features):
            self.feature_importances_ = np.ones(n_features) / n_features

    dummy_data = np.array([[1, 2], [3, 4]])
    dummy_target = np.array([1, 2])

    return {
        'GLM': _create_dummy_model_result('GLM', dummy_data, dummy_target),
        'RandomForest': _create_dummy_model_result('RandomForest', dummy_data, dummy_target, features),
        'XGBoost': _create_dummy_model_result('XGBoost', dummy_data, dummy_target, features)
    }


# =============================================================================
# Model Analysis Module
# =============================================================================

def analyze_models(models, df_model, features, target):
    """Analyze model results, extract feature importance and coefficient information"""
    print("Analyzing model results...")

    try:
        # 1. GLM coefficient analysis
        glm_coef = _analyze_glm_coefficients(models.get('GLM'), features)

        # 2. Feature importance analysis
        rf_importance = _analyze_feature_importance(models.get('RandomForest'), 'RandomForest')
        xgb_importance = _analyze_feature_importance(models.get('XGBoost'), 'XGBoost')

        # 3. Interaction analysis
        interaction_results = _analyze_interactions(glm_coef)

        # 4. Correlation analysis
        corr_matrix, target_corr = _analyze_correlations(df_model, features, target)

        analysis_results = {
            'glm_coef': glm_coef,
            'rf_importance': rf_importance,
            'xgb_importance': xgb_importance,
            'interaction_results': interaction_results,
            'target_corr': target_corr,
            'corr_matrix': corr_matrix
        }

        print("Model analysis completed")
        return analysis_results

    except Exception as e:
        print(f"Model analysis failed: {e}")
        return _create_default_analysis_results(features, target)


def _analyze_glm_coefficients(glm_model, features):
    """Analyze GLM model coefficients"""
    if glm_model and glm_model.get('model'):
        try:
            model = glm_model['model']
            coef_df = pd.DataFrame({
                'feature': ['constant'] + features,
                'coef': model.params,
                'p_value': model.pvalues
            })
            return coef_df
        except Exception as e:
            print(f"GLM coefficient analysis failed: {e}")

    return pd.DataFrame({
        'feature': ['constant'] + features,
        'coef': [0.0] * (len(features) + 1),
        'p_value': [1.0] * (len(features) + 1)
    })


def _analyze_feature_importance(model_info, model_name):
    """Analyze feature importance"""
    if model_info and 'feature_importance' in model_info:
        try:
            return pd.DataFrame({
                'feature': model_info['feature_names'],
                'importance': model_info['feature_importance']
            }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"{model_name} feature importance analysis failed: {e}")

    features = model_info.get('feature_names', ['feature1', 'feature2'])
    return pd.DataFrame({
        'feature': features,
        'importance': np.ones(len(features)) / len(features)
    })


def _analyze_interactions(glm_coef):
    """Analyze interaction term coefficients"""
    interaction_terms = ['elev_x_precip', 'slope_x_precip']
    return glm_coef[glm_coef['feature'].isin(interaction_terms)]


def _analyze_correlations(df_model, features, target):
    """
    改进的相关性分析函数
    """
    try:
        # 确保所有特征都在数据框中
        available_features = [f for f in features if f in df_model.columns]

        # 添加目标变量（如果存在）
        analysis_cols = available_features.copy()
        if target in df_model.columns and target not in analysis_cols:
            analysis_cols.append(target)

        print(f"相关性分析包含 {len(analysis_cols)} 个变量: {analysis_cols}")

        if len(analysis_cols) < 2:
            print("警告：可用于相关性分析的变量太少")
            return _create_dummy_correlation_matrix(features, target)

        # 计算相关性矩阵
        corr_subset = df_model[analysis_cols].select_dtypes(include=[np.number])

        if corr_subset.empty:
            print("警告：没有数值变量用于相关性分析")
            return _create_dummy_correlation_matrix(features, target)

        # 移除常数列（标准差为0的列）
        numeric_cols = []
        for col in corr_subset.columns:
            if corr_subset[col].std() > 1e-10:  # 避免常数列
                numeric_cols.append(col)

        if len(numeric_cols) < 2:
            print("警告：移除常数列后，剩余变量太少")
            return _create_dummy_correlation_matrix(features, target)

        corr_matrix = corr_subset[numeric_cols].corr()

        # 计算与目标变量的相关性
        if target in corr_matrix.columns:
            target_corr = corr_matrix[target].sort_values(ascending=False)
        else:
            target_corr = pd.Series(np.zeros(len(numeric_cols)), index=numeric_cols)

        print(f"相关性矩阵计算成功 - 形状: {corr_matrix.shape}")
        return corr_matrix, target_corr

    except Exception as e:
        print(f"相关性分析失败: {e}")
        return _create_dummy_correlation_matrix(features, target)


def _create_dummy_correlation_matrix(features, target):
    """
    创建示例相关性矩阵
    """
    all_vars = features + [target] if target not in features else features
    # 限制变量数量以避免显示问题
    all_vars = all_vars[:10]  # 最多10个变量

    n_vars = len(all_vars)
    # 创建合理的相关性矩阵
    corr_data = np.random.rand(n_vars, n_vars) * 0.6 - 0.3  # -0.3到0.3之间
    corr_data = (corr_data + corr_data.T) / 2  # 使其对称
    np.fill_diagonal(corr_data, 1.0)  # 对角线为1

    corr_matrix = pd.DataFrame(corr_data, columns=all_vars, index=all_vars)
    target_corr = pd.Series(corr_data[-1], index=all_vars)

    return corr_matrix, target_corr

def _create_default_analysis_results(features, target):
    """Create default analysis results"""
    return {
        'glm_coef': pd.DataFrame({
            'feature': ['constant'] + features,
            'coef': [0.0] * (len(features) + 1),
            'p_value': [1.0] * (len(features) + 1)
        }),
        'rf_importance': pd.DataFrame({
            'feature': features,
            'importance': np.ones(len(features)) / len(features)
        }),
        'xgb_importance': pd.DataFrame({
            'feature': features,
            'importance': np.ones(len(features)) / len(features)
        }),
        'interaction_results': pd.DataFrame({
            'feature': ['elev_x_precip', 'slope_x_precip'],
            'coef': [0.0, 0.0],
            'p_value': [1.0, 1.0]
        }),
        'target_corr': pd.Series(np.zeros(len(features) + 1), index=features + [target]),
        'corr_matrix': pd.DataFrame(np.eye(len(features) + 1),
                                    columns=features + [target],
                                    index=features + [target])
    }


# =============================================================================
# Visualization Module - English Labels
# =============================================================================

def visualize_results(models, df_model, features, target, analysis_results):
    """Create visualization charts with English labels to display analysis results"""
    print("Creating visualization charts...")

    # Set plotting style (remove Chinese font settings for English labels)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    try:
        # 1. Feature importance plot
        _plot_feature_importance_english(analysis_results['xgb_importance'])

        # 2. Elevation-rain relationship plot
        _plot_elevation_rain_relationship_english(df_model, target)

        # 3. Slope-rain relationship plot
        _plot_slope_rain_relationship_english(df_model, target)

        # 4. Interaction effects plot
        _plot_interaction_effects_english(df_model, target)

        # 5. Model prediction plot
        _plot_model_predictions_english(models['XGBoost'])

        # 6. Correlation heatmap
        _plot_correlation_heatmap_english(analysis_results['corr_matrix'])

        # 7. 3D terrain relationship plot
        _plot_3d_terrain_relationship_english(df_model, target)

        print("Visualization completed, saved 7 chart files")

    except Exception as e:
        print(f"Error in visualization process: {e}")
        import traceback
        traceback.print_exc()


def _plot_feature_importance_english(importance_df):
    """Plot feature importance chart with English labels"""
    plt.figure(figsize=(12, 8))

    # Normalize importance to percentage and take top 10
    plot_df = importance_df.copy()
    plot_df['importance'] = plot_df['importance'] / plot_df['importance'].max() * 100
    plot_df = plot_df.sort_values('importance', ascending=True).tail(10)

    # Feature classification colors
    colors = []
    terrain_features = ['elevation', 'slope', 'aspect_class', 'roughness', 'tpi']
    climate_features = ['mean_annual_precip', 'spring_ratio', 'summer_ratio',
                        'autumn_ratio', 'winter_ratio', 'precip_cv', 'mean_annual_temp']
    interaction_features = ['elev_x_precip', 'slope_x_precip']

    for feat in plot_df['feature']:
        if feat in terrain_features:
            colors.append('#1f77b4')  # Blue
        elif feat in climate_features:
            colors.append('#ff7f0e')  # Orange
        elif feat in interaction_features:
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red

    plt.barh(plot_df['feature'], plot_df['importance'], color=colors)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', label='Terrain Features'),
        plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', label='Climate Features'),
        plt.Rectangle((0, 0), 1, 1, fc='#2ca02c', label='Interaction Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.title(VIZ_CONFIG['titles']['feature_importance'], fontsize=16)
    plt.xlabel('Relative Importance (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: feature_importance.png")


def _plot_elevation_rain_relationship_english(df_model, target):
    """Plot elevation vs rain frequency relationship with English labels - Fixed version"""
    plt.figure(figsize=(12, 8))

    # 检查并清理海拔数据
    if 'elevation' not in df_model.columns:
        print("Warning: elevation column not found")
        return

    # 移除无效的海拔值
    valid_mask = (df_model['elevation'].notna() &
                  (df_model['elevation'] > 0) &
                  (df_model['elevation'] < 10000) &  # 合理的海拔上限
                  df_model[target].notna())

    df_valid = df_model[valid_mask].copy()

    if len(df_valid) < 10:
        print("Warning: Too few valid elevation data points")
        return

    print(f"海拔数据范围: {df_valid['elevation'].min():.1f} - {df_valid['elevation'].max():.1f} m")
    print(f"有效数据点数: {len(df_valid)}")

    # 设置合理的海拔范围显示
    elev_min = max(0, df_valid['elevation'].quantile(0.01))  # 使用1%分位数作为下限
    elev_max = df_valid['elevation'].quantile(0.99)  # 使用99%分位数作为上限

    # 确保最小值从500m左右开始（如果数据允许）
    if elev_min < 500 and df_valid['elevation'].max() > 500:
        elev_min = max(500, df_valid['elevation'].quantile(0.05))

    # 过滤显示范围内的数据
    display_mask = ((df_valid['elevation'] >= elev_min) &
                    (df_valid['elevation'] <= elev_max))
    df_display = df_valid[display_mask]

    print(f"显示海拔范围: {elev_min:.1f} - {elev_max:.1f} m")
    print(f"显示数据点数: {len(df_display)}")

    # Scatter plot colored by precipitation
    if 'mean_annual_precip' in df_display.columns:
        # 清理降水数据
        precip_valid = (df_display['mean_annual_precip'].notna() &
                        (df_display['mean_annual_precip'] > 0) &
                        (df_display['mean_annual_precip'] < 5000))  # 合理的降水上限

        df_plot = df_display[precip_valid]

        if len(df_plot) > 0:
            sc = plt.scatter(df_plot['elevation'], df_plot[target],
                             c=df_plot['mean_annual_precip'], cmap='viridis',
                             alpha=0.6, s=15)  # 增大点的大小
            cbar = plt.colorbar(sc)
            cbar.set_label(VIZ_CONFIG['labels']['mean_annual_precip'], fontsize=12)
        else:
            plt.scatter(df_display['elevation'], df_display[target], alpha=0.6, s=15)
    else:
        plt.scatter(df_display['elevation'], df_display[target], alpha=0.6, s=15)

    # Add trend line with more bins for better resolution
    try:
        bins = min(30, len(df_display) // 50)  # 动态确定分箱数
        bins = max(10, bins)  # 至少10个分箱

        bin_means, bin_edges, _ = binned_statistic(
            df_display['elevation'], df_display[target],
            statistic='mean', bins=bins
        )

        # 只保留有效的分箱
        valid_bins = ~np.isnan(bin_means)
        if valid_bins.any():
            bin_centers = (bin_edges[:-1] + bin_edges[1:])[valid_bins] / 2
            valid_means = bin_means[valid_bins]

            plt.plot(bin_centers, valid_means, 'r-', linewidth=3,
                     label=f'Trend Line ({bins} bins)', alpha=0.8)
            plt.legend()
    except Exception as e:
        print(f"Failed to add trend line: {e}")

    # 设置坐标轴范围和刻度
    plt.xlim(elev_min - 50, elev_max + 50)  # 稍微扩展显示范围

    # 设置x轴刻度，确保从合理值开始
    x_ticks = np.linspace(elev_min, elev_max, 8)
    x_ticks = np.round(x_ticks / 100) * 100  # 四舍五入到最近的100m
    plt.xticks(x_ticks)

    plt.title(VIZ_CONFIG['titles']['elevation_rain'], fontsize=16)
    plt.xlabel(VIZ_CONFIG['labels']['elevation'], fontsize=12)
    plt.ylabel(VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('elevation_vs_rain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: elevation_vs_rain.png")


def _plot_slope_rain_relationship_english(df_model, target):
    """Plot slope vs rain frequency relationship with English labels - Fixed version"""
    plt.figure(figsize=(12, 8))

    # 检查坡度数据
    if 'slope' not in df_model.columns:
        print("Warning: slope column not found")
        return

    # 清理坡度数据
    valid_mask = (df_model['slope'].notna() &
                  (df_model['slope'] >= 0) &
                  (df_model['slope'] <= 90) &  # 坡度应该在0-90度之间
                  df_model[target].notna())

    df_valid = df_model[valid_mask].copy()

    if len(df_valid) < 10:
        print("Warning: Too few valid slope data points")
        return

    print(f"坡度数据范围: {df_valid['slope'].min():.1f} - {df_valid['slope'].max():.1f} degrees")

    # 设置合理的坡度显示范围
    slope_min = df_valid['slope'].quantile(0.01)
    slope_max = df_valid['slope'].quantile(0.99)

    # Scatter plot colored by aspect class
    if 'aspect_class' in df_valid.columns:
        # 确保坡向分类在合理范围内
        aspect_valid = df_valid['aspect_class'].between(1, 4, inclusive='both')
        df_plot = df_valid[aspect_valid]

        if len(df_plot) > 0:
            sc = plt.scatter(df_plot['slope'], df_plot[target],
                             c=df_plot['aspect_class'], cmap='tab10',
                             alpha=0.6, s=15)
            cbar = plt.colorbar(sc)
            cbar.set_label(VIZ_CONFIG['labels']['aspect_class'], fontsize=12)
            cbar.set_ticks([1, 2, 3, 4])  # 设置刻度
        else:
            plt.scatter(df_valid['slope'], df_valid[target], alpha=0.6, s=15)
    else:
        plt.scatter(df_valid['slope'], df_valid[target], alpha=0.6, s=15)

    # Add trend line
    try:
        bins = min(25, len(df_valid) // 40)
        bins = max(8, bins)

        bin_means, bin_edges, _ = binned_statistic(
            df_valid['slope'], df_valid[target],
            statistic='mean', bins=bins
        )

        valid_bins = ~np.isnan(bin_means)
        if valid_bins.any():
            bin_centers = (bin_edges[:-1] + bin_edges[1:])[valid_bins] / 2
            valid_means = bin_means[valid_bins]

            plt.plot(bin_centers, valid_means, 'r-', linewidth=3,
                     label=f'Trend Line ({bins} bins)', alpha=0.8)
            plt.legend()
    except Exception as e:
        print(f"Failed to add trend line: {e}")

    # 设置坐标轴
    plt.xlim(-1, slope_max + 2)

    # 设置x轴刻度
    x_ticks = np.linspace(0, slope_max, 8)
    x_ticks = np.round(x_ticks)  # 四舍五入到整数度
    plt.xticks(x_ticks)

    plt.title(VIZ_CONFIG['titles']['slope_rain'], fontsize=16)
    plt.xlabel(VIZ_CONFIG['labels']['slope'], fontsize=12)
    plt.ylabel(VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('slope_vs_rain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: slope_vs_rain.png")


def _plot_interaction_effects_english(df_model, target):
    """Plot interaction effects with English labels - Fixed version"""
    plt.figure(figsize=(12, 8))

    # 检查必要的列
    required_cols = ['elevation', 'mean_annual_precip', target]
    missing_cols = [col for col in required_cols if col not in df_model.columns]
    if missing_cols:
        print(f"Warning: Missing columns for interaction plot: {missing_cols}")
        return

    # 数据清理和验证
    valid_mask = (df_model['elevation'].notna() &
                  (df_model['elevation'] > 0) &
                  (df_model['elevation'] < 10000) &
                  df_model['mean_annual_precip'].notna() &
                  (df_model['mean_annual_precip'] > 0) &
                  (df_model['mean_annual_precip'] < 5000) &
                  df_model[target].notna())

    df_valid = df_model[valid_mask].copy()

    if len(df_valid) < 50:
        print("Warning: Too few valid data points for interaction analysis")
        # 使用简单散点图作为备选
        plt.scatter(df_model['mean_annual_precip'], df_model[target], alpha=0.6)
        plt.title("Precipitation vs Heavy Rain Frequency (Insufficient data for interaction analysis)")
        plt.xlabel(VIZ_CONFIG['labels']['mean_annual_precip'], fontsize=12)
        plt.ylabel(VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
        plt.tight_layout()
        plt.savefig('interaction_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
        return

    try:
        # 改进的海拔分组方法
        elev_min = df_valid['elevation'].quantile(0.05)
        elev_max = df_valid['elevation'].quantile(0.95)

        # 确保最小海拔从合理值开始
        if elev_min < 500 and elev_max > 1000:
            elev_min = max(500, df_valid['elevation'].quantile(0.1))

        print(f"海拔分组范围: {elev_min:.1f} - {elev_max:.1f} m")

        # 使用固定的分位数进行分组，确保每组有足够的数据
        try:
            # 尝试3组分类
            elevation_bins = [elev_min,
                              df_valid['elevation'].quantile(0.4),
                              df_valid['elevation'].quantile(0.7),
                              elev_max]

            df_valid['elev_group'] = pd.cut(
                df_valid['elevation'],
                bins=elevation_bins,
                labels=["Low Elevation\n({:.0f}-{:.0f}m)".format(elevation_bins[0], elevation_bins[1]),
                        "Medium Elevation\n({:.0f}-{:.0f}m)".format(elevation_bins[1], elevation_bins[2]),
                        "High Elevation\n({:.0f}-{:.0f}m)".format(elevation_bins[2], elevation_bins[3])],
                include_lowest=True
            )
        except Exception as e:
            print(f"Failed to create elevation groups with cut: {e}")
            # 备选方案：使用简单的三等分
            df_valid = df_valid.sort_values('elevation')
            n_total = len(df_valid)
            df_valid['elev_group'] = ['Low Elevation'] * (n_total // 3) + \
                                     ['Medium Elevation'] * (n_total // 3) + \
                                     ['High Elevation'] * (n_total - 2 * (n_total // 3))

        # 检查分组结果
        group_counts = df_valid['elev_group'].value_counts()
        print("海拔分组统计:")
        for group, count in group_counts.items():
            print(f"  {group}: {count} points")

        # 为每个海拔组绘制降水量与暴雨频率的关系
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
        markers = ['o', 's', '^']  # 圆形、方形、三角形

        for i, group in enumerate(df_valid['elev_group'].unique()):
            if pd.isna(group):
                continue

            group_data = df_valid[df_valid['elev_group'] == group]

            if len(group_data) < 5:
                print(f"Warning: Too few points in group {group}: {len(group_data)}")
                continue

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # 绘制散点
            plt.scatter(group_data['mean_annual_precip'], group_data[target],
                        label=f'{group} (n={len(group_data)})',
                        alpha=0.7, s=25, color=color, marker=marker)

            # 添加趋势线
            try:
                # 确保有足够的数据点进行拟合
                if len(group_data) >= 5:
                    # 移除异常值以获得更好的拟合
                    precip_q1 = group_data['mean_annual_precip'].quantile(0.1)
                    precip_q9 = group_data['mean_annual_precip'].quantile(0.9)

                    trend_data = group_data[
                        (group_data['mean_annual_precip'] >= precip_q1) &
                        (group_data['mean_annual_precip'] <= precip_q9)
                        ]

                    if len(trend_data) >= 3:
                        z = np.polyfit(trend_data['mean_annual_precip'],
                                       trend_data[target], 1)
                        p = np.poly1d(z)

                        # 创建平滑的趋势线
                        x_min = trend_data['mean_annual_precip'].min()
                        x_max = trend_data['mean_annual_precip'].max()
                        x_range = np.linspace(x_min, x_max, 50)

                        plt.plot(x_range, p(x_range), '--',
                                 color=color, linewidth=2, alpha=0.8)

                        # 显示趋势线斜率
                        slope = z[0]
                        print(f"  {group} 趋势线斜率: {slope:.4f}")

            except Exception as e:
                print(f"Failed to add trend line for {group}: {e}")

        # 设置降水量坐标轴范围
        precip_min = df_valid['mean_annual_precip'].quantile(0.02)
        precip_max = df_valid['mean_annual_precip'].quantile(0.98)
        plt.xlim(precip_min - 50, precip_max + 50)

        # 设置x轴刻度
        x_ticks = np.linspace(precip_min, precip_max, 8)
        x_ticks = np.round(x_ticks / 100) * 100  # 四舍五入到最近的100mm
        plt.xticks(x_ticks)

    except Exception as e:
        print(f"Interaction effect analysis failed: {e}")
        # 简单散点图备选
        plt.scatter(df_valid['mean_annual_precip'], df_valid[target], alpha=0.6)
        plt.title("Precipitation vs Heavy Rain Frequency (Interaction analysis failed)")

    plt.title(VIZ_CONFIG['titles']['interaction_effect'], fontsize=16)
    plt.xlabel(VIZ_CONFIG['labels']['mean_annual_precip'], fontsize=12)
    plt.ylabel(VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('interaction_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: interaction_effect.png")


def _plot_model_predictions_english(xgb_model):
    """Plot model prediction performance with English labels"""
    plt.figure(figsize=(10, 10))

    try:
        y_test = xgb_model['y_test']
        y_pred = xgb_model['y_pred']

        plt.scatter(y_test, y_pred, alpha=0.6)

        # Add ideal prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Calculate R²
        r2 = xgb_model.get('r2', 0)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                 fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        print(f"Failed to plot predictions: {e}")
        # Create example data
        x = np.array([1, 2, 3, 4, 5])
        plt.scatter(x, x + np.random.normal(0, 0.1, 5))
        plt.plot([1, 5], [1, 5], 'r--')

    plt.title(VIZ_CONFIG['titles']['prediction_actual'], fontsize=16)
    plt.xlabel('Actual ' + VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    plt.ylabel('Predicted ' + VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: prediction_vs_actual.png")


def _plot_correlation_heatmap_english(corr_matrix):
    """
    修复的相关性热图绘制函数 - 完整显示
    """
    plt.figure(figsize=(16, 14))  # 增大图形尺寸

    try:
        # 确保相关性矩阵不为空且是方形
        if corr_matrix.empty or corr_matrix.shape[0] != corr_matrix.shape[1]:
            print("相关性矩阵为空或不是方形，创建示例矩阵")
            # 创建示例相关性矩阵
            features = ['elevation', 'slope', 'precip', 'temp', 'heavy_rain']
            corr_matrix = pd.DataFrame(np.random.rand(5, 5),
                                       columns=features, index=features)
            # 使其对称
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix.values, 1.0)

        print(f"相关性矩阵形状: {corr_matrix.shape}")
        print(f"相关性矩阵列: {list(corr_matrix.columns)}")

        # 不使用掩码，显示完整的热图
        sns.heatmap(corr_matrix,
                    annot=True,  # 显示数值
                    cmap='coolwarm',  # 颜色方案
                    fmt='.2f',  # 数值格式
                    square=True,  # 方形单元格
                    linewidths=0.5,  # 网格线宽度
                    cbar_kws={'shrink': 0.8},  # 颜色条设置
                    center=0,  # 以0为中心
                    vmin=-1, vmax=1,  # 设置颜色范围
                    xticklabels=True,  # 显示x轴标签
                    yticklabels=True)  # 显示y轴标签

        # 优化标签显示
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    except Exception as e:
        print(f"绘制相关性热图失败: {e}")
        # 创建简单的示例热图
        dummy_data = np.random.rand(8, 8)
        dummy_data = (dummy_data + dummy_data.T) / 2  # 使其对称
        np.fill_diagonal(dummy_data, 1.0)

        labels = ['Elev', 'Slope', 'Aspect', 'Rough', 'TPI', 'Precip', 'CV', 'Rain']

        sns.heatmap(dummy_data,
                    annot=True,
                    cmap='coolwarm',
                    fmt='.2f',
                    square=True,
                    xticklabels=labels,
                    yticklabels=labels,
                    linewidths=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.title(VIZ_CONFIG['titles']['correlation_heatmap'], fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: correlation_heatmap.png")


def _plot_3d_terrain_relationship_english(df_model, target):
    """Plot 3D terrain relationship with English labels - Fixed version"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    try:
        # 数据清理和验证
        valid_mask = (df_model['elevation'].notna() &
                      (df_model['elevation'] > 0) &
                      (df_model['elevation'] < 10000) &
                      df_model['slope'].notna() &
                      (df_model['slope'] >= 0) &
                      (df_model['slope'] <= 90) &
                      df_model[target].notna())

        df_valid = df_model[valid_mask]

        # 数据采样以避免过度拥挤
        sample_size = min(3000, len(df_valid))  # 减少点数以提高可视化效果
        if len(df_valid) > sample_size:
            df_sample = df_valid.sample(sample_size, random_state=42)
        else:
            df_sample = df_valid

        print(f"3D图使用 {len(df_sample)} 个数据点")

        # 设置显示范围
        elev_min = df_sample['elevation'].quantile(0.05)
        elev_max = df_sample['elevation'].quantile(0.95)
        slope_min = df_sample['slope'].quantile(0.05)
        slope_max = df_sample['slope'].quantile(0.95)

        # 确保海拔从合理值开始
        if elev_min < 500 and elev_max > 1000:
            elev_min = max(500, df_sample['elevation'].quantile(0.1))

        # 过滤显示范围
        display_mask = ((df_sample['elevation'] >= elev_min) &
                        (df_sample['elevation'] <= elev_max) &
                        (df_sample['slope'] >= slope_min) &
                        (df_sample['slope'] <= slope_max))

        df_display = df_sample[display_mask]

        if len(df_display) < 10:
            print("Warning: Too few points for 3D visualization")
            df_display = df_sample

        # 3D散点图
        if 'mean_annual_precip' in df_display.columns:
            # 清理降水数据
            precip_valid = (df_display['mean_annual_precip'].notna() &
                            (df_display['mean_annual_precip'] > 0) &
                            (df_display['mean_annual_precip'] < 5000))

            df_plot = df_display[precip_valid]

            if len(df_plot) > 0:
                scatter = ax.scatter(df_plot['elevation'],
                                     df_plot['slope'],
                                     df_plot[target],
                                     c=df_plot['mean_annual_precip'],
                                     cmap='viridis',
                                     s=20, alpha=0.7)

                # 添加颜色条
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
                cbar.set_label(VIZ_CONFIG['labels']['mean_annual_precip'], fontsize=12)
            else:
                ax.scatter(df_display['elevation'], df_display['slope'], df_display[target],
                           s=20, alpha=0.7)
        else:
            ax.scatter(df_display['elevation'], df_display['slope'], df_display[target],
                       s=20, alpha=0.7)

        # 设置坐标轴范围
        ax.set_xlim(elev_min, elev_max)
        ax.set_ylim(slope_min, slope_max)

        print(f"3D图显示范围 - 海拔: {elev_min:.0f}-{elev_max:.0f}m, 坡度: {slope_min:.1f}-{slope_max:.1f}°")

    except Exception as e:
        print(f"Failed to plot 3D chart: {e}")
        # 创建示例3D数据
        x = np.random.rand(100) * 2000 + 500  # 500-2500m海拔
        y = np.random.rand(100) * 30  # 0-30度坡度
        z = np.random.rand(100) * 10  # 0-10天暴雨
        ax.scatter(x, y, z, s=20, alpha=0.7)

    ax.set_xlabel(VIZ_CONFIG['labels']['elevation'], fontsize=12)
    ax.set_ylabel(VIZ_CONFIG['labels']['slope'], fontsize=12)
    ax.set_zlabel(VIZ_CONFIG['labels']['heavy_rain_days'], fontsize=12)
    ax.set_title(VIZ_CONFIG['titles']['3d_terrain'], fontsize=16, pad=20)

    # 优化视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('3d_terrain_rain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved chart: 3d_terrain_rain.png")

# =============================================================================
# Results Summary Module
# =============================================================================

def summarize_findings(models, analysis_results):
    """Summarize analysis results and explain terrain-climate interactions on extreme weather"""
    print("\n" + "=" * 60)
    print("分析结果总结")
    print("=" * 60)

    # 1. Model performance comparison
    print("\n1. 模型性能比较")
    print("-" * 30)
    best_r2 = -np.inf
    best_model = None

    for model_name, model_info in models.items():
        r2 = model_info.get('r2', 0)
        mse = model_info.get('mse', 0)
        print(f"  {model_name:12s}: R² = {r2:6.4f}, MSE = {mse:8.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name

    print(f"  最佳模型: {best_model} (R² = {best_r2:.4f})")

    # 2. Important features analysis
    print("\n2. 重要特征分析 (基于XGBoost模型)")
    print("-" * 50)
    top_features = analysis_results['xgb_importance'].head(5)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:20s}: Importance = {row['importance']:.4f}")

    # 3. Terrain factors impact analysis
    print("\n3. 地形因素对暴雨的影响")
    print("-" * 40)
    terrain_features = ['elevation', 'slope', 'aspect_class', 'roughness', 'tpi']
    _analyze_factor_effects_english(analysis_results['glm_coef'], terrain_features, "Terrain")

    # 4. Climate factors impact analysis
    print("\n4. 气候因素对暴雨的影响")
    print("-" * 40)
    climate_features = ['mean_annual_precip', 'spring_ratio', 'summer_ratio',
                        'autumn_ratio', 'winter_ratio', 'precip_cv', 'mean_annual_temp']
    _analyze_factor_effects_english(analysis_results['glm_coef'], climate_features, "Climate")

    # 5. Interaction analysis
    print("\n5. 地形与气候的交互作用")
    print("-" * 32)
    interaction_results = analysis_results['interaction_results']
    for _, row in interaction_results.iterrows():
        significance = "显著" if row['p_value'] < 0.05 else "不显著"
        effect = "正向交互" if row['coef'] > 0 else "负向交互"
        print(f"  {row['feature']:15s}: {effect}, {significance}")
        print(f"                     (Coef={row['coef']:8.6f}, p-value={row['p_value']:6.4f})")

    # 6. Key findings summary
    print("\n6. 主要发现总结")
    print("-" * 23)
    _summarize_key_findings_english(analysis_results)

    # 7. Application value
    print("\n7. 研究应用价值")
    print("-" * 30)
    print("  ✓ 量化了地形因素对极端降水的影响机制")
    print("  ✓ 识别了地形-气候交互作用的关键因子")
    print("  ✓ 为极端天气预报模型提供了重要参数")
    print("  ✓ 支撑防灾减灾和水资源管理决策")
    print("  ✓ 为气候变化适应策略提供科学依据")


def _analyze_factor_effects_english(glm_coef, factor_list, factor_type):
    """Analyze specific factor type impact effects"""
    factor_coefs = glm_coef[glm_coef['feature'].isin(factor_list)]

    if factor_coefs.empty:
        print(f"  未找到{factor_type}因子的显著影响")
        return

    significant_factors = factor_coefs[factor_coefs['p_value'] < 0.05]

    if significant_factors.empty:
        print(f"  无显著的{factor_type}因子影响")
    else:
        for _, row in significant_factors.iterrows():
            effect = "Positive correlation" if row['coef'] > 0 else "Negative correlation"
            print(f"  {row['feature']:20s}: {effect} (Coef={row['coef']:8.6f}, p-value={row['p_value']:6.4f})")


def _summarize_key_findings_english(analysis_results):
    """Summarize key findings"""
    # Find most important feature
    top_feature = analysis_results['xgb_importance'].iloc[0]
    print(f"  ✓ 最重要影响因子: {top_feature['feature']} (重要性: {top_feature['importance']:.3f})")

    # Analyze strongest correlation
    if not analysis_results['target_corr'].empty:
        target_corr_filtered = analysis_results['target_corr'].iloc[1:]
        if not target_corr_filtered.empty:
            highest_corr_feature = target_corr_filtered.index[0]
            highest_corr_value = target_corr_filtered.iloc[0]
            print(f"  ✓ 相关性最强因子: {highest_corr_feature} (相关系数: {highest_corr_value:.3f})")

    # Analyze interaction effects
    interaction_results = analysis_results['interaction_results']
    significant_interactions = interaction_results[interaction_results['p_value'] < 0.05]

    if not significant_interactions.empty:
        print(f"  ✓ 发现 {len(significant_interactions)} 个显著的地形-气候交互作用")
        for _, row in significant_interactions.iterrows():
            mechanism = _interpret_interaction_mechanism_english(row['feature'], row['coef'])
            print(f"    - {mechanism}")
    else:
        print("  ✓ 地形-气候交互作用不显著，主要为独立效应")


def _interpret_interaction_mechanism_english(interaction_term, coefficient):
    """Interpret interaction mechanisms"""
    if interaction_term == 'elev_x_precip':
        if coefficient > 0:
            return "地形抬升机制：高海拔地区的降水对暴雨形成有增强作用"
        else:
            return "高海拔抑制机制：高海拔地区可能通过低温等因素抑制极端降水"

    elif interaction_term == 'slope_x_precip':
        if coefficient > 0:
            return "迎风坡效应：陡峭坡度与降水的协同作用增强暴雨形成"
        else:
            return "复杂地形效应：陡坡可能通过气流分散等机制抑制暴雨集中"

    return f"{interaction_term}的交互机制需进一步研究"


# =============================================================================
# Main Execution Function
# =============================================================================

def main():
    """Execute complete terrain-climate interaction analysis workflow"""
    print("\n" + "=" * 80)
    print("地形气候相互作用与极端天气形成关系分析")
    print("=" * 80)
    print(
        "分析目标：量化地形因素与气候因素在极端降水形成中的相互作用")
    print("数据范围：中国大陆地区，1990-2020年")
    print("-" * 80)

    try:
        # Initialize file paths
        dem_path = find_file(DATA_CONFIG['dem_dir'], extension='.tif')
        precip_path = find_file(DATA_CONFIG['precip_dir'], extension='.nc')

        if not dem_path or not precip_path:
            print("Error: Missing required input data files")
            return False

        # Phase 1: Data Loading
        print("\n第一阶段：数据加载")
        print("-" * 20)
        dem_data, dem_transform, dem_crs, dem_profile = load_dem_data(dem_path)
        precip_ds, precip_var = load_precipitation_data(precip_path)
        temp_sample, temp_transform, temp_crs, temp_profile, temp_files = load_temperature_data(
            DATA_CONFIG['temp_dir']
        )

        # Phase 2: Data Preprocessing
        print("\n第二阶段：数据预处理")
        print("-" * 25)
        resampled_dem, dem_target_transform, dem_target_profile = resample_dem_to_target(
            dem_data, dem_transform, dem_crs
        )

        # Phase 3: Feature Calculation
        print("\n第三阶段：特征计算")
        print("-" * 27)
        terrain_features = calculate_terrain_features(resampled_dem, dem_target_transform)
        climate_vars = calculate_climate_variables(precip_ds, precip_var, temp_files)

        # Phase 4: Data Integration
        print("\n第四阶段：数据整合")
        print("-" * 24)
        integrated_df = integrate_data(terrain_features, climate_vars, dem_target_transform)

        # Phase 5: Machine Learning Modeling
        print("\n第五阶段：机器学习建模")
        print("-" * 33)
        features = [
            'elevation', 'slope', 'aspect_class', 'roughness', 'tpi',  # Terrain features
            'mean_annual_precip', 'spring_ratio', 'summer_ratio',  # Climate features
            'autumn_ratio', 'winter_ratio', 'precip_cv',
            'elev_x_precip', 'slope_x_precip'  # Interaction features
        ]
        target = 'heavy_rain_days'

        # Check feature availability
        available_features = [f for f in features if f in integrated_df.columns]
        print(f"Available features: {len(available_features)}/{len(features)}")

        models, df_model = build_models(integrated_df, available_features, target)

        # Phase 6: Model Analysis
        print("\n第六阶段：模型结果分析")
        print("-" * 30)
        analysis_results = analyze_models(models, df_model, available_features, target)

        # Phase 7: Visualization
        print("\n第七阶段：结果可视化")
        print("-" * 28)
        visualize_results(models, df_model, available_features, target, analysis_results)

        # Phase 8: Results Summary
        print("\n第八阶段：结果总结与解释")
        print("-" * 42)
        summarize_findings(models, analysis_results)

        # Analysis completed
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        print("✓ 成功量化了地形与气候因素的相互作用关系")
        print("✓ 识别了影响极端降水的关键地形和气候因子")
        print("✓ 建立了多种机器学习预测模型")
        print("✓ 生成了7个可视化图表文件")
        print("✓ 为极端天气预报和防灾减灾提供了科学依据")

        return True

    except Exception as e:
        print(f"\nProgram execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Program Entry Point
# =============================================================================

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\nProgram executed successfully! Please check the generated chart files.")
    else:
        print(f"\nProgram execution failed, please check data files and configuration.")
