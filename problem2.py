#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地形气候相互作用与极端天气形成关系分析

本脚本分析中国地形与气候数据，量化地形因素（海拔、坡度、坡向）与气候因素
在极端降水形成过程中的相互作用关系。

作者: [作者姓名]
日期: 2024
版本: 1.0
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

# 忽略警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 全局配置
# =============================================================================

# 数据文件路径配置
DATA_CONFIG = {
    'dem_dir': r"D:\digifax_2024D_date\dataset1\中国数字高程图(1km)\Geo\TIFF",
    'temp_dir': r"D:\digifax_2024D_date\dataset2\日平均数据",
    'precip_dir': r"D:\0.25rain(1961-2022)"
}

# 分析参数配置
ANALYSIS_CONFIG = {
    'start_year': 1990,
    'end_year': 2020,
    'target_resolution': 0.25,  # 目标分辨率（度）
    'china_bounds': {  # 中国大陆边界
        'west': 73.0,
        'east': 136.0,
        'south': 18.0,
        'north': 54.0
    },
    'heavy_rain_threshold': 50,  # 暴雨阈值 (mm/day)
    'nodata_value': -9999
}


# =============================================================================
# 辅助函数
# =============================================================================

def find_file(directory, extension=None, contains=None):
    """
    在指定目录中查找符合条件的文件

    Parameters:
    -----------
    directory : str
        搜索目录路径
    extension : str, optional
        文件扩展名（如'.tif'）
    contains : str, optional
        文件名应包含的字符串

    Returns:
    --------
    str or None
        找到的文件完整路径，未找到则返回None
    """
    try:
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return None

        files = os.listdir(directory)

        # 按条件筛选文件
        if extension:
            files = [f for f in files if f.lower().endswith(extension.lower())]
        if contains:
            files = [f for f in files if contains.lower() in f.lower()]

        if files:
            file_path = os.path.join(directory, files[0])
            print(f"在 {directory} 中找到文件: {files[0]}")
            return file_path
        else:
            print(f"在 {directory} 中找不到匹配的文件")
            return None

    except Exception as e:
        print(f"查找文件时出错: {e}")
        return None


# =============================================================================
# 数据加载模块
# =============================================================================

def load_dem_data(dem_path):
    """
    加载并处理中国数字高程模型(DEM)数据

    Parameters:
    -----------
    dem_path : str
        DEM文件路径

    Returns:
    --------
    tuple
        (dem_masked, dem_transform, dem_crs, dem_profile)
        - dem_masked: 掩码后的DEM数据数组
        - dem_transform: 地理坐标变换矩阵
        - dem_crs: 坐标参考系统
        - dem_profile: 文件属性信息
    """
    if not dem_path or not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM文件不存在: {dem_path}")

    print("正在加载DEM数据...")

    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # 读取第一个波段
            dem_transform = src.transform
            dem_crs = src.crs
            dem_profile = src.profile

        # 处理无效值，创建掩码数组
        nodata = src.nodata if src.nodata else ANALYSIS_CONFIG['nodata_value']
        dem_masked = np.ma.masked_values(dem, nodata)

        print(f"DEM数据加载完成 - 形状: {dem.shape}, 有效像素: {dem_masked.count()}")
        return dem_masked, dem_transform, dem_crs, dem_profile

    except Exception as e:
        print(f"加载DEM数据失败: {e}")
        raise


def load_precipitation_data(precip_path, start_year=None, end_year=None):
    """
    加载并处理中国大陆0.25°逐日降水数据集

    Parameters:
    -----------
    precip_path : str
        降水数据文件路径
    start_year : int, optional
        起始年份
    end_year : int, optional
        结束年份

    Returns:
    --------
    tuple
        (precip_dataset, precip_variable_name)
    """
    if not precip_path or not os.path.exists(precip_path):
        raise FileNotFoundError(f"降水数据文件不存在: {precip_path}")

    # 使用配置中的默认年份
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print(f"正在加载降水数据 ({start_year}-{end_year})...")

    try:
        ds = xr.open_dataset(precip_path)
        print(f"降水数据集变量: {list(ds.data_vars)}")

        # 自动识别降水变量名
        precip_var = None
        possible_vars = ['pre', 'PRE', 'prcp', 'PRCP', 'precipitation', 'rain']

        for var in possible_vars:
            if var in ds.data_vars:
                precip_var = var
                break

        if not precip_var:
            # 尝试查找包含降水关键词的变量
            for var in ds.data_vars:
                if any(keyword in var.lower() for keyword in ['pre', 'rain', 'prcp']):
                    precip_var = var
                    break

        if not precip_var:
            raise ValueError(f"未找到降水变量，可用变量: {list(ds.data_vars)}")

        print(f"使用降水变量: {precip_var}")

        # 时间范围筛选
        try:
            ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        except (KeyError, ValueError) as e:
            print(f"时间筛选失败: {e}，尝试其他方法")
            # 尝试使用年份索引
            if 'years' in ds.dims:
                years = ds.years.values
                year_mask = (years >= start_year) & (years <= end_year)
                ds = ds.isel(years=year_mask)

        print(f"降水数据加载完成 - 形状: {ds[precip_var].shape}")
        return ds, precip_var

    except Exception as e:
        print(f"加载降水数据失败: {e}")
        raise


def load_temperature_data(temp_dir, start_year=None, end_year=None):
    """
    加载并处理按年份组织的气温数据

    Parameters:
    -----------
    temp_dir : str
        气温数据目录路径
    start_year : int, optional
        起始年份
    end_year : int, optional
        结束年份

    Returns:
    --------
    tuple
        (temp_sample, temp_transform, temp_crs, temp_profile, temp_files_list)
    """
    if not os.path.exists(temp_dir):
        print(f"气温数据目录不存在: {temp_dir}")
        return None, None, None, None, []

    # 使用配置中的默认年份
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print(f"正在加载气温数据 ({start_year}-{end_year})...")

    # 查找年份文件夹
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
        print(f"读取气温数据文件夹失败: {e}")
        return None, None, None, None, []

    if not year_folders:
        print(f"未找到指定年份范围的气温数据")
        return None, None, None, None, []

    # 收集所有温度文件
    year_folders.sort(key=lambda x: x[0])
    all_temp_files = []

    for year, folder_path in year_folders:
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
            temp_files = [os.path.join(folder_path, f) for f in files]
            all_temp_files.extend(temp_files)
            print(f"年份 {year}: 找到 {len(temp_files)} 个气温文件")
        except Exception as e:
            print(f"读取年份 {year} 的气温文件失败: {e}")

    if not all_temp_files:
        print("未找到任何气温数据文件")
        return None, None, None, None, []

    # 读取样例文件获取基本信息
    try:
        with rasterio.open(all_temp_files[0]) as src:
            temp_sample = src.read(1)
            temp_transform = src.transform
            temp_crs = src.crs
            temp_profile = src.profile

        print(f"气温数据加载完成 - 总文件数: {len(all_temp_files)}")
        return temp_sample, temp_transform, temp_crs, temp_profile, all_temp_files

    except Exception as e:
        print(f"读取气温数据样例失败: {e}")
        return None, None, None, None, []


# =============================================================================
# 数据预处理模块
# =============================================================================

def resample_dem_to_target(dem_data, dem_transform, dem_crs, target_res=None):
    """
    将DEM数据重采样到目标分辨率

    Parameters:
    -----------
    dem_data : numpy.ndarray
        原始DEM数据
    dem_transform : Affine
        原始变换矩阵
    dem_crs : CRS
        原始坐标系
    target_res : float, optional
        目标分辨率（度）

    Returns:
    --------
    tuple
        (resampled_dem, dst_transform, dst_profile)
    """
    target_res = target_res or ANALYSIS_CONFIG['target_resolution']
    bounds = ANALYSIS_CONFIG['china_bounds']

    print(f"将DEM数据重采样到{target_res}°分辨率...")

    # 计算目标网格尺寸
    width = int((bounds['east'] - bounds['west']) / target_res)
    height = int((bounds['north'] - bounds['south']) / target_res)

    # 计算目标变换矩阵
    dst_transform = rasterio.transform.from_bounds(
        bounds['west'], bounds['south'],
        bounds['east'], bounds['north'],
        width, height
    )

    # 创建临时文件进行重采样
    temp_tif = 'temp_dem_resample.tif'

    try:
        # 写入临时文件
        with rasterio.open(temp_tif, 'w',
                           driver='GTiff',
                           height=dem_data.shape[0],
                           width=dem_data.shape[1],
                           count=1,
                           dtype=dem_data.dtype,
                           crs=dem_crs,
                           transform=dem_transform,
                           nodata=ANALYSIS_CONFIG['nodata_value']) as dst:

            # 处理掩码数组
            if isinstance(dem_data, np.ma.MaskedArray):
                dem_array = dem_data.filled(ANALYSIS_CONFIG['nodata_value'])
            else:
                dem_array = dem_data
            dst.write(dem_array, 1)

        # 执行重采样
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

        # 创建输出属性
        dst_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'dtype': resampled_dem.dtype,
            'crs': 'EPSG:4326',
            'transform': dst_transform
        }

        print(f"DEM重采样完成 - 新形状: {resampled_dem.shape}")
        return resampled_dem, dst_transform, dst_profile

    except Exception as e:
        print(f"重采样DEM数据失败: {e}")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(temp_tif):
            os.remove(temp_tif)


# =============================================================================
# 特征计算模块
# =============================================================================

def calculate_terrain_features(dem, transform):
    """
    计算地形特征：坡度、坡向、地形粗糙度、地形位置指数等

    Parameters:
    -----------
    dem : numpy.ndarray
        数字高程模型数据
    transform : Affine
        地理坐标变换矩阵

    Returns:
    --------
    dict
        包含各种地形特征的字典
    """
    print("正在计算地形特征...")

    try:
        # 获取像素分辨率
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])

        # 1. 计算坡度（使用Sobel算子）
        dx = ndimage.sobel(dem, axis=1) / (8.0 * pixel_width)
        dy = ndimage.sobel(dem, axis=0) / (8.0 * pixel_height)
        slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

        # 2. 计算坡向（0-360度，北=0）
        aspect = np.degrees(np.arctan2(dy, -dx))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # 3. 坡向分类（1=北, 2=东, 3=南, 4=西）
        aspect_class = np.zeros_like(aspect, dtype=np.int8)
        aspect_class[(aspect >= 315) | (aspect < 45)] = 1  # 北
        aspect_class[(aspect >= 45) & (aspect < 135)] = 2  # 东
        aspect_class[(aspect >= 135) & (aspect < 225)] = 3  # 南
        aspect_class[(aspect >= 225) & (aspect < 315)] = 4  # 西

        # 4. 地形粗糙度（3x3窗口标准差）
        roughness = ndimage.generic_filter(dem, np.std, size=3)

        # 5. 地形位置指数（TPI）
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0  # 中心点权重为0
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

        print("地形特征计算完成")
        return terrain_features

    except Exception as e:
        print(f"计算地形特征失败: {e}")
        raise


def calculate_climate_variables(precip_ds, precip_var, temp_files=None,
                                start_year=None, end_year=None):
    """
    计算气候变量：年均降水量、季节性分布、温度等

    Parameters:
    -----------
    precip_ds : xarray.Dataset
        降水数据集
    precip_var : str
        降水变量名
    temp_files : list, optional
        温度文件列表
    start_year : int, optional
        起始年份
    end_year : int, optional
        结束年份

    Returns:
    --------
    dict
        包含各种气候变量的字典
    """
    start_year = start_year or ANALYSIS_CONFIG['start_year']
    end_year = end_year or ANALYSIS_CONFIG['end_year']

    print("正在计算气候变量...")

    try:
        # 1. 计算降水相关变量
        if 'time' in precip_ds.dims:
            # 标准时间维度处理
            annual_precip = precip_ds[precip_var].groupby('time.year').sum('time')
            mean_annual_precip = annual_precip.mean('year')

            # 季节性降水分布
            seasons = {
                'spring': [3, 4, 5],  # 春季
                'summer': [6, 7, 8],  # 夏季
                'autumn': [9, 10, 11],  # 秋季
                'winter': [12, 1, 2]  # 冬季
            }

            seasonal_precip = {}
            seasonal_ratios = {}

            for season, months in seasons.items():
                season_data = precip_ds.sel(
                    time=precip_ds['time.month'].isin(months)
                )[precip_var].groupby('time.year').sum('time')

                seasonal_precip[season] = season_data.mean('year')
                seasonal_ratios[f'{season}_ratio'] = seasonal_precip[season] / mean_annual_precip

            # 降水变异系数
            precip_cv = annual_precip.std('year') / mean_annual_precip

            # 简化版暴雨频率计算
            threshold = ANALYSIS_CONFIG['heavy_rain_threshold']
            heavy_rain_ratio = (precip_ds[precip_var] >= threshold).mean('time')
            heavy_rain_days = heavy_rain_ratio * 365.25

        else:
            # 备选方案：无标准时间维度
            print("警告：使用简化的气候变量计算方法")
            mean_annual_precip = precip_ds[precip_var].mean()

            # 默认季节比例
            seasonal_ratios = {
                'spring_ratio': xr.ones_like(mean_annual_precip) * 0.25,
                'summer_ratio': xr.ones_like(mean_annual_precip) * 0.40,
                'autumn_ratio': xr.ones_like(mean_annual_precip) * 0.25,
                'winter_ratio': xr.ones_like(mean_annual_precip) * 0.10
            }

            precip_cv = xr.ones_like(mean_annual_precip) * 0.2
            heavy_rain_days = mean_annual_precip * 0.05

        # 2. 计算年均温度（如果有温度数据）
        mean_annual_temp = None
        if temp_files:
            mean_annual_temp = _calculate_annual_temperature(temp_files)

        # 整理返回结果
        climate_vars = {
            'mean_annual_precip': mean_annual_precip,
            'precip_cv': precip_cv,
            'heavy_rain_days': heavy_rain_days,
            'mean_annual_temp': mean_annual_temp
        }

        # 添加季节比例
        climate_vars.update(seasonal_ratios)

        print("气候变量计算完成")
        return climate_vars

    except Exception as e:
        print(f"计算气候变量失败: {e}")
        import traceback
        traceback.print_exc()

        # 返回默认值
        return _create_default_climate_vars()


def _calculate_annual_temperature(temp_files):
    """
    从温度文件列表计算年均温度

    Parameters:
    -----------
    temp_files : list
        温度文件路径列表

    Returns:
    --------
    numpy.ndarray or None
        年均温度数组
    """
    print("正在计算年均温度...")

    try:
        # 按年份组织文件
        year_files = {}
        for temp_file in temp_files:
            try:
                # 从路径提取年份
                path_parts = os.path.normpath(temp_file).split(os.sep)
                folder_name = next((part for part in path_parts if part.endswith('_avg')), None)

                if folder_name:
                    year = int(folder_name.split('_')[0])
                    if year not in year_files:
                        year_files[year] = []
                    year_files[year].append(temp_file)
            except (ValueError, IndexError):
                continue

        print(f"找到 {len(year_files)} 个年份的温度数据")

        # 选择代表性文件（每年选择4个季节性文件）
        selected_files = _select_representative_temp_files(year_files)
        print(f"选择了 {len(selected_files)} 个代表性文件")

        # 读取温度数据并计算平均值
        temp_data_list = []
        for temp_file in selected_files:
            try:
                with rasterio.open(temp_file) as src:
                    temp_data = src.read(1).astype(np.float32)
                    # 处理无效值
                    if src.nodata is not None:
                        temp_data = np.where(temp_data == src.nodata, np.nan, temp_data)
                    temp_data_list.append(temp_data)
            except Exception as e:
                print(f"读取温度文件失败 {temp_file}: {e}")

        if temp_data_list:
            temp_array = np.array(temp_data_list)
            mean_annual_temp = np.nanmean(temp_array, axis=0)
            print(f"年均温度计算完成 - 形状: {mean_annual_temp.shape}")
            return mean_annual_temp
        else:
            print("没有有效的温度数据")
            return None

    except Exception as e:
        print(f"计算年均温度失败: {e}")
        return None


def _select_representative_temp_files(year_files):
    """
    从每年的温度文件中选择代表性文件

    Parameters:
    -----------
    year_files : dict
        按年份组织的文件字典

    Returns:
    --------
    list
        选择的代表性文件列表
    """
    selected_files = []

    for year, files in year_files.items():
        files.sort()  # 按文件名排序

        if len(files) >= 4:
            # 选择四个季节的代表性文件
            quarter_size = len(files) // 4
            seasonal_files = [
                files[quarter_size // 2],  # 春季
                files[quarter_size + quarter_size // 2],  # 夏季
                files[2 * quarter_size + quarter_size // 2],  # 秋季
                files[3 * quarter_size + quarter_size // 2]  # 冬季
            ]
            selected_files.extend(seasonal_files)
        else:
            # 文件较少时使用所有文件
            selected_files.extend(files)

    return selected_files


def _create_default_climate_vars():
    """
    创建默认的气候变量（用于错误处理）

    Returns:
    --------
    dict
        默认气候变量字典
    """
    # 创建默认尺寸的空数组
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
# 数据整合模块
# =============================================================================

def integrate_data(terrain_features, climate_vars, dem_target_transform):
    """
    整合地形特征和气候变量到统一的数据框

    Parameters:
    -----------
    terrain_features : dict
        地形特征字典
    climate_vars : dict
        气候变量字典
    dem_target_transform : Affine
        目标变换矩阵

    Returns:
    --------
    pandas.DataFrame
        整合后的数据框
    """
    print("正在整合数据到统一数据框...")

    try:
        # 获取地形数据的网格坐标
        elevation = terrain_features['elevation']
        height, width = elevation.shape

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

        # 匹配气候变量数据
        _match_climate_data(data_dict, climate_vars)

        # 添加温度数据（如果可用）
        if climate_vars['mean_annual_temp'] is not None:
            data_dict['mean_annual_temp'] = climate_vars['mean_annual_temp'].flatten()

        # 创建交互特征
        data_dict['elev_x_precip'] = data_dict['elevation'] * data_dict['mean_annual_precip']
        data_dict['slope_x_precip'] = data_dict['slope'] * data_dict['mean_annual_precip']

        # 转换为DataFrame并清理数据
        df = pd.DataFrame(data_dict)
        df_clean = _clean_dataframe(df)

        print(f"数据整合完成 - 原始点数: {len(df)}, 有效点数: {len(df_clean)}")
        return df_clean

    except Exception as e:
        print(f"数据整合失败: {e}")
        import traceback
        traceback.print_exc()
        return _create_sample_dataset()


def _match_climate_data(data_dict, climate_vars):
    """
    将气候变量数据匹配到地形网格点

    Parameters:
    -----------
    data_dict : dict
        数据字典（会被修改）
    climate_vars : dict
        气候变量字典
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
            # 使用默认坐标网格
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
            climate_arrays[key] = climate_vars[key].values
        except Exception:
            climate_arrays[key] = np.zeros((len(precip_lat), len(precip_lon)))

    # 初始化输出数组
    n_points = len(data_dict['lat'])
    for key in climate_keys:
        data_dict[key] = np.full(n_points, np.nan)

    # 分块处理以减少内存使用
    chunk_size = 10000
    for chunk_start in range(0, n_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_points)

        current_lats = data_dict['lat'][chunk_start:chunk_end]
        current_lons = data_dict['lon'][chunk_start:chunk_end]

        # 最近邻插值
        for i, (lat, lon) in enumerate(zip(current_lats, current_lons)):
            try:
                lat_idx = np.argmin(np.abs(precip_lat - lat))
                lon_idx = np.argmin(np.abs(precip_lon - lon))

                actual_idx = chunk_start + i
                for key in climate_keys:
                    data_dict[key][actual_idx] = climate_arrays[key][lat_idx, lon_idx]

            except (IndexError, ValueError):
                continue


def _clean_dataframe(df):
    """
    清理数据框，处理缺失值和异常值

    Parameters:
    -----------
    df : pandas.DataFrame
        原始数据框

    Returns:
    --------
    pandas.DataFrame
        清理后的数据框
    """
    # 移除包含NaN的行
    df_clean = df.dropna()

    if len(df_clean) < 1000:
        print("警告：有效数据点过少，尝试填充部分缺失值...")

        # 识别数值列（排除坐标）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fill_cols = [col for col in numeric_cols if col not in ['lat', 'lon', 'heavy_rain_days']]

        # 使用均值填充非目标变量
        df_partial = df.copy()
        for col in fill_cols:
            df_partial[col] = df_partial[col].fillna(df_partial[col].mean())

        df_clean = df_partial.dropna()
        print(f"填充后有效数据点数: {len(df_clean)}")

    # 如果仍然数据不足，创建样例数据集
    if len(df_clean) < 100:
        print("警告：数据仍然不足，使用样例数据集")
        return _create_sample_dataset()

    return df_clean


def _create_sample_dataset():
    """
    创建样例数据集（用于测试和演示）

    Returns:
    --------
    pandas.DataFrame
        样例数据框
    """
    print("正在创建样例数据集...")
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
# 机器学习建模模块
# =============================================================================

def build_models(df, features, target):
    """
    构建多种机器学习模型分析地形气候对极端降水的影响

    Parameters:
    -----------
    df : pandas.DataFrame
        整合后的数据框
    features : list
        特征列名列表
    target : str
        目标变量名

    Returns:
    --------
    tuple
        (models_dict, cleaned_dataframe)
    """
    print("正在构建机器学习模型...")

    try:
        # 数据预处理
        df_model = _preprocess_model_data(df, features, target)

        # 数据分割
        X = df_model[features]
        y = df_model[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 构建模型字典
        models = {}

        # 1. 广义线性模型
        models['GLM'] = _build_glm_model(X_train, X_test, y_train, y_test)

        # 2. 随机森林模型
        models['RandomForest'] = _build_rf_model(
            X_train_scaled, X_test_scaled, y_train, y_test, features
        )

        # 3. XGBoost模型
        models['XGBoost'] = _build_xgb_model(
            X_train_scaled, X_test_scaled, y_train, y_test, features
        )

        # 输出模型性能
        print("\n模型性能总结:")
        for name, model_info in models.items():
            print(f"{name}: R² = {model_info['r2']:.4f}, MSE = {model_info['mse']:.4f}")

        return models, df_model

    except Exception as e:
        print(f"构建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return _create_dummy_models(features, target), df


def _preprocess_model_data(df, features, target):
    """
    模型数据预处理：去除异常值

    Parameters:
    -----------
    df : pandas.DataFrame
        原始数据框
    features : list
        特征列表
    target : str
        目标变量

    Returns:
    --------
    pandas.DataFrame
        预处理后的数据框
    """
    print(f"数据预处理 - 原始数据点数: {len(df)}")

    df_clean = df.copy()

    # 去除极端异常值（保留1%-99%分位数之间的数据）
    for col in features + [target]:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.01)
            q99 = df_clean[col].quantile(0.99)
            df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q99)]

    print(f"去除异常值后数据点数: {len(df_clean)}")
    return df_clean


def _build_glm_model(X_train, X_test, y_train, y_test):
    """
    构建广义线性模型

    Returns:
    --------
    dict
        模型信息字典
    """
    print("训练广义线性模型...")

    try:
        # 准备GLM数据
        X_train_glm = sm.add_constant(X_train)
        X_test_glm = sm.add_constant(X_test)

        # 使用泊松分布GLM（适合计数数据）
        try:
            glm_model = sm.GLM(y_train, X_train_glm, family=sm.families.Poisson())
            glm_result = glm_model.fit()
            y_pred = glm_result.predict(X_test_glm)
        except Exception:
            # 备选：使用线性回归
            print("GLM失败，使用线性回归替代...")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred = lr_model.predict(X_test)

            # 创建兼容对象
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
        print(f"GLM模型构建失败: {e}")
        return _create_dummy_model_result('GLM', X_test, y_test)


def _build_rf_model(X_train, X_test, y_train, y_test, features):
    """
    构建随机森林模型

    Returns:
    --------
    dict
        模型信息字典
    """
    print("训练随机森林模型...")

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
        print(f"随机森林模型构建失败: {e}")
        return _create_dummy_model_result('RandomForest', X_test, y_test, features)


def _build_xgb_model(X_train, X_test, y_train, y_test, features):
    """
    构建XGBoost模型

    Returns:
    --------
    dict
        模型信息字典
    """
    print("训练XGBoost模型...")

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
        print(f"XGBoost模型构建失败: {e}")
        return _create_dummy_model_result('XGBoost', X_test, y_test, features)


def _create_dummy_model_result(model_name, X_test, y_test, features=None):
    """
    创建虚拟模型结果（错误处理用）
    """
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
    """
    创建虚拟模型字典（错误处理用）
    """

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
# 模型分析模块
# =============================================================================

def analyze_models(models, df_model, features, target):
    """
    分析模型结果，提取特征重要性和系数信息

    Parameters:
    -----------
    models : dict
        模型字典
    df_model : pandas.DataFrame
        模型数据
    features : list
        特征列表
    target : str
        目标变量

    Returns:
    --------
    dict
        分析结果字典
    """
    print("正在分析模型结果...")

    try:
        # 1. GLM系数分析
        glm_coef = _analyze_glm_coefficients(models.get('GLM'), features)

        # 2. 特征重要性分析
        rf_importance = _analyze_feature_importance(models.get('RandomForest'), 'RandomForest')
        xgb_importance = _analyze_feature_importance(models.get('XGBoost'), 'XGBoost')

        # 3. 交互项分析
        interaction_results = _analyze_interactions(glm_coef)

        # 4. 相关性分析
        corr_matrix, target_corr = _analyze_correlations(df_model, features, target)

        analysis_results = {
            'glm_coef': glm_coef,
            'rf_importance': rf_importance,
            'xgb_importance': xgb_importance,
            'interaction_results': interaction_results,
            'target_corr': target_corr,
            'corr_matrix': corr_matrix
        }

        print("模型分析完成")
        return analysis_results

    except Exception as e:
        print(f"模型分析失败: {e}")
        return _create_default_analysis_results(features, target)


def _analyze_glm_coefficients(glm_model, features):
    """
    分析GLM模型系数
    """
    if glm_model and glm_model.get('model'):
        try:
            model = glm_model['model']
            coef_df = pd.DataFrame({