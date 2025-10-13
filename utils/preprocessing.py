"""
Anomalyze Preprocessing Module
Optimized data preprocessing for network intrusion detection
Python 3.12 Compatible
"""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import numpy as np
import sys
from pathlib import Path
from typing import Union

def load_and_preprocess_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the NSL-KDD dataset, assigns column names, and handles categorical features.
    """
    #Column names for the NSL-KDD dataset
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty'
    ]

    try:
        if hasattr(file_path, 'read'):
            df = pd.read_csv(file_path, header = None, names = columns)
        else:
            df = pd.read_csv(file_path, header = None, names = columns)
        
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis = 1)

        # Handle missing values
        df = df.fillna(0)

        # Encode categorical variables
        categorical_columns = ['protocol_type', 'service', 'flag']

        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Focus on the most important features for anomaly tdetection
        important_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_serror_rate'
        ]

        # Keep only the important features (label it if it's existed)
        features_to_keep = important_features.copy()
        if 'label' in df.columns:
            features_to_keep.append('label')

        # Filter to only the important features that exist in the DataFrame
        features_to_keep = [f for f in features_to_keep if f in df.columns]
        df =df[features_to_keep]

        # Ensure all remaining columns are numeric (Except for the label)
        for col in df.columns:
            if col != 'label':
                # Force conversion to numeric, replacing any non-numeric values with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Ensure the column is float type for consistency
                df[col] = df[col].astype(float)

                # Apply log transformation to highly skewed features
                if col in ['src_bytes', 'dst_bytes', 'count', 'srv_count']:
                    df[col] = np.log1p(df[col])  # log1p handles zeros better

        return df
    except Exception as e:
        print(f"Error in processing: {e}")
        raise

def create_advanced_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advance network-specific features optimized for K-means clustering"""
    df_enhanced = df.copy()

    # Traffic intensity ratios (check if columns exist first)
    if 'src_bytes' in df_enhanced.columns and 'dst_bytes' in df_enhanced.columns:
        df_enhanced['bytes_ratio'] = df_enhanced['src_bytes'] / (df_enhanced['dst_bytes'] + 1)
        df_enhanced['total_bytes'] = df_enhanced['src_bytes'] + df_enhanced['dst_bytes']

    # Connection pattern features
    if 'srv_count' in df_enhanced.columns and 'count' in df_enhanced.columns:
        df_enhanced['srv_rate'] = df_enhanced['srv_count'] / (df_enhanced['count'] + 1)
    
    if 'serror_rate' in df_enhanced.columns and 'srv_serror_rate' in df_enhanced.columns:
        df_enhanced['error_density'] = (df_enhanced['serror_rate'] + df_enhanced['srv_serror_rate']) / 2

    # Protocol behavior patterns
    if 'duration' in df_enhanced.columns and 'total_bytes' in df_enhanced.columns:
        df_enhanced['bytes_per_second'] = df_enhanced['total_bytes'] / (df_enhanced['duration'] + 0.001)
        df_enhanced['duration_category'] = pd.cut(df_enhanced['duration'],
                                                  bins = [0, 1,10, 100, float('inf')],
                                                  labels = [0, 1, 2, 3])
        
    # Host behavior clustering features (check if columns exist)
    if 'dst_host_diff_srv_rate' in df_enhanced.columns and 'dst_host_srv_count' in df_enhanced.columns:
        df_enhanced['host_diversity'] = (df_enhanced['dst_host_diff_srv_rate'] *
                                       df_enhanced['dst_host_srv_count'])
    
    # Attack pattern indicator (check if columns exist)
    if 'su_attempted' in df_enhanced.columns and 'root_shell' in df_enhanced.columns:
        df_enhanced['sus_flag_ratio'] = (df_enhanced['su_attempted'] + df_enhanced['root_shell']) / 2
    
    if 'land' in df_enhanced.columns:
        df_enhanced['land_flag'] = df_enhanced['land']

    # Network flow characteristics
    if 'same_srv_rate' in df_enhanced.columns:
        # Ensure numeric type before comparison
        df_enhanced['same_srv_rate'] = pd.to_numeric(df_enhanced['same_srv_rate'], errors='coerce').fillna(0).astype(float)
        df_enhanced['same_srv_rate_high'] = (df_enhanced['same_srv_rate'] > 0.8).astype(int)
    if 'diff_srv_rate' in df_enhanced.columns:
        # Ensure numeric type before comparison
        df_enhanced['diff_srv_rate'] = pd.to_numeric(df_enhanced['diff_srv_rate'], errors='coerce').fillna(0).astype(float)
        df_enhanced['diff_srv_rate_high'] = (df_enhanced['diff_srv_rate'] > 0.8).astype(int)

    return df_enhanced

def enhanced_preprocessing_for_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced preprocessing specifically optimized for K-means clustering"""
    
    try:
        # Apply feature engineering only if we have the required columns
        df = create_advanced_network_features(df)

        # Handle categorical variables better for clustering
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'label':
                # Use frequency encoding for high cardinality categoricals
                freq_encoding = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq_encoding)
                df = df.drop(col, axis=1)

        # Note: Don't apply scaling here as it will be done later with the saved scaler
        
        return df
        
    except Exception as e:
        print(f"Error in enhanced preprocessing: {e}")
        # Return original dataframe if enhancement fails
        return df