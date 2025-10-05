import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess_data(file_path):
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

        # Ensure all remaining columns are numeric (Except for the wlabel)
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors = 'coerce').fillna(0)

                # Apply log transformation to highly skewed features
                if col in ['src_bytes', 'dst_bytes', 'count', 'srv_count']:
                    df[col] = np.log1p(df[col]) # log1p handles zeros better

        return df
    except Exception as e:
        print(f"Error in processing: {e}")
        raise