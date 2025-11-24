# enterprise_aiops_merged_full.py
"""
Merged single-file: UI from enterprise AIoPS.py + backend functions from enterprise_aiops_refined.py
Preserves the original enterprise AIoPS.py UI and injects the refined backend implementations.
Sources: enterprise AIoPS.py, enterprise_aiops_refined.py. (User-provided files)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import io

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="AIOps Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Exact Elite Professional Styling (kept from enterprise AIoPS.py)
# -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
    }
    
    h1 {
        color: #1a202c;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 1.5rem;
    }
    
    .stTabs {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f7fafc;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 10px;
        padding: 0 24px;
        font-weight: 600;
        color: #4a5568;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: none;
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white;
        font-size: 2rem;
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: rgba(255, 255, 255, 0.8);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Session state initialization (same as enterprise AIoPS.py)
# -------------------------
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'feature_data' not in st.session_state:
    st.session_state.feature_data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'regression_target' not in st.session_state:
    st.session_state.regression_target = 'resolution_time'

# -------------------------
# Backend: AdvancedDataProcessor (merged/refined)
# -------------------------
class AdvancedDataProcessor:
    @staticmethod
    def intelligent_clean(df):
        """Advanced data cleaning with intelligent imputation"""
        df_clean = df.copy()
        
        # Handle timestamp
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
            df_clean = df_clean.dropna(subset=['timestamp'])
        
        # Intelligent numeric imputation (Median)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical (Mode or 'unknown')
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown')
        
        return df_clean
    
    @staticmethod
    def advanced_feature_engineering(df):
        """Comprehensive feature engineering with advanced techniques"""
        df_feat = df.copy()
        
        # Ensure 'incident_flag' is created robustly for the classification target
        if 'incident_type' in df_feat.columns:
            df_feat['incident_flag'] = (~df_feat['incident_type'].astype(str).str.lower().isin(['none', 'no_incident'])).astype(int)
        else:
            df_feat['incident_flag'] = 0
            st.warning("Warning: 'incident_type' column not found. Incident prediction model will be trained on a dummy target (all 0s).")
        
        if 'timestamp' in df_feat.columns:
            df_feat = df_feat.sort_values('timestamp').reset_index(drop=True)
            
            # Temporal features
            df_feat['hour'] = df_feat['timestamp'].dt.hour
            df_feat['day_of_week'] = df_feat['timestamp'].dt.dayofweek
            df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        
        # Identify telemetry columns
        telemetry_cols = [col for col in df_feat.columns if any(x in col.lower() for x in 
                          ['cpu', 'memory', 'disk', 'network', 'usage', 'traffic', 'io'])]
        
        # Advanced rolling statistics (window=5 for simplicity)
        for col in telemetry_cols:
            if col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[col]):
                df_feat[f'{col}_roll_mean_5'] = df_feat.groupby('system_id')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                ) if 'system_id' in df_feat.columns else df_feat[col].rolling(window=5, min_periods=1).mean().shift(1)
        
        # Fill remaining NaN from rolling features
        df_feat = df_feat.fillna(0)
        
        return df_feat

# -------------------------
# Backend: LiveMLPipeline (refined)
# -------------------------
class LiveMLPipeline:
    def __init__(self):
        self.class_model = None
        self.reg_model = None
        self.anom_model = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    def _prepare_data(self, df, target_col=None, is_training=True):
        """Prepares data for ML models (scaling, encoding, feature selection)."""

        # Drop only truly irrelevant columns first
        exclude_cols = ['timestamp', 'system_id', 'incident_flag']
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')

        # Encode categorical columns safely if they exist
        categorical_cols = [c for c in ['incident_type', 'priority'] if c in X.columns]
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        if is_training:
            # Save feature columns
            self.feature_columns = list(X.columns)
            # Scale
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            if target_col and target_col in X_scaled.columns:
                X_scaled = X_scaled.drop(columns=[target_col])
            return X_scaled
        else:
            # Ensure same columns as training
            for c in set(self.feature_columns) - set(X.columns):
                X[c] = 0
            X = X[self.feature_columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
            if target_col and target_col in X_scaled.columns:
                X_scaled = X_scaled.drop(columns=[target_col])
            return X_scaled

    def train_classification(self, df, test_size=0.2):
        if 'incident_flag' not in df.columns:
            st.error("Error: 'incident_flag' column is missing.")
            return None, None

        X = self._prepare_data(df, is_training=True)
        y = df['incident_flag']
        if y.nunique() < 2:
            st.warning("Classification target is single-class. Skipping training.")
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.class_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, class_weight='balanced'
        )
        self.class_model.fit(X_train, y_train)
        y_pred = self.class_model.predict(X_test)
        y_proba = self.class_model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        try:
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
        except:
            auc, fpr, tpr = None, [], []

        cm = confusion_matrix(y_test, y_pred)
        results = {
            'report': report,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm,
            'feature_importance': self.class_model.feature_importances_
        }
        return self.class_model, results

    def train_regression(self, df, target_col='resolution_time', test_size=0.2):
        if target_col not in df.columns:
            st.warning(f"Regression target '{target_col}' not found.")
            return None, None

        X = self._prepare_data(df, target_col=target_col, is_training=True)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        self.reg_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        self.reg_model.fit(X_train, y_train)
        preds = self.reg_model.predict(X_test)

        results = {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
            'r2': r2_score(y_test, preds),
            'y_test': y_test,
            'predictions': preds
        }
        return self.reg_model, results

    def train_anomaly_detection(self, df, contamination=0.01):
        X = self._prepare_data(df, is_training=True)
        self.anom_model = IsolationForest(contamination=contamination, random_state=42)
        self.anom_model.fit(X)

        labels = self.anom_model.predict(X)
        scores = self.anom_model.decision_function(X)
        results = {
            'anomaly_count': (labels == -1).sum(),
            'anomaly_percentage': ((labels == -1).sum() / len(df)) * 100 if len(df) > 0 else 0,
            'anomaly_labels': labels,
            'anomaly_scores': scores
        }
        return self.anom_model, results

    def predict_live(self, df):
        X = self._prepare_data(df, is_training=False)
        preds = {}

        if self.class_model:
            preds['incident_probability'] = self.class_model.predict_proba(X)[:, 1]
            preds['incident_prediction'] = self.class_model.predict(X)

        if self.reg_model:
            reg_target = st.session_state.get('regression_target', 'resolution_time')
            X_reg = self._prepare_data(df, target_col=reg_target, is_training=False)
            preds[f'predicted_{reg_target}'] = self.reg_model.predict(X_reg)

        if self.anom_model:
            preds['is_anomaly'] = self.anom_model.predict(X)
            preds['anomaly_score'] = self.anom_model.decision_function(X)

        return preds
# -------------------------
# Visualizer (merged/refined)
# -------------------------
class EliteVisualizer:
    
    COLOR_PALETTE = ['#1e3a8a', '#3b82f6', '#93c5fd', '#f87171', '#ef4444', '#10b981']
    
    @staticmethod
    def create_premium_kpi_cards(df):
        """Creates professional KPI cards."""
        total_records = len(df)
        incidents = int(df['incident_flag'].sum()) if 'incident_flag' in df.columns else 0
        incident_rate = (incidents / total_records) * 100 if total_records > 0 else 0
        
        mttr_avg = df['resolution_time'].mean() if 'resolution_time' in df.columns else 120
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records Analyzed", f"{total_records:,}")
        with col2:
            st.metric("Historical Incidents", f"{incidents:,}", delta=f"{incident_rate:.2f}% of total")
        with col3:
            st.metric("Avg Resolution Time (min)", f"{mttr_avg:.1f}")
        with col4:
            systems_affected = df['system_id'].nunique() if 'system_id' in df.columns else 0
            st.metric("Systems Monitored", f"{systems_affected}")

    @staticmethod
    def create_modern_timeseries(df):
        """Create modern time series visualization for core telemetry."""
        telemetry_cols = [col for col in df.columns if any(x in col.lower() for x in 
                          ['cpu_usage', 'memory_usage', 'disk_io', 'network_traffic'])]
        
        if not telemetry_cols or 'timestamp' not in df.columns:
            return None
        
        # Resample for performance and cleaner look
        df_resampled = df.set_index('timestamp')[telemetry_cols].resample('1H').mean().reset_index()
        
        fig = make_subplots(
            rows=len(telemetry_cols), cols=1,
            subplot_titles=[col.replace('_', ' ').title() for col in telemetry_cols],
            vertical_spacing=0.08
        )
        
        for idx, col in enumerate(telemetry_cols):
            fig.add_trace(
                go.Scatter(
                    x=df_resampled['timestamp'],
                    y=df_resampled[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=EliteVisualizer.COLOR_PALETTE[idx % len(EliteVisualizer.COLOR_PALETTE)], width=2),
                    fill='tozeroy',
                    hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
                ),
                row=idx+1, col=1
            )
        
        fig.update_layout(
            height=300 * max(1, len(telemetry_cols)),
            showlegend=False,
            template='plotly_white',
            title_text="Core Telemetry Trends (Hourly Average)",
            title_font=dict(size=20, color='#334155', family='Inter'),
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        
        return fig

    @staticmethod
    def create_3d_heatmap(df):
        """Create advanced temporal incident heatmap."""
        if 'incident_type' not in df.columns or 'timestamp' not in df.columns:
            return None
            
        df_incidents = df[~df['incident_type'].astype(str).str.lower().isin(['none', 'no_incident'])].copy()
        
        if len(df_incidents) == 0:
            return None
            
        df_incidents['hour'] = df_incidents['timestamp'].dt.hour
        df_incidents['day_of_week'] = df_incidents['timestamp'].dt.dayofweek
        
        pivot_data = df_incidents.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        pivot_matrix = pivot_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_matrix.values,
            x=pivot_matrix.columns,
            y=[day_names[i] for i in pivot_matrix.index],
            colorscale='Blues',
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Incidents: %{z}<extra></extra>',
            colorbar=dict(title=dict(text="Incidents"))
        ))
        
        fig.update_layout(
            title='Incident Temporal Distribution (Day of Week vs Hour)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=500,
            template='plotly_white',
            title_font=dict(size=18, color='#334155')
        )
        
        return fig
    
    @staticmethod
    def create_roc_curve(fpr, tpr, auc):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})' if auc is not None else 'ROC Curve',
            line=dict(color=EliteVisualizer.COLOR_PALETTE[0], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='#cbd5e0', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=450,
            template='plotly_white',
            title_font=dict(size=18, color='#334155'),
            hovermode='closest'
        )
        
        return fig

    @staticmethod
    def create_feature_importance(importance, features, top_n=15):
        # Ensure importance and features are aligned and sorted
        feature_series = pd.Series(importance, index=features).sort_values(ascending=False).head(top_n)
        
        fig = go.Figure(go.Bar(
            x=feature_series.values,
            y=feature_series.index,
            orientation='h',
            marker=dict(
                color=feature_series.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Top Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='',
            height=500,
            template='plotly_white',
            title_font=dict(size=18, color='#334155'),
            yaxis={'autorange': 'reversed'}
        )
        
        return fig

    @staticmethod
    def create_confusion_matrix(cm, labels=['No Incident', 'Incident']):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        text = [[f'{cm[i][j]}<br>({cm_normalized[i][j]:.1f}%)' for j in range(len(cm[0]))] for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=450,
            template='plotly_white',
            title_font=dict(size=18, color='#334155')
        )
        
        return fig

    @staticmethod
    def create_anomaly_visualization(df, anomaly_labels):
        if 'timestamp' not in df.columns:
            return None
            
        telemetry_col = 'cpu_usage' if 'cpu_usage' in df.columns else df.select_dtypes(include=np.number).columns[0]
        
        fig = go.Figure()
        
        normal_mask = anomaly_labels != -1
        fig.add_trace(go.Scatter(
            x=df[normal_mask]['timestamp'],
            y=df[normal_mask][telemetry_col],
            mode='markers',
            name='Normal',
            marker=dict(size=5, color='#93c5fd', opacity=0.6),
            hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
        ))
        
        anomaly_mask = anomaly_labels == -1
        fig.add_trace(go.Scatter(
            x=df[anomaly_mask]['timestamp'],
            y=df[anomaly_mask][telemetry_col],
            mode='markers',
            name='Anomaly',
            marker=dict(size=8, color='#ef4444', symbol='circle', line=dict(width=1, color='#b91c1c')),
            hovertemplate='<b>ANOMALY</b><br>%{y:.2f}<br>%{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Real-Time Anomaly Detection on {telemetry_col.replace("_", " ").title()}',
            xaxis_title='Timestamp',
            yaxis_title=telemetry_col.replace('_', ' ').title(),
            height=500,
            template='plotly_white',
            title_font=dict(size=18, color='#334155'),
            hovermode='x unified'
        )
        
        return fig

# -------------------------
# Utility: sample data generator (from refined)
# -------------------------
@st.cache_data
def generate_sample_data(n_rows=5000):
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=30)
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start=start_date, periods=n_rows, freq='10min')),
        'system_id': np.random.choice([f'SYS-{i:03d}' for i in range(10)], n_rows),
        'incident_type': np.random.choice(['Network', 'Server', 'Application', 'none'], n_rows, p=[0.1, 0.15, 0.05, 0.7]),
        'priority': np.random.choice(['High', 'Medium', 'Low'], n_rows),
        'cpu_usage': np.clip(np.random.normal(loc=50, scale=20, size=n_rows), 0, 100),
        'memory_usage': np.clip(np.random.normal(loc=60, scale=15, size=n_rows), 0, 100),
        'disk_io': np.clip(np.random.lognormal(mean=1.5, sigma=0.5, size=n_rows), 0, 10),
        'network_traffic': np.clip(np.random.lognormal(mean=3, sigma=1, size=n_rows), 0, 50),
        'change_count': np.random.randint(0, 5, n_rows),
        'tickets_opened': np.random.randint(0, 10, n_rows),
        'resolution_time': np.random.randint(30, 300, n_rows)
    }
    df = pd.DataFrame(data)
    for col in ['cpu_usage', 'network_traffic', 'resolution_time']:
        df.loc[df.sample(frac=0.02).index, col] = np.nan
    return df

# -------------------------
# Main Application (UI preserved)
# -------------------------
def main():
    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Enterprise AIOps Analytics Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #718096; font-size: 1.1rem; margin-top: 0;'>AI-Powered Incident Prediction & Operations Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")
        
        if st.session_state.get('cleaned_data') is not None:
            st.success(f"Dataset Active: {len(st.session_state.cleaned_data):,} records")
            
            st.markdown("### Data Filters")
            
            # System filter
            if 'system_id' in st.session_state.cleaned_data.columns:
                systems = ['All Systems'] + sorted(st.session_state.cleaned_data['system_id'].unique().tolist())
                selected_system = st.selectbox("System", systems, key='system_filter')
            else:
                selected_system = 'All Systems'
            
            # Incident type filter
            if 'incident_type' in st.session_state.cleaned_data.columns:
                incident_types = ['All Types'] + sorted(st.session_state.cleaned_data['incident_type'].unique().tolist())
                selected_incident = st.selectbox("Incident Type", incident_types, key='incident_filter')
            else:
                selected_incident = 'All Types'
            
            # Date range filter
            if 'timestamp' in st.session_state.cleaned_data.columns:
                min_date = st.session_state.cleaned_data['timestamp'].min().date()
                max_date = st.session_state.cleaned_data['timestamp'].max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key='date_filter'
                )
            
            # Apply filters
            filtered_data = st.session_state.cleaned_data.copy()
            
            if selected_system != 'All Systems' and 'system_id' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['system_id'] == selected_system]
            
            if selected_incident != 'All Types' and 'incident_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['incident_type'] == selected_incident]
            
            if 'timestamp' in filtered_data.columns and len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['timestamp'].dt.date >= date_range[0]) &
                    (filtered_data['timestamp'].dt.date <= date_range[1])
                ]
            
            st.session_state.filtered_data = filtered_data
            st.info(f"Filtered: {len(filtered_data):,} records")
            
            st.markdown("---")
            
            # Model status
            if st.session_state.models_trained:
                st.success("ML Models: Active")
            else:
                st.warning("ML Models: Not Trained")
        else:
            st.info("Upload data to begin analysis")
        
        # Upload area (kept in sidebar for convenience)
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Drop CSV file here or click to browse",
            type=['csv'],
            help="Maximum file size: 200 MB"
        )
        if uploaded_file is not None:
            try:
                with st.spinner("Loading dataset..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.success(f"Dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main Tabs (exact layout from enterprise AIoPS.py)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Upload",
        "Data Processing",
        "Analytics Dashboard",
        "ML Predictions",
        "Live Monitoring",
        "Export Center"
    ])
    
    # Tab 1: Data Upload
    with tab1:
        st.markdown("## Data Upload Center")
        st.markdown("Upload your IT operations telemetry data for intelligent analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file2 = st.file_uploader(
                "Upload CSV here (or use sidebar)",
                type=['csv'],
                key='uploader_main'
            )
            if uploaded_file2 is not None and st.session_state.get('data') is None:
                try:
                    with st.spinner("Loading dataset..."):
                        df2 = pd.read_csv(uploaded_file2)
                        st.session_state.data = df2
                    st.success(f"Dataset loaded successfully: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        with col2:
            st.markdown("### Requirements")
            st.markdown("""
            - **Format**: CSV
            - **Size**: ≤ 200 MB
            - **Encoding**: UTF-8
            - **Columns**: timestamp, system_id, metrics
            """)
        
        if st.session_state.get('data') is not None:
            df = st.session_state.data
            st.markdown("### Data Preview")
            st.dataframe(df.head(50), use_container_width=True, height=400)
            
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Records", f"{len(df):,}")
            with colB:
                st.metric("Features", f"{len(df.columns)}")
            with colC:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with colD:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.2f}%")
            
            st.markdown("### Column Analysis")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
    
    # Tab 2: Data Processing
    with tab2:
        st.markdown("## Data Processing Pipeline")
        
        if st.session_state.data is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Automated Data Processing")
                st.markdown("""
                Our advanced pipeline performs:
                - Intelligent missing value imputation
                - Automated feature engineering
                - Time-series aware transformations
                - System-level aggregations
                """)
            
            with col2:
                if st.button("Process Data", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Cleaning
                    status_text.text("Step 1/2: Cleaning data...")
                    progress_bar.progress(33)
                    processor = AdvancedDataProcessor()
                    cleaned_df = processor.intelligent_clean(st.session_state.data)
                    st.session_state.cleaned_data = cleaned_df
                    st.session_state.filtered_data = cleaned_df
                    
                    # Step 2: Feature Engineering
                    status_text.text("Step 2/2: Engineering features...")
                    progress_bar.progress(66)
                    feature_df = processor.advanced_feature_engineering(cleaned_df)
                    st.session_state.feature_data = feature_df
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    st.success(f"Data processed successfully: {len(feature_df.columns)} total features")
            
            if st.session_state.cleaned_data is not None:
                st.markdown("---")
                st.markdown("### Processing Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Data Quality")
                    original_missing = st.session_state.data.isnull().sum().sum()
                    cleaned_missing = st.session_state.cleaned_data.isnull().sum().sum()
                    st.metric("Missing Values Resolved", f"{original_missing - cleaned_missing:,}")
                
                with col2:
                    st.markdown("#### Feature Engineering")
                    if st.session_state.feature_data is not None:
                        original_cols = len(st.session_state.cleaned_data.columns)
                        total_cols = len(st.session_state.feature_data.columns)
                        st.metric("Features Generated", f"{total_cols - original_cols}")
                
                with col3:
                    st.markdown("#### Dataset Size")
                    st.metric("Final Record Count", f"{len(st.session_state.cleaned_data):,}")
                
                if st.session_state.feature_data is not None:
                    with st.expander("View Feature Categories"):
                        new_cols = [col for col in st.session_state.feature_data.columns 
                                   if col not in st.session_state.cleaned_data.columns]
                        
                        categories = {
                            'Temporal Features': [c for c in new_cols if any(x in c for x in ['hour', 'day', 'week', 'month', 'quarter', 'weekend', 'business', 'night'])],
                            'Rolling Statistics': [c for c in new_cols if 'roll' in c],
                            'Rate of Change': [c for c in new_cols if any(x in c for x in ['diff', 'pct_change', 'ewm'])],
                            'Lag Features': [c for c in new_cols if 'lag' in c],
                            'System Aggregates': [c for c in new_cols if 'sys' in c]
                        }
                        
                        for cat, features in categories.items():
                            if features:
                                st.markdown(f"**{cat}** ({len(features)})")
                                st.write(features[:10])
                                if len(features) > 10:
                                    st.caption(f"... and {len(features) - 10} more")
        else:
            st.warning("Please upload data in the Data Upload tab first")
    
    # Tab 3: Analytics Dashboard
    with tab3:
        st.markdown("## Real-Time Analytics Dashboard")
        
        if st.session_state.cleaned_data is not None:
            viz = EliteVisualizer()
            data_to_viz = st.session_state.get('filtered_data', st.session_state.cleaned_data)
            
            # Sample for performance
            if len(data_to_viz) > 10000:
                data_sample = data_to_viz.sample(n=10000, random_state=42).sort_values('timestamp') if 'timestamp' in data_to_viz.columns else data_to_viz.sample(n=10000, random_state=42)
                st.info(f"Displaying 10,000 sample records from {len(data_to_viz):,} total for optimal performance.")
            else:
                data_sample = data_to_viz
            
            # KPI Section
            st.subheader("Key Operational Metrics")
            viz.create_premium_kpi_cards(data_to_viz)
            
            st.markdown("---")
            
            # Time Series
            st.subheader("Telemetry Monitoring")
            fig_ts = viz.create_modern_timeseries(data_sample)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
            
            # Heatmap
            st.subheader("Incident Temporal Distribution")
            fig_heatmap = viz.create_3d_heatmap(data_to_viz)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Heatmap not generated. Ensure your data contains 'incident_type' and 'timestamp' columns and has recorded incidents.")
        else:
            st.warning("Please process data first in the Data Processing tab.")
    
    # Tab 4: ML Predictions
    with tab4:
        st.markdown("## Machine Learning")
        st.markdown("Train and evaluate incident classification, regression, and anomaly detection models.")
        
        if st.session_state.feature_data is None:
            st.warning("Please process data first (Data Processing).")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                prediction_target = st.selectbox("Regression Target", ['resolution_time','cpu_usage','memory_usage','network_traffic','disk_io'])
                st.session_state['regression_target'] = prediction_target
            with col3:
                if st.button("Train All Models"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    ml_pipeline = LiveMLPipeline()
                    
                    # Train Classification
                    status_text.text("Training incident classification model.")
                    progress_bar.progress(25)
                    class_model, class_results = ml_pipeline.train_classification(
                        st.session_state.feature_data, 
                        test_size=test_size
                    )
                    st.session_state.class_results = class_results
                    
                    # Train Regression
                    status_text.text("Training workload prediction model.")
                    progress_bar.progress(50)
                    reg_model, reg_results = ml_pipeline.train_regression(
                        st.session_state.feature_data,
                        target_col=prediction_target,
                        test_size=test_size
                    )
                    st.session_state.reg_results = reg_results
                    
                    # Train Anomaly Detection
                    status_text.text("Training anomaly detection model.")
                    progress_bar.progress(75)
                    anom_model, anom_results = ml_pipeline.train_anomaly_detection(
                        st.session_state.feature_data
                    )
                    st.session_state.anom_results = anom_results
                    
                    # Store pipeline and state
                    st.session_state.ml_pipeline = ml_pipeline
                    st.session_state.models_trained = True
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    st.success("All models trained successfully and are ready for live prediction.")
            
            if st.session_state.models_trained:
                st.markdown("---")
                st.subheader("Model Performance Analysis")
                viz = EliteVisualizer()
                
                # Classification Results
                st.markdown("#### Incident Classification Model (Random Forest)")
                class_results = st.session_state.class_results
                
                if class_results and class_results.get('auc') is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{class_results['report']['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision (Incident)", f"{class_results['report'].get('1', {}).get('precision', 0):.3f}")
                    with col3:
                        st.metric("Recall (Incident)", f"{class_results['report'].get('1', {}).get('recall', 0):.3f}")
                    with col4:
                        st.metric("ROC-AUC", f"{class_results['auc']:.3f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_cm = viz.create_confusion_matrix(class_results['confusion_matrix'])
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with col2:
                        fig_roc = viz.create_roc_curve(class_results['fpr'], class_results['tpr'], class_results['auc'])
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    fig_importance = viz.create_feature_importance(
                        class_results['feature_importance'],
                        st.session_state.ml_pipeline.feature_columns
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Classification model training skipped due to single-class target or missing data.")
                
                st.markdown("---")
                
                # Regression Results
                st.markdown(f"#### Workload Prediction Model (Target: {prediction_target})")
                reg_results = st.session_state.reg_results
                
                if reg_results:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{reg_results['rmse']:.2f}")
                    with col2:
                        st.metric("MAE", f"{reg_results['mae']:.2f}")
                    with col3:
                        st.metric("R² Score", f"{reg_results['r2']:.3f}")
                else:
                    st.info("Regression model training skipped due to missing target column.")
                
                # Anomaly Detection Results
                st.markdown("#### Anomaly Detection Model (Isolation Forest)")
                anom_results = st.session_state.anom_results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Anomalies Detected", f"{anom_results['anomaly_count']:,}")
                with col2:
                    st.metric("Anomaly Rate", f"{anom_results['anomaly_percentage']:.2f}%")
    
    # Tab 5: Live Monitoring
    with tab5:
        st.markdown("## Live Monitoring & Predictions")
        if not st.session_state.get('models_trained', False):
            st.warning("Train models first in the ML Predictions tab.")
        else:
            if st.button("Run Live Predictions on Processed Data"):
                with st.spinner("Generating live predictions..."):
                    ml = st.session_state.ml_pipeline
                    # Predict on the latest processed feature_data
                    preds = ml.predict_live(st.session_state.feature_data)
                    pred_df = st.session_state.feature_data.copy()
                    for k, v in preds.items():
                        pred_df[k] = v
                    st.session_state.live_predictions = pred_df
                st.success("Predictions generated.")
            
            if st.session_state.get('live_predictions') is not None:
                st.markdown("### Prediction Results")
                pred_df = st.session_state.live_predictions
                
                # High Risk Incidents
                if 'incident_probability' in pred_df.columns:
                    high_risk = pred_df[pred_df['incident_probability'] > 0.7]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk Incidents", f"{len(high_risk):,}")
                    with col2:
                        st.metric("Avg Incident Probability", f"{pred_df['incident_probability'].mean():.1%}")
                    with col3:
                        anomaly_count = (pred_df['is_anomaly'] == -1).sum() if 'is_anomaly' in pred_df.columns else 0
                        st.metric("Anomalies", f"{anomaly_count:,}")
                    
                    if len(high_risk) > 0:
                        st.markdown("#### High Risk Alert")
                        st.warning(f"{len(high_risk)} high-risk incidents predicted (>70% probability)")
                        st.dataframe(high_risk.head(50), use_container_width=True)
                
                st.markdown("#### All Predictions")
                st.dataframe(pred_df.head(100), use_container_width=True)
                
                # Anomaly visualization if available
                if 'is_anomaly' in pred_df.columns:
                    viz = EliteVisualizer()
                    fig_anom = viz.create_anomaly_visualization(pred_df.head(5000), pred_df['is_anomaly'])
                    if fig_anom:
                        st.plotly_chart(fig_anom, use_container_width=True)
    
    # Tab 6: Export Center
    with tab6:
        st.markdown("## Export Center")
        st.markdown("Download processed data and model results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Exports")
            
            if st.session_state.cleaned_data is not None:
                csv_clean = st.session_state.cleaned_data.to_csv(index=False)
                st.download_button(
                    label="Download Cleaned Data",
                    data=csv_clean,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if st.session_state.feature_data is not None:
                csv_features = st.session_state.feature_data.to_csv(index=False)
                st.download_button(
                    label="Download Feature Data",
                    data=csv_features,
                    file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if 'live_predictions' in st.session_state:
                csv_pred = st.session_state.live_predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv_pred,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("### Model Reports")
            
            if st.session_state.models_trained:
                report_text = "ENTERPRISE AIOPS PLATFORM - ML MODEL REPORT\n"
                report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report_text += "="*80 + "\n\n"
                
                # Classification Report
                class_results = st.session_state.class_results
                if class_results and class_results.get('auc') is not None:
                    report_text += "INCIDENT CLASSIFICATION MODEL\n"
                    report_text += "-"*80 + "\n"
                    report_text += f"Accuracy:  {class_results['report']['accuracy']:.4f}\n"
                    report_text += f"ROC-AUC:   {class_results['auc']:.4f}\n\n"
                
                # Regression Report
                reg_results = st.session_state.reg_results
                if reg_results:
                    report_text += "WORKLOAD PREDICTION MODEL\n"
                    report_text += "-"*80 + "\n"
                    report_text += f"RMSE:      {reg_results['rmse']:.4f}\n"
                    report_text += f"R² Score:  {reg_results['r2']:.4f}\n\n"
                
                # Anomaly Report
                anom_results = st.session_state.anom_results
                report_text += "ANOMALY DETECTION MODEL\n"
                report_text += "-"*80 + "\n"
                report_text += f"Anomalies:       {anom_results['anomaly_count']:,}\n"
                report_text += f"Anomaly Rate:    {anom_results['anomaly_percentage']:.2f}%\n"
                
                st.download_button(
                    label="Download Model Report (TXT)",
                    data=report_text,
                    file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.info("Train models to generate a performance report.")

if __name__ == "__main__":
    main()
