import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from datetime import datetime, timedelta
import warnings
import traceback
from typing import Optional, Dict, Any
warnings.filterwarnings('ignore')

# Custom CSS for better styling and error handling
st.set_page_config(
    page_title="🏥 Advanced Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with error styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-container {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #ff4757;
    }
    .success-container {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #2ed573;
    }
    .info-container {
        background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #3742fa;
    }
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .dropdown-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .loading-container {
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 5px;
        color: #333 !important;
    }
    .stSelectbox > div > div > div {
        color: #333 !important;
    }
    .stSelectbox > div > div > div > div {
        color: #333 !important;
    }
    /* Ensure dropdown text is visible */
    .stSelectbox label {
        color: #333 !important;
        font-weight: 500;
    }
    /* Style the dropdown options */
    .stSelectbox ul {
        background-color: white !important;
        color: #333 !important;
    }
    .stSelectbox li {
        color: #333 !important;
        background-color: white !important;
    }
    .stSelectbox li:hover {
        background-color: #f0f0f0 !important;
    }
    /* Additional dropdown styling for better visibility */
    .stSelectbox > div {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        background-color: white !important;
    }
    .stSelectbox > div:hover {
        border-color: #764ba2 !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    /* Ensure the selected value is visible */
    .stSelectbox > div > div > div > div > div {
        color: #333 !important;
        font-weight: 500 !important;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Error handling decorator
def handle_errors(func):
    """Decorator to handle errors gracefully with beautiful UI warnings"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"""
            <div class="warning-container">
                <h4>⚠️ Error in {func.__name__}</h4>
                <p><strong>Error:</strong> {str(e)}</p>
                <p><strong>Tip:</strong> Please check your data and try again.</p>
            </div>
            """
            st.markdown(error_msg, unsafe_allow_html=True)
            return None
    return wrapper

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">🏥 Advanced Healthcare Analytics Dashboard</h1>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Comprehensive Healthcare Data Analysis & Predictive Modeling</p>
</div>
""", unsafe_allow_html=True)

# Load and clean data with progress indicator
@st.cache_data
@handle_errors
def load_data():
    with st.spinner("🔄 Loading and processing healthcare data..."):
        df_original = pd.read_csv("healthcare_dataset.csv")
        df = df_original.copy()
        df.columns = df.columns.str.strip()
        
        # Convert dates and create new features
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
        df['Stay Length'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
        
        # Create additional features
        df['Month'] = df['Date of Admission'].dt.month
        df['Year'] = df['Date of Admission'].dt.year
        df['Day of Week'] = df['Date of Admission'].dt.day_name()
        df['Season'] = df['Date of Admission'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Clean data
        df_cleaned = df.dropna(subset=['Age', 'Billing Amount', 'Stay Length', 'Test Results'])
        df_cleaned.to_csv("cleaned_healthcare_dataset.csv", index=False)
        
        return df_cleaned

# Load data
df_cleaned = load_data()

if df_cleaned is None:
    st.error("❌ Failed to load data. Please check if 'healthcare_dataset.csv' exists.")
    st.stop()

# Sidebar with enhanced filters
st.sidebar.markdown("## 🔍 Advanced Filters")

# Date range filter
st.sidebar.subheader("📅 Date Range")
min_date = df_cleaned['Date of Admission'].min()
max_date = df_cleaned['Date of Admission'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Enhanced filters
st.sidebar.subheader("👥 Demographics")
gender_filter = st.sidebar.multiselect(
    "Gender",
    df_cleaned['Gender'].unique(),
    default=df_cleaned['Gender'].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df_cleaned['Age'].min()),
    max_value=int(df_cleaned['Age'].max()),
    value=(int(df_cleaned['Age'].min()), int(df_cleaned['Age'].max()))
)

st.sidebar.subheader("🏥 Medical Information")
condition_filter = st.sidebar.multiselect(
    "Medical Condition",
    df_cleaned['Medical Condition'].unique(),
    default=df_cleaned['Medical Condition'].unique()
)

test_result_filter = st.sidebar.multiselect(
    "Test Results",
    df_cleaned['Test Results'].unique(),
    default=df_cleaned['Test Results'].unique()
)

# Apply filters
filtered_df = df_cleaned[
    (df_cleaned['Gender'].isin(gender_filter)) &
    (df_cleaned['Medical Condition'].isin(condition_filter)) &
    (df_cleaned['Test Results'].isin(test_result_filter)) &
    (df_cleaned['Age'] >= age_range[0]) &
    (df_cleaned['Age'] <= age_range[1]) &
    (df_cleaned['Date of Admission'].dt.date >= date_range[0]) &
    (df_cleaned['Date of Admission'].dt.date <= date_range[1])
]

# Section selector with dropdown
st.markdown("## 📋 Dashboard Sections")
section_options = {
    "📊 Overview & Analytics": "overview",
    "📈 Advanced Visualizations": "visualizations", 
    "🤖 Machine Learning": "ml",
    "📋 Data Insights": "insights",
    "📤 Export & Reports": "export"
}

selected_section = st.selectbox(
    "Choose a section to explore:",
    list(section_options.keys()),
    index=0
)

# Section content based on dropdown selection
if selected_section == "📊 Overview & Analytics":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📊 Overview & Analytics")
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Patients</h3>
            <h2>{len(filtered_df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏥 Unique Conditions</h3>
            <h2>{filtered_df['Medical Condition'].nunique()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Avg Stay Length</h3>
            <h2>{filtered_df['Stay Length'].mean():.1f} days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Billing</h3>
            <h2>${filtered_df['Billing Amount'].mean():,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data quality metrics
    st.subheader("📋 Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_data = df_cleaned.isnull().sum().sum()
        total_data = df_cleaned.size
        completeness = ((total_data - missing_data) / total_data) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        st.metric("Unique Doctors", filtered_df['Doctor'].nunique())
    
    with col3:
        st.metric("Unique Hospitals", filtered_df['Hospital'].nunique())
    
    # Basic charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_age = px.histogram(
                filtered_df, 
                x="Age", 
                color="Gender", 
                title="Age Distribution by Gender",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Chart Error</h4>
                <p>Could not generate Age Distribution chart: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        try:
            fig_billing = px.histogram(
                filtered_df, 
                x="Billing Amount", 
                color="Medical Condition", 
                title="Billing Amount by Condition",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_billing.update_layout(height=400)
            st.plotly_chart(fig_billing, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Chart Error</h4>
                <p>Could not generate Billing Amount chart: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_section == "📈 Advanced Visualizations":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📈 Advanced Visualizations")
    
    # Time series analysis
    st.subheader("📅 Time Series Analysis")
    
    try:
        # Monthly admissions
        monthly_admissions = filtered_df.groupby(filtered_df['Date of Admission'].dt.to_period('M')).size().reset_index()
        monthly_admissions.columns = ['Month', 'Admissions']
        monthly_admissions['Month'] = monthly_admissions['Month'].astype(str)
        
        fig_time = px.line(
            monthly_admissions, 
            x='Month', 
            y='Admissions',
            title="Monthly Admission Trends",
            markers=True
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Time Series Error</h4>
            <p>Could not generate time series chart: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    try:
        numeric_cols = ['Age', 'Billing Amount', 'Stay Length']
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Correlation Error</h4>
            <p>Could not generate correlation matrix: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_box = px.box(
                filtered_df, 
                x="Gender", 
                y="Stay Length", 
                color="Gender",
                title="Stay Length Distribution by Gender",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Chart Error</h4>
                <p>Could not generate box plot: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        try:
            test_counts = filtered_df['Test Results'].value_counts()
            fig_pie = px.pie(
                values=test_counts.values,
                names=test_counts.index,
                title="Test Result Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Chart Error</h4>
                <p>Could not generate pie chart: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 3D Scatter plot
    st.subheader("🎯 3D Analysis")
    try:
        fig_3d = px.scatter_3d(
            filtered_df,
            x='Age',
            y='Billing Amount',
            z='Stay Length',
            color='Test Results',
            title="3D Relationship: Age vs Billing vs Stay Length",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ 3D Chart Error</h4>
            <p>Could not generate 3D scatter plot: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_section == "🤖 Machine Learning":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("🤖 Machine Learning & Predictive Analytics")
    
    # Check if we have enough data for ML
    if len(filtered_df) < 50:
        st.markdown("""
        <div class="warning-container">
            <h4>⚠️ Insufficient Data</h4>
            <p>Not enough data for reliable machine learning. Need at least 50 samples.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Feature selection
    st.subheader("🔍 Feature Analysis")
    
    try:
        features = ['Age', 'Billing Amount', 'Stay Length']
        X = filtered_df[features]
        le = LabelEncoder()
        y = le.fit_transform(filtered_df['Test Results'])
        
        # Feature importance analysis
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        feature_scores = pd.DataFrame({
            'Feature': features,
            'Score': selector.scores_
        }).sort_values('Score', ascending=True)
        
        fig_importance = px.bar(
            feature_scores,
            x='Score',
            y='Feature',
            orientation='h',
            title="Feature Importance Scores",
            color='Score',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=300)
        st.plotly_chart(fig_importance, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Feature Analysis Error</h4>
            <p>Could not perform feature analysis: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model training with progress and optimization
    st.subheader("🚀 Model Training")
    
    # Add training options
    training_mode = st.selectbox(
        "Training Mode:",
        ["Fast (Basic Models)", "Standard (Cross-validation)", "Comprehensive (Full Analysis)"],
        help="Choose training intensity based on your needs"
    )
    
    if st.button("🚀 Start Model Training", type="primary"):
        with st.spinner("Training models..."):
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Optimized models based on training mode
                if training_mode == "Fast (Basic Models)":
                    models = {
                        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
                    }
                    cv_folds = 3
                elif training_mode == "Standard (Cross-validation)":
                    models = {
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
                    }
                    cv_folds = 5
                else:  # Comprehensive
                    models = {
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
                    }
                    cv_folds = 10
                
                results = []
                progress_bar = st.progress(0)
                
                for i, (name, model) in enumerate(models.items()):
                    # Update progress
                    progress = (i + 1) / len(models)
                    progress_bar.progress(progress)
                    
                    # Cross-validation (only for standard and comprehensive)
                    if training_mode != "Fast (Basic Models)":
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean = 0
                        cv_std = 0
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                    except:
                        auc = 0
                    
                    results.append({
                        "name": name,
                        "model": model,
                        "accuracy": acc,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "auc": auc,
                        "preds": y_pred,
                        "report": classification_report(y_test, y_pred, output_dict=True),
                        "confusion": confusion_matrix(y_test, y_pred)
                    })
                
                progress_bar.progress(1.0)
                
                # Model comparison
                st.subheader("📊 Model Performance Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                for i, res in enumerate(results):
                    with [col1, col2, col3, col4][i]:
                        if training_mode != "Fast (Basic Models)":
                            st.metric(
                                res["name"],
                                f"{res['accuracy']:.3f}",
                                f"CV: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}"
                            )
                        else:
                            st.metric(res["name"], f"{res['accuracy']:.3f}")
                
                # Best model details
                best_model_result = max(results, key=lambda r: r["accuracy"])
                
                st.markdown(f"""
                <div class="success-container">
                    <h4>🏆 Best Model: {best_model_result['name']}</h4>
                    <p>Accuracy: {best_model_result['accuracy']:.3f}</p>
                    <p>AUC Score: {best_model_result['auc']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # ROC Curve
                if best_model_result['auc'] > 0:
                    try:
                        y_pred_proba = best_model_result['model'].predict_proba(X_test_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {best_model_result["auc"]:.3f})'))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="warning-container">
                            <h4>⚠️ ROC Curve Error</h4>
                            <p>Could not generate ROC curve: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Confusion Matrix
                try:
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        best_model_result['confusion'], 
                        annot=True, 
                        fmt='d',
                        xticklabels=le.classes_, 
                        yticklabels=le.classes_,
                        cmap='Blues'
                    )
                    plt.title("Confusion Matrix")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-container">
                        <h4>⚠️ Confusion Matrix Error</h4>
                        <p>Could not generate confusion matrix: {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save best model
                try:
                    joblib.dump(best_model_result['model'], "best_model_rf.joblib")
                    joblib.dump(scaler, "scaler.joblib")
                    joblib.dump(le, "label_encoder.joblib")
                    
                    st.markdown("""
                    <div class="success-container">
                        <h4>✅ Model Saved Successfully</h4>
                        <p>Best model and preprocessing objects have been saved.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-container">
                        <h4>⚠️ Model Save Error</h4>
                        <p>Could not save model: {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="warning-container">
                    <h4>⚠️ Training Error</h4>
                    <p>An error occurred during model training: {str(e)}</p>
                    <p><strong>Possible solutions:</strong></p>
                    <ul>
                        <li>Check if you have enough data (at least 50 samples)</li>
                        <li>Ensure all required features are present</li>
                        <li>Try a different training mode</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_section == "📋 Data Insights":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📋 Data Insights & Statistics")
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    try:
        numeric_summary = filtered_df[['Age', 'Billing Amount', 'Stay Length']].describe()
        st.dataframe(numeric_summary, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Statistical Summary Error</h4>
            <p>Could not generate statistical summary: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top conditions analysis
    st.subheader("🏥 Top Medical Conditions")
    
    try:
        condition_analysis = filtered_df.groupby('Medical Condition').agg({
            'Age': 'mean',
            'Billing Amount': 'mean',
            'Stay Length': 'mean',
            'Test Results': 'count'
        }).round(2)
        condition_analysis.columns = ['Avg Age', 'Avg Billing', 'Avg Stay Length', 'Patient Count']
        condition_analysis = condition_analysis.sort_values('Patient Count', ascending=False)
        
        st.dataframe(condition_analysis, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Condition Analysis Error</h4>
            <p>Could not generate condition analysis: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Seasonal analysis
    st.subheader("🌤️ Seasonal Analysis")
    
    try:
        seasonal_data = filtered_df.groupby('Season').agg({
            'Billing Amount': 'mean',
            'Stay Length': 'mean',
            'Test Results': 'count'
        }).round(2)
        seasonal_data.columns = ['Avg Billing', 'Avg Stay Length', 'Patient Count']
        
        fig_seasonal = px.bar(
            seasonal_data,
            title="Seasonal Patient Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    except Exception as e:
        st.markdown(f"""
        <div class="warning-container">
            <h4>⚠️ Seasonal Analysis Error</h4>
            <p>Could not generate seasonal analysis: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_section == "📤 Export & Reports":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📤 Export & Reports")
    
    # Data export options
    st.subheader("📁 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Export filtered data
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📊 Download Filtered Dataset (CSV)",
                csv_data,
                file_name=f"healthcare_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Export Error</h4>
                <p>Could not export filtered data: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            # Export statistics
            stats_data = filtered_df.describe().to_csv()
            st.download_button(
                "📈 Download Statistics Report (CSV)",
                stats_data,
                file_name=f"healthcare_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Statistics Export Error</h4>
                <p>Could not export statistics: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        try:
            # Export model files
            with open("best_model_rf.joblib", "rb") as f:
                st.download_button(
                    "🤖 Download Best Model",
                    f.read(),
                    file_name="best_model_rf.joblib",
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.markdown("""
            <div class="info-container">
                <h4>ℹ️ No Model Available</h4>
                <p>Train a model first to download it.</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Model Export Error</h4>
                <p>Could not export model: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            with open("scaler.joblib", "rb") as f:
                st.download_button(
                    "⚖️ Download Scaler",
                    f.read(),
                    file_name="scaler.joblib",
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.markdown("""
            <div class="info-container">
                <h4>ℹ️ No Scaler Available</h4>
                <p>Train a model first to download the scaler.</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-container">
                <h4>⚠️ Scaler Export Error</h4>
                <p>Could not export scaler: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate report
    st.subheader("📋 Generate Report")
    
    if st.button("📄 Generate Comprehensive Report"):
        with st.spinner("Generating report..."):
            try:
                report = f"""
# Healthcare Analytics Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total Patients: {len(filtered_df):,}
- Date Range: {filtered_df['Date of Admission'].min().strftime('%Y-%m-%d')} to {filtered_df['Date of Admission'].max().strftime('%Y-%m-%d')}
- Unique Conditions: {filtered_df['Medical Condition'].nunique()}
- Average Stay Length: {filtered_df['Stay Length'].mean():.2f} days
- Average Billing Amount: ${filtered_df['Billing Amount'].mean():,.2f}

## Key Insights
- Most Common Condition: {filtered_df['Medical Condition'].mode().iloc[0] if len(filtered_df['Medical Condition'].mode()) > 0 else 'N/A'}
- Gender Distribution: {filtered_df['Gender'].value_counts().to_dict()}
- Test Results Distribution: {filtered_df['Test Results'].value_counts().to_dict()}

## Data Quality
- Data Completeness: {((filtered_df.size - filtered_df.isnull().sum().sum()) / filtered_df.size * 100):.1f}%
- Unique Doctors: {filtered_df['Doctor'].nunique()}
- Unique Hospitals: {filtered_df['Hospital'].nunique()}
                """
                
                st.download_button(
                    "📄 Download Report (TXT)",
                    report,
                    file_name=f"healthcare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.markdown("""
                <div class="success-container">
                    <h4>✅ Report Generated Successfully</h4>
                    <p>Your comprehensive healthcare analytics report is ready for download.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="warning-container">
                    <h4>⚠️ Report Generation Error</h4>
                    <p>Could not generate report: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 Advanced Healthcare Analytics Dashboard | Built with Streamlit</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True) 