import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_chat import message
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "o3"  # "o3-mini" might have been renamed

# Page configuration
st.set_page_config(
    page_title="AnalyticQ",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'mode' not in st.session_state:
    st.session_state.mode = "Professional"
if 'llm_choice' not in st.session_state:
    st.session_state.llm_choice = "o3"
if 'api_processing' not in st.session_state:
    st.session_state.api_processing = False
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0
if 'last_error' not in st.session_state:
    st.session_state.last_error = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Helper function for OpenAI queries
def query_openai(prompt, system_prompt="You are a helpful data analysis assistant.", max_tokens=500):
    try:
        st.session_state['api_processing'] = True
        
        # Log the request details for debugging
        print(f"Sending request to OpenAI API with model: {OPENAI_MODEL}")
        
        # Trim prompt if too long (rough token limit)
        if len(prompt) > 4000:
            print(f"Prompt too long ({len(prompt)} chars), trimming...")
            prompt = prompt[:4000] + "... [trimmed for length]"
        
        # Try with model fallbacks if needed
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
        except Exception as model_error:
            print(f"Error with primary model {OPENAI_MODEL}: {str(model_error)}")
            # Try with a fallback model
            fallback_model = "gpt-3.5-turbo"
            print(f"Attempting with fallback model: {fallback_model}")
            try:
                response = openai.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens  # Note: using max_tokens for older models
                )
            except Exception as fallback_error:
                print(f"Fallback model also failed: {str(fallback_error)}")
                # Last resort - use text-ada-001 with simplified prompt
                last_resort_model = "text-ada-001"
                print(f"Attempting with basic model: {last_resort_model}")
                simplified_prompt = f"USER: {prompt[-1000:]}"  # Just use the last part of the query
                response = openai.completions.create(
                    model=last_resort_model,
                    prompt=simplified_prompt,
                    max_tokens=150
                )
                # Format the response to match the expected structure
                return response.choices[0].text.strip()
        
        st.session_state['api_processing'] = False
        return response.choices[0].message.content
    except Exception as e:
        st.session_state['api_processing'] = False
        error_msg = f"Error querying OpenAI: {str(e)}"
        print(f"OpenAI API error: {str(e)}")
        # Return a more user-friendly message
        return "I'm having trouble connecting to my AI services right now. Please try again in a moment."

# Sidebar for configuration and navigation
with st.sidebar:
    st.title("AnalyticQ")
    st.subheader("AI-Powered Data Analysis")
    
    # Simplified mode selection - just two options
    st.session_state.mode = st.radio("Select User Mode:", ["Professional", "Simplified"])
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")

if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")
        
        # Data info
        if st.session_state.df is not None:
            st.write(f"Rows: {st.session_state.df.shape[0]}, Columns: {st.session_state.df.shape[1]}")
            
            if st.button("Show Column Details"):
                col_info = pd.DataFrame({
                    'Column': st.session_state.df.columns,
                    'Type': st.session_state.df.dtypes,
                    'Non-Null': st.session_state.df.notnull().sum().values
                })
                st.dataframe(col_info)
    
    # Advanced settings in expander
    with st.expander("Advanced Settings", expanded=False):
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        
        if st.session_state.debug_mode:
            st.info("Debug mode enabled. Errors will be displayed with more detail.")
            
            # Model selection
            st.session_state.llm_choice = st.selectbox(
                "LLM Model",
                ["o3", "gpt-3.5-turbo", "text-ada-001"],
                index=0
            )
            
            # Reset options
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
                
            if st.button("Reset API State"):
                st.session_state.api_processing = False
                st.session_state.retry_count = 0
                st.session_state.last_error = None
                st.rerun()

# Main content area
if st.session_state.df is None:
    # Landing page when no data is loaded
    st.title("Welcome to AnalyticQ")
    st.write("Upload your data through the sidebar to begin analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### Professional Mode\nAdvanced analytics for data professionals with comprehensive tools and detailed insights.")
    with col2:
        st.info("### Simplified Mode\nEasy-to-use interface for quick analysis with natural language queries and simplified visualizations.")
        
    st.warning("Please upload a CSV file in the sidebar to get started.")
else:
    # Main tabs for navigation
    tab1, tab2 = st.tabs(["Dashboard", "Chat Assistant"])
    
    # Dashboard Tab
    with tab1:
        st.title(f"{st.session_state.mode} Dashboard")
    st.write("### Data Preview")
        st.dataframe(st.session_state.df.head())
        
        # Four card layout for analysis options using containers instead of st.card
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        # Card 1: EDA
        with col1:
            with st.expander("📊 Exploratory Data Analysis", expanded=True):
                st.markdown("**Analyze the statistical properties of your dataset.**")
                eda_option = st.selectbox(
                    "Select EDA Analysis:",
                    ["Summary Statistics", "Missing Values", "Outlier Detection", "Distribution Analysis"]
                )
                
                if eda_option == "Summary Statistics":
        st.write("### Summary Statistics")
                    st.dataframe(st.session_state.df.describe())

                elif eda_option == "Missing Values":
        st.write("### Missing Values")
                    missing = st.session_state.df.isnull().sum().reset_index()
                    missing.columns = ['Column', 'Missing Count']
                    missing['Missing Percentage'] = (missing['Missing Count'] / len(st.session_state.df)) * 100
                    st.dataframe(missing)
                    
                    fig = px.bar(missing, x='Column', y='Missing Percentage', 
                                title='Missing Values by Column (%)',
                                labels={'Missing Percentage': '% Missing'})
                    st.plotly_chart(fig)
                    
                elif eda_option == "Outlier Detection":
                    st.write("### Outlier Detection")
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select column for outlier detection:", numeric_cols)
                        
                        q1 = st.session_state.df[selected_col].quantile(0.25)
                        q3 = st.session_state.df[selected_col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        outliers = st.session_state.df[(st.session_state.df[selected_col] < lower_bound) | 
                                                    (st.session_state.df[selected_col] > upper_bound)]
                        
                        st.write(f"Potential outliers: {len(outliers)} rows")
                        fig = px.box(st.session_state.df, y=selected_col, title=f'Box Plot: {selected_col}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("No numeric columns found for outlier detection.")
                    
                elif eda_option == "Distribution Analysis":
                    st.write("### Distribution Analysis")
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
                        
                        fig = px.histogram(st.session_state.df, x=selected_col, marginal="box", 
                                        title=f'Distribution of {selected_col}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("No numeric columns found for distribution analysis.")
        
        # Card 2: Data Visualization
        with col2:
            with st.expander("📈 Data Visualization", expanded=True):
                st.markdown("**Create insightful visualizations from your data.**")
                viz_option = st.selectbox(
                    "Select Visualization Type:",
                    ["Correlation Heatmap", "Scatter Plot", "Line Chart", "Bar Chart", "Pie Chart"]
                )
                
                if viz_option == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
                    numeric_df = st.session_state.df.select_dtypes(include=np.number)
                    if not numeric_df.empty and numeric_df.shape[1] > 1:
                        fig = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto",
                                    color_continuous_scale='RdBu_r')
                        fig.update_layout(height=600)
                        st.plotly_chart(fig)
                    else:
                        st.warning("Not enough numeric columns for correlation analysis.")
                    
                elif viz_option == "Scatter Plot":
                    st.write("### Scatter Plot")
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("Select X-axis:", numeric_cols, index=0)
                        y_col = st.selectbox("Select Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
                        color_col = st.selectbox("Color by (optional):", ['None'] + st.session_state.df.columns.tolist())
                        
                        if color_col == 'None':
                            fig = px.scatter(st.session_state.df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
                        else:
                            fig = px.scatter(st.session_state.df, x=x_col, y=y_col, color=color_col, 
                                            title=f'{y_col} vs {x_col}, colored by {color_col}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("Need at least 2 numeric columns for scatter plot.")
                    
                elif viz_option == "Line Chart":
                    st.write("### Line Chart")
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        y_col = st.selectbox("Select Y-axis:", numeric_cols)
                        
                        # Try to find a date column
                        date_cols = [col for col in st.session_state.df.columns if 
                                    any(date_str in col.lower() for date_str in ['date', 'time', 'day', 'year'])]
                        
                        if date_cols:
                            x_col = st.selectbox("Select X-axis (time):", date_cols + [col for col in st.session_state.df.columns if col not in date_cols])
                        else:
                            x_col = st.selectbox("Select X-axis:", st.session_state.df.columns)
                        
                        fig = px.line(st.session_state.df, x=x_col, y=y_col, title=f'{y_col} over {x_col}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("No numeric columns found for Y-axis.")
                    
                elif viz_option == "Bar Chart":
                    st.write("### Bar Chart")
                    y_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if y_cols:
                        y_col = st.selectbox("Select Value Column:", y_cols)
                        categorical_cols = [col for col in st.session_state.df.columns if st.session_state.df[col].nunique() < 20]
                        if categorical_cols:
                            x_col = st.selectbox("Select Category Column:", categorical_cols)
                            fig = px.bar(st.session_state.df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
                            st.plotly_chart(fig)
                        else:
                            st.write("No suitable categorical columns found for bar chart.")
                    else:
                        st.warning("No numeric columns found for values.")
                        
                elif viz_option == "Pie Chart":
                    st.write("### Pie Chart")
                    categorical_cols = [col for col in st.session_state.df.columns if st.session_state.df[col].nunique() < 15]
                    if categorical_cols:
                        cat_col = st.selectbox("Select Category Column:", categorical_cols)
                        value_counts = st.session_state.df[cat_col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Distribution of {cat_col}')
                        st.plotly_chart(fig)
                    else:
                        st.write("No suitable categorical columns found with fewer than 15 categories.")
        
        # Card 3: Data Cleaning
        with col3:
            with st.expander("🧹 Data Cleaning", expanded=True):
                st.markdown("**Clean and preprocess your dataset.**")
                cleaning_option = st.selectbox(
                    "Select Cleaning Task:",
                    ["Handle Missing Values", "Remove Duplicates", "Fix Data Types", "Filter Data"]
                )
                
                if cleaning_option == "Handle Missing Values":
                    st.write("### Handle Missing Values")
                    col_to_clean = st.selectbox("Select column to handle missing values:", 
                                            st.session_state.df.columns)
                    
                    missing_count = st.session_state.df[col_to_clean].isnull().sum()
                    st.write(f"Missing values in {col_to_clean}: {missing_count}")
                    
                    if missing_count > 0:
                        strategy = st.selectbox("Select strategy:", 
                                            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with constant"])
                        
                        if st.button("Apply Cleaning"):
                            if strategy == "Drop rows":
                                st.session_state.df = st.session_state.df.dropna(subset=[col_to_clean])
                            elif strategy == "Fill with mean" and pd.api.types.is_numeric_dtype(st.session_state.df[col_to_clean]):
                                st.session_state.df[col_to_clean] = st.session_state.df[col_to_clean].fillna(st.session_state.df[col_to_clean].mean())
                            elif strategy == "Fill with median" and pd.api.types.is_numeric_dtype(st.session_state.df[col_to_clean]):
                                st.session_state.df[col_to_clean] = st.session_state.df[col_to_clean].fillna(st.session_state.df[col_to_clean].median())
                            elif strategy == "Fill with mode":
                                st.session_state.df[col_to_clean] = st.session_state.df[col_to_clean].fillna(st.session_state.df[col_to_clean].mode()[0])
                            elif strategy == "Fill with constant":
                                fill_value = st.text_input("Enter fill value:")
                                if fill_value:
                                    try:
                                        if pd.api.types.is_numeric_dtype(st.session_state.df[col_to_clean]):
                                            st.session_state.df[col_to_clean] = st.session_state.df[col_to_clean].fillna(float(fill_value))
                                        else:
                                            st.session_state.df[col_to_clean] = st.session_state.df[col_to_clean].fillna(fill_value)
                                    except ValueError:
                                        st.error("Invalid fill value for this column type")
                            
                            st.success(f"Applied {strategy} to {col_to_clean}")
                            st.write("Updated Data Preview:")
                            st.dataframe(st.session_state.df.head())
                    else:
                        st.write("No missing values in this column.")
                        
                elif cleaning_option == "Remove Duplicates":
                    st.write("### Remove Duplicates")
                    dup_count = st.session_state.df.duplicated().sum()
                    st.write(f"Duplicate rows: {dup_count}")
                    
                    if dup_count > 0:
                        if st.button("Remove Duplicates"):
                            original_count = len(st.session_state.df)
                            st.session_state.df = st.session_state.df.drop_duplicates()
                            st.success(f"Removed {original_count - len(st.session_state.df)} duplicate rows")
                            
                            st.write("Updated Data Preview:")
                            st.dataframe(st.session_state.df.head())
                    else:
                        st.write("No duplicate rows found in the dataset.")
                        
                elif cleaning_option == "Fix Data Types":
                    st.write("### Fix Data Types")
                    col_to_convert = st.selectbox("Select column to convert:", st.session_state.df.columns)
                    current_type = st.session_state.df[col_to_convert].dtype
                    st.write(f"Current data type: {current_type}")
                    
                    target_type = st.selectbox("Convert to:", ["int", "float", "string", "datetime", "category"])
                    
                    if st.button("Convert Data Type"):
                        try:
                            if target_type == "int":
                                st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(int)
                            elif target_type == "float":
                                st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(float)
                            elif target_type == "string":
                                st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype(str)
                            elif target_type == "datetime":
                                st.session_state.df[col_to_convert] = pd.to_datetime(st.session_state.df[col_to_convert])
                            elif target_type == "category":
                                st.session_state.df[col_to_convert] = st.session_state.df[col_to_convert].astype('category')
                            
                            st.success(f"Converted {col_to_convert} to {target_type}")
                            st.write(f"New data type: {st.session_state.df[col_to_convert].dtype}")
                            st.write("Updated Data Preview:")
                            st.dataframe(st.session_state.df.head())
                        except Exception as e:
                            st.error(f"Error converting data type: {str(e)}")
                            
                elif cleaning_option == "Filter Data":
                    st.write("### Filter Data")
                    filter_col = st.selectbox("Select column to filter on:", st.session_state.df.columns)
                    
                    if pd.api.types.is_numeric_dtype(st.session_state.df[filter_col]):
                        min_val, max_val = st.session_state.df[filter_col].min(), st.session_state.df[filter_col].max()
                        filter_range = st.slider(f"Filter range for {filter_col}:", 
                                            float(min_val), float(max_val), 
                                            (float(min_val), float(max_val)))
                        
                        if st.button("Apply Filter"):
                            filtered_df = st.session_state.df[(st.session_state.df[filter_col] >= filter_range[0]) & 
                                                            (st.session_state.df[filter_col] <= filter_range[1])]
                            st.write(f"Filtered data ({len(filtered_df)} rows):")
                            st.dataframe(filtered_df)
                            
                            if st.button("Commit Filter"):
                                st.session_state.df = filtered_df
                                st.success(f"Dataset updated to {len(filtered_df)} rows")
                    else:
                        unique_vals = st.session_state.df[filter_col].unique()
                        selected_vals = st.multiselect(f"Select values to keep for {filter_col}:", unique_vals, default=list(unique_vals))
                        
                        if st.button("Apply Filter"):
                            filtered_df = st.session_state.df[st.session_state.df[filter_col].isin(selected_vals)]
                            st.write(f"Filtered data ({len(filtered_df)} rows):")
                            st.dataframe(filtered_df)
                            
                            if st.button("Commit Filter"):
                                st.session_state.df = filtered_df
                                st.success(f"Dataset updated to {len(filtered_df)} rows")
        
        # Card 4: Trend Analysis
        with col4:
            with st.expander("📈 Trend Analysis", expanded=True):
                st.markdown("**Analyze patterns and trends in your time series data.**")
                trend_option = st.selectbox(
                    "Select Trend Analysis:",
                    ["Time Series Decomposition", "Moving Averages", "Growth Rates", "Seasonality Detection"]
                )
                
                # Check if there's a potential date column
                date_cols = [col for col in st.session_state.df.columns if 
                            any(date_str in col.lower() for date_str in ['date', 'time', 'day', 'year'])]
                
                if not date_cols:
                    st.warning("No date/time columns detected. Trend analysis works best with time series data.")
                    
                if trend_option == "Time Series Decomposition":
                    st.write("### Time Series Decomposition")
                    if date_cols:
                        date_col = st.selectbox("Select date column:", date_cols)
                    else:
                        date_col = st.selectbox("Select date column:", st.session_state.df.columns)
                    
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        value_col = st.selectbox("Select value column:", numeric_cols)
                        
                        if st.button("Analyze Trend"):
                            try:
                                # Convert to datetime if not already
                                if not pd.api.types.is_datetime64_dtype(st.session_state.df[date_col]):
                                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col])
                                
                                # Sort by date
                                temp_df = st.session_state.df.sort_values(by=date_col)
                                temp_df = temp_df.set_index(date_col)
                                
                                # Simple trend visualization
                                fig = px.line(temp_df, y=value_col, title=f'Trend of {value_col} over time')
                                st.plotly_chart(fig)
                                
                                # Check if we have enough data points for rolling calculations
                                if len(temp_df) > 10:
                                    # Add trend line (rolling mean)
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df[value_col], mode='lines', name=value_col))
                                    fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df[value_col].rolling(window=min(7, len(temp_df)//3)).mean(), 
                                                        mode='lines', name='Trend (Rolling Mean)', line=dict(color='red')))
                                    fig.update_layout(title=f'Trend Analysis of {value_col}')
                                    st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error in trend analysis: {str(e)}")
                    else:
                        st.warning("No numeric columns found for trend analysis.")
                        
                elif trend_option == "Moving Averages":
                    st.write("### Moving Averages")
                    if date_cols:
                        date_col = st.selectbox("Select date column:", date_cols)
                    else:
                        date_col = st.selectbox("Select date column:", st.session_state.df.columns)
                    
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        value_col = st.selectbox("Select value column:", numeric_cols)
                        
                        window_size = st.slider("Window size for moving average:", 2, 
                                            min(30, max(2, len(st.session_state.df) // 2)), 7)
                        
                        if st.button("Calculate Moving Averages"):
                            try:
                                # Convert to datetime if not already
                                if not pd.api.types.is_datetime64_dtype(st.session_state.df[date_col]):
                                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col])
                                
                                # Sort by date
                                temp_df = st.session_state.df.sort_values(by=date_col)
                                
                                # Calculate moving averages
                                temp_df['SMA'] = temp_df[value_col].rolling(window=window_size).mean()
                                temp_df['EMA'] = temp_df[value_col].ewm(span=window_size).mean()
                                
                                # Plot
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df[value_col], mode='lines', name=value_col))
                                fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df['SMA'], mode='lines', 
                                                    name=f'Simple MA ({window_size})', line=dict(color='red')))
                                fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df['EMA'], mode='lines', 
                                                    name=f'Exponential MA ({window_size})', line=dict(color='green')))
                                
                                fig.update_layout(title=f'Moving Averages of {value_col}')
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error calculating moving averages: {str(e)}")
                    else:
                        st.warning("No numeric columns found for moving averages.")
                            
                elif trend_option == "Growth Rates":
                    st.write("### Growth Rates")
                    if date_cols:
                        date_col = st.selectbox("Select date column:", date_cols)
                    else:
                        date_col = st.selectbox("Select date column:", st.session_state.df.columns)
                    
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        value_col = st.selectbox("Select value column:", numeric_cols)
                        
                        if st.button("Calculate Growth Rates"):
                            try:
                                # Convert to datetime if not already
                                if not pd.api.types.is_datetime64_dtype(st.session_state.df[date_col]):
                                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col])
                                
                                # Sort by date
                                temp_df = st.session_state.df.sort_values(by=date_col)
                                
                                # Calculate period-over-period growth
                                temp_df['Growth_Rate'] = temp_df[value_col].pct_change() * 100
                                
                                # Plot
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=temp_df[date_col], y=temp_df['Growth_Rate'], 
                                                    mode='lines+markers', name='Growth Rate (%)'))
                                
                                fig.update_layout(title=f'Growth Rate of {value_col}', yaxis_title='Growth Rate (%)')
                                st.plotly_chart(fig)
                                
                                # Summary statistics for growth rate
                                growth_stats = pd.DataFrame({
                                    'Statistic': ['Mean Growth (%)', 'Median Growth (%)', 'Min Growth (%)', 'Max Growth (%)'],
                                    'Value': [
                                        round(temp_df['Growth_Rate'].mean(), 2),
                                        round(temp_df['Growth_Rate'].median(), 2),
                                        round(temp_df['Growth_Rate'].min(), 2),
                                        round(temp_df['Growth_Rate'].max(), 2)
                                    ]
                                })
                                st.dataframe(growth_stats)
                            except Exception as e:
                                st.error(f"Error calculating growth rates: {str(e)}")
                    else:
                        st.warning("No numeric columns found for growth rate analysis.")
                            
                elif trend_option == "Seasonality Detection":
                    st.write("### Seasonality Detection")
                    if date_cols:
                        date_col = st.selectbox("Select date column:", date_cols)
                    else:
                        date_col = st.selectbox("Select date column:", st.session_state.df.columns)
                    
                    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    if numeric_cols:
                        value_col = st.selectbox("Select value column:", numeric_cols)
                        
                        if st.button("Detect Seasonality"):
                            try:
                                # Convert to datetime if not already
                                if not pd.api.types.is_datetime64_dtype(st.session_state.df[date_col]):
                                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col])
                                
                                # Sort by date and extract components
                                temp_df = st.session_state.df.sort_values(by=date_col).copy()
                                temp_df['Year'] = temp_df[date_col].dt.year
                                temp_df['Month'] = temp_df[date_col].dt.month
                                temp_df['Quarter'] = temp_df[date_col].dt.quarter
                                temp_df['Day_of_Week'] = temp_df[date_col].dt.dayofweek
                                
                                # Monthly seasonality
                                if temp_df['Month'].nunique() > 1:
                                    monthly_avg = temp_df.groupby('Month')[value_col].mean().reset_index()
                                    fig = px.bar(monthly_avg, x='Month', y=value_col, 
                                                title=f'Monthly Seasonality of {value_col}')
                                    st.plotly_chart(fig)
                                
                                # Quarterly seasonality
                                if temp_df['Quarter'].nunique() > 1:
                                    quarterly_avg = temp_df.groupby('Quarter')[value_col].mean().reset_index()
                                    fig = px.bar(quarterly_avg, x='Quarter', y=value_col, 
                                                title=f'Quarterly Seasonality of {value_col}')
                                    st.plotly_chart(fig)
                                
                                # Day of week seasonality
                                if temp_df['Day_of_Week'].nunique() > 1:
                                    dow_avg = temp_df.groupby('Day_of_Week')[value_col].mean().reset_index()
                                    dow_avg['Day_Name'] = dow_avg['Day_of_Week'].map({
                                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                                    })
                                    fig = px.bar(dow_avg, x='Day_Name', y=value_col, 
                                                title=f'Day of Week Seasonality of {value_col}')
                                    st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error detecting seasonality: {str(e)}")
                    else:
                        st.warning("No numeric columns found for seasonality detection.")
    
    # Chat Assistant Tab
    with tab2:
        st.title("AI Data Assistant")
        st.write(f"Chat with your data using {st.session_state.llm_choice}")
        
        # Display any previous API errors
        if st.session_state.last_error and st.session_state.debug_mode:
            st.error(f"API Error: {st.session_state.last_error}")
            if st.button("Clear Error"):
                st.session_state.last_error = None
                st.rerun()
        
        # Chat interface
        user_input = st.chat_input("Ask questions about your data...", disabled=st.session_state.api_processing)
        
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                message(chat["content"], is_user=True, key=f"user_msg_{i}")
            else:
                message(chat["content"], is_user=False, key=f"ai_msg_{i}")
        
        # Show loading indicator when processing API request
        if st.session_state.api_processing:
            with st.spinner("AI is thinking..."):
                st.info("Generating response, please wait...")
                
                if st.session_state.debug_mode and st.button("Cancel Request"):
                    st.session_state.api_processing = False
                    st.session_state.retry_count = 0
                    st.rerun()
        
        # Process user input
        if user_input and not st.session_state.api_processing:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate simplified response for testing if needed
            if st.session_state.df is not None:
                try:
                    # Create a simpler prompt with just the essential information
                    columns_info = ", ".join(st.session_state.df.columns.tolist())
                    
                    # In debug mode, allow viewing the generated prompt
                    if st.session_state.debug_mode:
                        with st.expander("View prompt being sent to API", expanded=False):
                            st.code(f"Model: {st.session_state.llm_choice}\nQuestion: {user_input}\nColumns: {columns_info}")
                    
                    # Simple check for basic commands to have fallback answers
                    if "describe" in user_input.lower() or "what is" in user_input.lower():
                        data_description = f"The dataset has {st.session_state.df.shape[0]} rows and {st.session_state.df.shape[1]} columns: {columns_info}. User asked: {user_input}"
                    else:
                        # Keep description short to avoid token limits
                        data_description = f"""
                        Dataset: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns.
                        Columns: {columns_info}
                        User question: {user_input}
                        """
                    
                    # Adjust prompt complexity based on mode
                    system_prompt = "You are a helpful data analysis assistant."
                    if st.session_state.mode == "Professional":
                        system_prompt += " Technical mode."
                    else:
                        system_prompt += " Simple mode."
                    
                    response = query_openai(data_description, system_prompt=system_prompt, max_tokens=250)
                    
                    # If response is an error message and we haven't retried too many times
                    if "Error querying OpenAI" in response and st.session_state.retry_count < 2:
                        st.session_state.retry_count += 1
                        st.session_state.last_error = response
                        
                        # Try with a very simple prompt as fallback
                        simplified_prompt = f"Analyze this: Columns: {columns_info}. Question: {user_input}"
                        response = query_openai(simplified_prompt, max_tokens=150)
                    else:
                        st.session_state.retry_count = 0
                        
                except Exception as e:
                    error_message = str(e)
                    st.session_state.last_error = error_message
                    
                    if st.session_state.debug_mode:
                        response = f"Debug Error: {error_message}"
                    else:
                        response = f"I'm having trouble analyzing your data right now. Please try a simpler question."
                    
                    print(f"Error in chat processing: {error_message}")
            else:
                response = "Please upload a dataset first so I can help analyze it."
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()

# Download button for processed data
if st.session_state.df is not None:
    st.sidebar.download_button(
        label="Download Processed Data",
        data=st.session_state.df.to_csv(index=False).encode('utf-8'),
        file_name="processed_data.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("AnalyticQ - AI-Powered Data Analysis")
st.sidebar.caption("© 2023 AnalyticQ")
