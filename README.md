# AnalyticQ - AI-Powered Data Analysis

AnalyticQ is a comprehensive data analysis platform that combines powerful analytics tools with intuitive AI assistance to make data exploration accessible to both professionals and non-technical users.

## Features

- **Dual User Modes**:
  - **Professional Mode**: Advanced analytics tools for data professionals
  - **Simplified Mode**: User-friendly interface for non-technical users

- **Interactive Dashboard**:
  - **Exploratory Data Analysis**: Summary statistics, missing value analysis, outlier detection, and distribution analysis
  - **Data Visualization**: Create various plots including correlation heatmaps, scatter plots, line charts, and more
  - **Data Cleaning**: Tools for handling missing values, removing duplicates, fixing data types, and filtering
  - **Trend Analysis**: Time series decomposition, moving averages, growth rates, and seasonality detection

- **AI Chat Assistant**:
  - Ask questions about your data in natural language
  - Powered by OpenAI's o3-mini model
  - Get instant insights without coding

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/AnalytiQ-AI/backend.git
   cd analyticq
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a .env file to store the api key as below
   ```
   OPENAI_API_KEY= your_OPEN_AI_key
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501`

## Usage

1. Upload your CSV data file using the sidebar
2. Select your preferred mode (Professional or Simplified)
3. Navigate between the Dashboard and Chat Assistant tabs
4. Use the Dashboard to perform various analyses on your data
5. Use the Chat Assistant to ask questions about your data in natural language

## OpenAI Integration

The chat assistant is powered by OpenAI's o3-mini model for efficient and cost-effective data analysis. The integration provides:

- Context-aware responses based on your dataset
- Mode-specific responses (technical for professionals, simplified for non-experts)
- Data-informed insights without manual analysis

## Future Development

- Integration with more data sources (databases, APIs, etc.)
- Advanced ML model building and deployment
- Customizable dashboards and reporting
- User authentication and data privacy features
- Collaborative features for team analysis

## License

MIT License 
