# Credit Card Fraud Detection System

A real-time credit card fraud detection system built with Streamlit and machine learning. This application uses an ensemble of machine learning models to detect fraudulent transactions with high accuracy.

## Features

- Real-time transaction analysis
- Multiple input methods (Manual, CSV upload, Text paste)
- Interactive visualizations
- PDF report generation
- Multiple theme options
- Transaction history logging
- Advanced feature engineering

## Technologies Used

- Python 3.12
- Streamlit
- XGBoost
- LightGBM
- CatBoost
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install libomp (required for CatBoost on macOS):
```bash
brew install libomp
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Choose your preferred input method:
   - Manual Input: Enter transaction details manually
   - CSV Upload: Upload a CSV file with transaction data
   - Text Paste: Paste transaction data in text format

2. The system will analyze the transaction and provide:
   - Fraud probability score
   - Visual analysis
   - Detailed insights
   - PDF report (optional)

## Deployment

This application is deployed on Streamlit Cloud. You can access it at: (https://fraud-detector-2486.streamlit.app/)

## License

MIT License 
