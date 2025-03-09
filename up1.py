# Import required libraries
import pandas as pd
from scipy.stats import zscore
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert 'Anomaly_Flag' to binary (1 for Normal, 0 for Anomaly)
    df['Anomaly_Flag'] = df['Anomaly_Flag'].apply(lambda x: 1 if x == 'Normal' else 0)

    return df

# Step 2: Statistical anomaly detection
def detect_anomalies(df):
    # Calculate Z-scores for numerical columns
    numerical_columns = ['Speed_mps', 'Heading_deg', 'Battery_Level_Percent', 'Signal_Strength_Percent']
    z_threshold = 2
    for col in numerical_columns:
        df[f'{col}_Z_score'] = zscore(df[col])
        df[f'{col}_Z_Anomaly'] = df[f'{col}_Z_score'].apply(lambda x: 1 if abs(x) > z_threshold else 0)

    # Calculate moving averages and standard deviations
    window = 10
    ma_threshold = 1.5
    for col in numerical_columns:
        df[f'{col}_Moving_Avg'] = df[col].rolling(window=window).mean()
        df[f'{col}_Moving_Std'] = df[col].rolling(window=window).std()
        df[f'{col}_MA_Anomaly'] = (abs(df[col] - df[f'{col}_Moving_Avg']) > ma_threshold * df[f'{col}_Moving_Std']).astype(int)

    return df

# Step 3: Real-time visualization dashboard
def create_dashboard(df):
    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define the layout of the dashboard
    app.layout = html.Div([
        html.H1("Real-Time Anomaly Detection Dashboard", style={'textAlign': 'center'}),
        html.Label("Select a metric to visualize:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Speed (m/s)', 'value': 'Speed_mps'},
                {'label': 'Heading (deg)', 'value': 'Heading_deg'},
                {'label': 'Battery Level (%)', 'value': 'Battery_Level_Percent'},
                {'label': 'Signal Strength (%)', 'value': 'Signal_Strength_Percent'}
            ],
            value='Speed_mps',
            multi=False
        ),
        dcc.Graph(id='live-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # Update every 1 second
            n_intervals=0
        )
    ])

    # Define callback to update the graph in real-time
    @app.callback(
        Output('live-graph', 'figure'),
        [Input('interval-component', 'n_intervals'),
         Input('metric-dropdown', 'value')]
    )
    def update_graph(n, selected_metric):
        # Simulate real-time data updates (replace with actual data source)
        updated_df = df.copy()  # Replace with real-time data fetching logic

        # Create the line graph with anomaly markers
        fig = px.line(updated_df, x='Timestamp', y=selected_metric, title=f'{selected_metric} Over Time')
        fig.add_scatter(
            x=updated_df[updated_df[f'{selected_metric}_Z_Anomaly'] == 1]['Timestamp'],
            y=updated_df[updated_df[f'{selected_metric}_Z_Anomaly'] == 1][selected_metric],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Z-Score Anomaly'
        )
        fig.add_scatter(
            x=updated_df[updated_df[f'{selected_metric}_MA_Anomaly'] == 1]['Timestamp'],
            y=updated_df[updated_df[f'{selected_metric}_MA_Anomaly'] == 1][selected_metric],
            mode='markers',
            marker=dict(color='orange', size=10),
            name='Moving Average Anomaly'
        )
        return fig

    return app

# Main function to run the entire process
def main():
    # Step 1: Load and preprocess the data
    file_path = 'UP-1_Anomaly_Detection_SynData.csv'  # Replace with your file path
    df = load_and_preprocess_data(file_path)

    # Step 2: Detect anomalies
    df = detect_anomalies(df)

    # Step 3: Evaluate and print anomaly detection performance(optional)
    # Combine Z-score and moving average anomalies (1 if either method flags an anomaly)
    df['Combined_Anomaly'] = df[['Speed_mps_Z_Anomaly', 'Speed_mps_MA_Anomaly']].max(axis=1)

    # Compare detected anomalies with ground truth
    print("Classification Report:")
    print(classification_report(df['Anomaly_Flag'], df['Combined_Anomaly']))

    # Calculate prediction accuracy
    correct_predictions = (df['Anomaly_Flag'] == df['Combined_Anomaly']).sum()
    total_predictions = len(df)
    prediction_accuracy = (correct_predictions / total_predictions) * 100
    print(f"Prediction Accuracy: {prediction_accuracy:.2f}%")

    # Step 3: Create and run the dashboard
    app = create_dashboard(df)
    app.run_server(debug=True)

# Run the program
if __name__ == '__main__':
    main()