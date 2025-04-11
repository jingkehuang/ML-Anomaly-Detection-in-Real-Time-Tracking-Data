# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset for correlation analysis"""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Convert 'Anomaly_Flag' to binary (0 for Normal, 1 for Anomaly)
    df['Anomaly_Flag_Binary'] = df['Anomaly_Flag'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    return df

# Step 2: Calculate rolling correlations between metrics
def calculate_rolling_correlations(df, metrics, window_size=24):
    """Calculate rolling correlations between pairs of metrics"""
    # Initialize a dictionary to store correlation dataframes
    correlation_data = {}
    
    # Calculate rolling correlations for each pair of metrics
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            pair_name = f"{metric1} vs {metric2}"
            
            # Calculate rolling correlation
            rolling_corr = df[metric1].rolling(window=window_size).corr(df[metric2])
            
            # Store in dictionary
            correlation_data[pair_name] = rolling_corr
    
    return correlation_data

# Step 3: Detect correlation-based anomalies
def detect_correlation_anomalies(df, correlation_data, window_size=24, threshold_percentile=95):
    """Detect anomalies based on correlation changes"""
    # Create a copy of the dataframe to store anomaly flags
    anomaly_df = df.copy()
    
    # Initialize anomaly column
    anomaly_df['Correlation_Anomaly'] = 0
    
    # Calculate correlation changes and detect anomalies
    for pair_name, correlation_series in correlation_data.items():
        # Calculate the absolute change in correlation
        correlation_change = correlation_series.diff().abs()
        
        # Set NaN values to 0
        correlation_change = correlation_change.fillna(0)
        
        # Calculate threshold based on percentile
        threshold = np.percentile(correlation_change.dropna(), threshold_percentile)
        
        # Detect anomalies where correlation change exceeds threshold
        anomalies = correlation_change > threshold
        
        # Add pair-specific anomaly column
        pair_column = f"{pair_name}_Anomaly"
        anomaly_df[pair_column] = 0
        anomaly_df.loc[anomalies.index, pair_column] = anomalies.astype(int)
        
        # Update overall anomaly flag (union of all pair anomalies)
        anomaly_df.loc[anomalies.index, 'Correlation_Anomaly'] = (
            anomaly_df.loc[anomalies.index, 'Correlation_Anomaly'] | anomalies.astype(int)
        )
    
    return anomaly_df, correlation_data

# Step 4: Create correlation visualization dashboard
def create_correlation_dashboard(df, anomaly_df, correlation_data, metrics):
    """Create a Dash dashboard for correlation-based anomaly visualization"""
    app = dash.Dash(__name__)
    
    # Get all metric pairs for dropdown
    metric_pairs = list(correlation_data.keys())
    
    app.layout = html.Div([
        html.H1("Multi-Metric Correlation-Based Anomaly Detection", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                html.Label("Select Metric Pair:"),
                dcc.Dropdown(
                    id='metric-pair-dropdown',
                    options=[{'label': pair, 'value': pair} for pair in metric_pairs],
                    value=metric_pairs[0],
                    multi=False
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Correlation Window Size:"),
                dcc.Slider(
                    id='window-size-slider',
                    min=6,
                    max=48,
                    step=6,
                    marks={i: f'{i*5} min' for i in range(6, 49, 6)},
                    value=24
                )
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
        ]),
        
        dcc.Graph(id='correlation-graph'),
        
        dcc.Graph(id='metrics-comparison-graph'),
        
        html.Div([
            html.H3("Correlation Analysis Summary", style={'color': '#2c3e50'}),
            html.Div(id='correlation-summary')
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '30px'}),
        
        html.Div([
            html.H3("Performance Metrics", style={'color': '#2c3e50'}),
            html.Div(id='performance-metrics')
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '30px'})
    ])
    
    @app.callback(
        [Output('correlation-graph', 'figure'),
         Output('metrics-comparison-graph', 'figure'),
         Output('correlation-summary', 'children'),
         Output('performance-metrics', 'children')],
        [Input('metric-pair-dropdown', 'value'),
         Input('window-size-slider', 'value')]
    )
    def update_graphs(selected_pair, window_size):
        # Recalculate correlations with the selected window size
        updated_correlation_data = calculate_rolling_correlations(
            df, metrics, window_size=window_size
        )
        
        # Recalculate anomalies
        updated_anomaly_df, _ = detect_correlation_anomalies(
            df, updated_correlation_data, window_size=window_size
        )
        
        # Get the selected correlation series
        correlation_series = updated_correlation_data[selected_pair]
        
        # Extract the metric names
        metric1, metric2 = selected_pair.split(' vs ')
        
        # Create correlation visualization
        corr_fig = go.Figure()
        
        # Add correlation line
        corr_fig.add_trace(
            go.Scatter(
                x=df['Timestamp'][window_size-1:],
                y=correlation_series,
                mode='lines',
                name='Correlation Coefficient',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add correlation change anomalies
        pair_column = f"{selected_pair}_Anomaly"
        if pair_column in updated_anomaly_df.columns:
            anomaly_points = updated_anomaly_df[updated_anomaly_df[pair_column] == 1]
            if not anomaly_points.empty:
                corr_fig.add_trace(
                    go.Scatter(
                        x=anomaly_points['Timestamp'],
                        y=correlation_series[anomaly_points.index],
                        mode='markers',
                        name='Correlation Anomalies',
                        marker=dict(color='red', size=10, symbol='circle')
                    )
                )
        
        # Add ground truth anomalies
        ground_truth_points = updated_anomaly_df[updated_anomaly_df['Anomaly_Flag_Binary'] == 1]
        if not ground_truth_points.empty:
            ground_truth_with_corr = ground_truth_points[ground_truth_points.index >= window_size-1]
            if not ground_truth_with_corr.empty:
                corr_fig.add_trace(
                    go.Scatter(
                        x=ground_truth_with_corr['Timestamp'],
                        y=correlation_series[ground_truth_with_corr.index],
                        mode='markers',
                        name='Ground Truth Anomalies',
                        marker=dict(color='orange', size=12, symbol='x')
                    )
                )
        
        corr_fig.update_layout(
            title=f"Rolling Correlation Between {metric1} and {metric2} (Window: {window_size*5} minutes)",
            xaxis_title="Timestamp",
            yaxis_title="Correlation Coefficient",
            legend=dict(x=0, y=1, traceorder="normal"),
            height=400,
            template="plotly_white"
        )
        
        # Create metrics comparison visualization
        metrics_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add first metric
        metrics_fig.add_trace(
            go.Scatter(
                x=df['Timestamp'],
                y=df[metric1],
                mode='lines',
                name=metric1,
                line=dict(color='green', width=2)
            ),
            secondary_y=False
        )
        
        # Add second metric
        metrics_fig.add_trace(
            go.Scatter(
                x=df['Timestamp'],
                y=df[metric2],
                mode='lines',
                name=metric2,
                line=dict(color='purple', width=2)
            ),
            secondary_y=True
        )
        
        # Add correlation anomalies
        if not anomaly_points.empty:
            metrics_fig.add_trace(
                go.Scatter(
                    x=anomaly_points['Timestamp'],
                    y=anomaly_points[metric1],
                    mode='markers',
                    name='Correlation Anomalies',
                    marker=dict(color='red', size=10, symbol='circle')
                ),
                secondary_y=False
            )
        
        # Add ground truth anomalies
        if not ground_truth_points.empty:
            metrics_fig.add_trace(
                go.Scatter(
                    x=ground_truth_points['Timestamp'],
                    y=ground_truth_points[metric1],
                    mode='markers',
                    name='Ground Truth Anomalies',
                    marker=dict(color='orange', size=12, symbol='x')
                ),
                secondary_y=False
            )
        
        metrics_fig.update_layout(
            title=f"Comparison of {metric1} and {metric2} Over Time",
            xaxis_title="Timestamp",
            legend=dict(x=0, y=1, traceorder="normal"),
            height=400,
            template="plotly_white"
        )
        
        metrics_fig.update_yaxes(title_text=metric1, secondary_y=False)
        metrics_fig.update_yaxes(title_text=metric2, secondary_y=True)
        
        # Create summary statistics
        # Count correlation anomalies for the selected pair
        pair_anomalies = updated_anomaly_df[pair_column].sum() if pair_column in updated_anomaly_df.columns else 0
        total_points = len(updated_anomaly_df) - (window_size - 1)
        anomaly_rate = (pair_anomalies / total_points) * 100 if total_points > 0 else 0
        
        # Calculate average correlation and standard deviation
        avg_correlation = correlation_series.mean()
        std_correlation = correlation_series.std()
        
        summary_children = [
            html.Div([
                html.Div([
                    html.P("Total Data Points:", style={'fontWeight': 'bold'}),
                    html.P(f"{total_points}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Correlation Anomalies:", style={'fontWeight': 'bold'}),
                    html.P(f"{pair_anomalies}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Anomaly Rate:", style={'fontWeight': 'bold'}),
                    html.P(f"{anomaly_rate:.2f}%")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Window Size:", style={'fontWeight': 'bold'}),
                    html.P(f"{window_size*5} minutes")
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'display': 'flex'}),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.P("Average Correlation:", style={'fontWeight': 'bold'}),
                    html.P(f"{avg_correlation:.4f}")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Correlation Std Dev:", style={'fontWeight': 'bold'}),
                    html.P(f"{std_correlation:.4f}")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Correlation Range:", style={'fontWeight': 'bold'}),
                    html.P(f"{correlation_series.min():.4f} to {correlation_series.max():.4f}")
                ], style={'width': '33%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ]
        
        # Calculate performance metrics for correlation-based anomaly detection
        # Compare detected anomalies with ground truth
        y_true = updated_anomaly_df['Anomaly_Flag_Binary'].values[window_size-1:]
        y_pred = updated_anomaly_df['Correlation_Anomaly'].values[window_size-1:]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate precision, recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        performance_children = [
            html.Div([
                html.Div([
                    html.P("True Positives:", style={'fontWeight': 'bold'}),
                    html.P(f"{tp}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("False Positives:", style={'fontWeight': 'bold'}),
                    html.P(f"{fp}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("False Negatives:", style={'fontWeight': 'bold'}),
                    html.P(f"{fn}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("True Negatives:", style={'fontWeight': 'bold'}),
                    html.P(f"{tn}")
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'display': 'flex'}),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.P("Precision:", style={'fontWeight': 'bold'}),
                    html.P(f"{precision:.4f}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Recall:", style={'fontWeight': 'bold'}),
                    html.P(f"{recall:.4f}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("F1 Score:", style={'fontWeight': 'bold'}),
                    html.P(f"{f1:.4f}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Accuracy:", style={'fontWeight': 'bold'}),
                    html.P(f"{accuracy:.4f}")
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ]
        
        return corr_fig, metrics_fig, summary_children, performance_children
    
    return app

# Main function
def main():
    print("Loading and preprocessing data...")
    # Load and preprocess data
    file_path = 'UP-1_Anomaly_Detection_SynData.csv'
    df = load_and_preprocess_data(file_path)
    
    # Define metrics for correlation analysis
    metrics = ['Speed_mps', 'Battery_Level_Percent', 'Signal_Strength_Percent', 'Heading_deg']
    
    print("Calculating rolling correlations...")
    # Calculate initial correlations with default window size
    window_size = 24  # 2-hour window (with 5-minute intervals)
    correlation_data = calculate_rolling_correlations(df, metrics, window_size)
    
    print("Detecting correlation-based anomalies...")
    # Detect anomalies based on correlation changes
    anomaly_df, _ = detect_correlation_anomalies(df, correlation_data, window_size)
    
    # Print summary statistics
    total_corr_anomalies = anomaly_df['Correlation_Anomaly'].sum()
    total_points = len(anomaly_df) - (window_size - 1)
    anomaly_rate = (total_corr_anomalies / total_points) * 100 if total_points > 0 else 0
    
    print("\n=== CORRELATION ANOMALY DETECTION SUMMARY ===")
    print(f"Total data points: {total_points}")
    print(f"Detected correlation anomalies: {total_corr_anomalies}")
    print(f"Correlation anomaly rate: {anomaly_rate:.2f}%")
    print(f"Ground truth anomalies: {anomaly_df['Anomaly_Flag_Binary'].sum()}")
    
    # Calculate performance metrics
    y_true = anomaly_df['Anomaly_Flag_Binary'].values[window_size-1:]
    y_pred = anomaly_df['Correlation_Anomaly'].values[window_size-1:]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
    
    print("\nLaunching visualization dashboard...")
    # Create and run dashboard
    app = create_correlation_dashboard(df, anomaly_df, correlation_data, metrics)
    app.run_server(debug=True)

if __name__ == '__main__':
    main()