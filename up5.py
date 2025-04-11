# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset for sliding window analysis"""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Convert 'Anomaly_Flag' to binary (0 for Normal, 1 for Anomaly)
    df['Anomaly_Flag_Binary'] = df['Anomaly_Flag'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    return df

# Step 2: Define sliding window analysis function
def sliding_window_analysis(df, features, window_size=12, step_size=1, contamination=0.05):
    """Perform sliding window analysis on the dataset"""
    
    # Create a copy of the dataframe
    window_df = df.copy()
    
    # Initialize columns for anomaly scores and predictions
    window_df['Anomaly_Score'] = np.nan
    window_df['SW_Anomaly'] = 0
    
    # Create a StandardScaler for each feature
    scaler = StandardScaler()
    
    # Iterate through the dataset with the sliding window
    for i in range(0, len(df) - window_size + 1, step_size):
        # Get the current window
        current_window = df.iloc[i:i+window_size]
        
        # Extract features for the current window
        X = current_window[features].values
        
        # Scale the features
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest on the window
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)
        
        # Get anomaly scores for the last point in the window
        last_point_idx = i + window_size - 1
        last_point = X_scaled[-1].reshape(1, -1)
        score = model.decision_function(last_point)[0]
        prediction = model.predict(last_point)[0]
        
        # Store the results
        window_df.loc[window_df.index[last_point_idx], 'Anomaly_Score'] = score
        window_df.loc[window_df.index[last_point_idx], 'SW_Anomaly'] = 1 if prediction == -1 else 0
    
    return window_df

# Step 3: Create visualization dashboard
def create_sliding_window_dashboard(df, window_df, features):
    """Create a Dash dashboard for sliding window analysis visualization"""
    app = dash.Dash(__name__)
    
    # Calculate feature statistics for the heatmap
    feature_stats = {}
    for feature in features:
        # Calculate z-scores for better visualization
        z_scores = (df[feature] - df[feature].mean()) / df[feature].std()
        feature_stats[feature] = z_scores
    
    # Create a dataframe for the heatmap
    heatmap_df = pd.DataFrame(feature_stats)
    
    app.layout = html.Div([
        html.H1("Sliding Window Time-Series Analysis with Real-Time Visualization", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                html.Label("Window Size:"),
                dcc.Slider(
                    id='window-size-slider',
                    min=6,
                    max=24,
                    step=6,
                    marks={i: f'{i*5} min' for i in range(6, 25, 6)},
                    value=12
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Step Size:"),
                dcc.Slider(
                    id='step-size-slider',
                    min=1,
                    max=6,
                    step=1,
                    marks={i: f'{i*5} min' for i in range(1, 7)},
                    value=1
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
        ]),
        
        html.Div([
            html.Label("Contamination Parameter (expected anomaly rate):"),
            dcc.Slider(
                id='contamination-slider',
                min=0.01,
                max=0.1,
                step=0.01,
                marks={i/100: f'{i}%' for i in range(1, 11)},
                value=0.05
            ),
        ]),
        
        html.Button('Update Analysis', id='update-button', n_clicks=0,
                   style={'marginTop': '20px', 'marginBottom': '20px', 
                          'backgroundColor': '#3498db', 'color': 'white',
                          'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px'}),
        
        dcc.Graph(id='heatmap-graph'),
        
        dcc.Graph(id='anomaly-score-graph'),
        
        html.Div([
            html.Label("Select Feature for Detail View:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature} for feature in features],
                value=features[0],
                multi=False
            )
        ]),
        
        dcc.Graph(id='feature-detail-graph'),
        
        html.Div([
            html.H3("Sliding Window Analysis Summary", style={'color': '#2c3e50'}),
            html.Div(id='analysis-summary')
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '30px'}),
        
        html.Div([
            html.H3("Performance Metrics", style={'color': '#2c3e50'}),
            html.Div(id='performance-metrics')
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '30px'})
    ])
    
    @app.callback(
        [Output('heatmap-graph', 'figure'),
         Output('anomaly-score-graph', 'figure'),
         Output('feature-detail-graph', 'figure'),
         Output('analysis-summary', 'children'),
         Output('performance-metrics', 'children')],
        [Input('update-button', 'n_clicks'),
         Input('feature-dropdown', 'value')],
        [State('window-size-slider', 'value'),
         State('step-size-slider', 'value'),
         State('contamination-slider', 'value')]
    )
    def update_graphs(n_clicks, selected_feature, window_size, step_size, contamination):
        # Re-run analysis with current parameters
        updated_window_df = sliding_window_analysis(
            df, features, window_size, step_size, contamination
        )
        
        # Create heatmap visualization
        heatmap_data = []
        for feature in features:
            # Calculate z-scores for the feature
            z_scores = (df[feature] - df[feature].mean()) / df[feature].std()
            # Resample to reduce data points (for better visualization)
            resampled = z_scores.rolling(window=window_size, min_periods=1).mean()
            heatmap_data.append(resampled)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=df['Timestamp'],
            y=features,
            colorscale='Viridis',
            colorbar=dict(title='Z-Score')
        ))
        
        heatmap_fig.update_layout(
            title="Feature Intensity Heatmap Over Time",
            xaxis_title="Time",
            yaxis_title="Feature",
            height=400,
            template="plotly_white"
        )
        
        # Mark anomalies on the heatmap
        anomaly_points = updated_window_df[updated_window_df['SW_Anomaly'] == 1]
        if not anomaly_points.empty:
            # Add markers for detected anomalies
            for feature_idx, feature in enumerate(features):
                heatmap_fig.add_trace(
                    go.Scatter(
                        x=anomaly_points['Timestamp'],
                        y=[feature] * len(anomaly_points),
                        mode='markers',
                        name='Detected Anomalies',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=feature_idx == 0  # Only show legend once
                    )
                )
        
        # Ground truth anomalies
        ground_truth_points = df[df['Anomaly_Flag_Binary'] == 1]
        if not ground_truth_points.empty:
            # Add markers for ground truth anomalies
            for feature_idx, feature in enumerate(features):
                heatmap_fig.add_trace(
                    go.Scatter(
                        x=ground_truth_points['Timestamp'],
                        y=[feature] * len(ground_truth_points),
                        mode='markers',
                        name='Ground Truth Anomalies',
                        marker=dict(color='orange', size=8, symbol='circle'),
                        showlegend=feature_idx == 0  # Only show legend once
                    )
                )
        
        # Create anomaly score visualization
        score_fig = go.Figure()
        
        # Add anomaly scores
        valid_scores = updated_window_df.dropna(subset=['Anomaly_Score'])
        score_fig.add_trace(
            go.Scatter(
                x=valid_scores['Timestamp'],
                y=valid_scores['Anomaly_Score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add threshold line (typically 0 for Isolation Forest)
        score_fig.add_trace(
            go.Scatter(
                x=valid_scores['Timestamp'],
                y=[0] * len(valid_scores),
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash', width=1.5)
            )
        )
        
        # Mark detected anomalies
        if not anomaly_points.empty:
            score_fig.add_trace(
                go.Scatter(
                    x=anomaly_points['Timestamp'],
                    y=anomaly_points['Anomaly_Score'],
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                )
            )
        
        score_fig.update_layout(
            title="Anomaly Scores from Sliding Window Analysis",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=300,
            template="plotly_white"
        )
        
        # Create feature detail visualization
        detail_fig = go.Figure()
        
        # Add raw feature data
        detail_fig.add_trace(
            go.Scatter(
                x=df['Timestamp'],
                y=df[selected_feature],
                mode='lines',
                name=selected_feature,
                line=dict(color='blue', width=2)
            )
        )
        
        # Add moving average
        ma = df[selected_feature].rolling(window=window_size).mean()
        detail_fig.add_trace(
            go.Scatter(
                x=df['Timestamp'],
                y=ma,
                mode='lines',
                name=f'Moving Avg ({window_size*5} min)',
                line=dict(color='green', dash='dash', width=1.5)
            )
        )
        
        # Mark detected anomalies
        if not anomaly_points.empty:
            detail_fig.add_trace(
                go.Scatter(
                    x=anomaly_points['Timestamp'],
                    y=anomaly_points[selected_feature],
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                )
            )
        
        # Mark ground truth anomalies
        if not ground_truth_points.empty:
            detail_fig.add_trace(
                go.Scatter(
                    x=ground_truth_points['Timestamp'],
                    y=ground_truth_points[selected_feature],
                    mode='markers',
                    name='Ground Truth Anomalies',
                    marker=dict(color='orange', size=12, symbol='circle')
                )
            )
        
        detail_fig.update_layout(
            title=f"Detailed View of {selected_feature}",
            xaxis_title="Time",
            yaxis_title=selected_feature,
            height=400,
            template="plotly_white"
        )
        
        # Create summary statistics
        valid_results = updated_window_df.dropna(subset=['Anomaly_Score'])
        total_points = len(valid_results)
        detected_anomalies = valid_results['SW_Anomaly'].sum()
        anomaly_rate = (detected_anomalies / total_points) * 100 if total_points > 0 else 0
        
        summary_children = [
            html.Div([
                html.Div([
                    html.P("Total Analyzed Points:", style={'fontWeight': 'bold'}),
                    html.P(f"{total_points}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Detected Anomalies:", style={'fontWeight': 'bold'}),
                    html.P(f"{detected_anomalies}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Anomaly Rate:", style={'fontWeight': 'bold'}),
                    html.P(f"{anomaly_rate:.2f}%")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Expected Anomaly Rate:", style={'fontWeight': 'bold'}),
                    html.P(f"{contamination*100:.2f}%")
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'display': 'flex'}),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.P("Window Size:", style={'fontWeight': 'bold'}),
                    html.P(f"{window_size*5} minutes")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Step Size:", style={'fontWeight': 'bold'}),
                    html.P(f"{step_size*5} minutes")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Average Anomaly Score:", style={'fontWeight': 'bold'}),
                    html.P(f"{valid_results['Anomaly_Score'].mean():.4f}")
                ], style={'width': '33%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ]
        
        # Calculate performance metrics
        # Compare detected anomalies with ground truth
        # First align the arrays (only where we have predictions)
        y_true = valid_results['Anomaly_Flag_Binary'].values
        y_pred = valid_results['SW_Anomaly'].values
        
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
        
        return heatmap_fig, score_fig, detail_fig, summary_children, performance_children
    
    return app

# Main function
def main():
    print("Loading and preprocessing data...")
    # Load and preprocess data
    file_path = 'UP-1_Anomaly_Detection_SynData.csv'
    df = load_and_preprocess_data(file_path)
    
    # Define features for analysis
    features = ['Speed_mps', 'Battery_Level_Percent', 'Signal_Strength_Percent', 'Heading_deg']
    
    print("Performing initial sliding window analysis...")
    # Perform initial sliding window analysis
    window_size = 12  # 1-hour window (with 5-minute intervals)
    step_size = 1
    contamination = 0.05  # Expected anomaly rate
    
    window_df = sliding_window_analysis(df, features, window_size, step_size, contamination)
    
    # Print summary statistics
    valid_results = window_df.dropna(subset=['Anomaly_Score'])
    detected_anomalies = valid_results['SW_Anomaly'].sum()
    total_points = len(valid_results)
    anomaly_rate = (detected_anomalies / total_points) * 100 if total_points > 0 else 0
    
    print("\n=== SLIDING WINDOW ANOMALY DETECTION SUMMARY ===")
    print(f"Total analyzed points: {total_points}")
    print(f"Detected anomalies: {detected_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print(f"Ground truth anomalies: {df['Anomaly_Flag_Binary'].sum()}")
    
    # Calculate performance metrics
    y_true = valid_results['Anomaly_Flag_Binary'].values
    y_pred = valid_results['SW_Anomaly'].values
    
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
    app = create_sliding_window_dashboard(df, window_df, features)
    app.run_server(debug=True)

if __name__ == '__main__':
    main()