# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Convert 'Anomaly_Flag' to binary (0 for Normal, 1 for Anomaly)
    df['Anomaly_Flag_Binary'] = df['Anomaly_Flag'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    return df

# Step 2: Prepare data for LSTM with proper train-test split
def prepare_lstm_data(df, features, sequence_length=12, train_ratio=0.8):
    # Select relevant features
    data = df[features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing sets
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Store the corresponding timestamps and ground truth labels
    timestamps = df['Timestamp'][sequence_length:].reset_index(drop=True)
    ground_truth = df['Anomaly_Flag_Binary'][sequence_length:].reset_index(drop=True)
    
    train_timestamps = timestamps[:train_size]
    test_timestamps = timestamps[train_size:train_size+len(X_test)]
    
    train_ground_truth = ground_truth[:train_size]
    test_ground_truth = ground_truth[train_size:train_size+len(X_test)]
    
    return (X_train, X_test, y_train, y_test, scaler, data_scaled, 
            train_timestamps, test_timestamps, train_ground_truth, test_ground_truth)

# Step 3: Build and train LSTM model
def build_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_shape))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Step 4: Detect anomalies using prediction errors (separate for train and test)
def detect_anomalies_lstm(model, X, data_scaled, scaler, sequence_length, start_idx, threshold_percentile=95):
    # Make predictions
    X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    predictions = model.predict(X_reshaped)
    
    # Inverse transform predictions
    predictions_full = np.zeros((predictions.shape[0], data_scaled.shape[1]))
    predictions_full[:, :predictions.shape[1]] = predictions
    predictions_inverse = scaler.inverse_transform(predictions_full)
    
    # Extract only the predicted features (not all columns)
    predictions_inverse = predictions_inverse[:, :predictions.shape[1]]
    
    # Get actual values
    actual_full = np.zeros((predictions.shape[0], data_scaled.shape[1]))
    for i in range(len(X)):
        actual_full[i, :predictions.shape[1]] = scaler.inverse_transform(
            data_scaled[start_idx + i + sequence_length].reshape(1, -1)
        )[0, :predictions.shape[1]]
    
    # Calculate errors (using mean absolute error across features)
    errors = np.mean(np.abs(predictions_inverse - actual_full[:, :predictions.shape[1]]), axis=1)
    
    # Use percentile-based threshold for better sensitivity to anomalies
    threshold = np.percentile(errors, threshold_percentile)
    
    # Detect anomalies
    anomalies = errors > threshold
    
    return anomalies, errors, threshold, predictions_inverse, actual_full[:, :predictions.shape[1]]

# Step 5: Create visualization dashboard focused on test set results
def create_lstm_dashboard(df, test_anomalies, test_errors, test_threshold, 
                         features, test_predictions, test_actuals, 
                         test_timestamps, test_ground_truth):
    app = dash.Dash(__name__)
    
    # Create a dataframe with test set anomaly predictions
    test_anomaly_df = pd.DataFrame({
        'Timestamp': test_timestamps,
        'Error': test_errors,
        'Is_Anomaly': test_anomalies,
        'Threshold': test_threshold,
        'Ground_Truth': test_ground_truth
    })
    
    # Add prediction and actual values for each feature
    for i, feature in enumerate(features):
        if i < test_predictions.shape[1]:  # Ensure we don't exceed dimensions
            test_anomaly_df[f'{feature}_Predicted'] = test_predictions[:, i]
            test_anomaly_df[f'{feature}_Actual'] = test_actuals[:, i]
    
    # Calculate performance metrics
    true_positives = ((test_anomaly_df['Is_Anomaly'] == True) & (test_anomaly_df['Ground_Truth'] == 1)).sum()
    false_positives = ((test_anomaly_df['Is_Anomaly'] == True) & (test_anomaly_df['Ground_Truth'] == 0)).sum()
    false_negatives = ((test_anomaly_df['Is_Anomaly'] == False) & (test_anomaly_df['Ground_Truth'] == 1)).sum()
    true_negatives = ((test_anomaly_df['Is_Anomaly'] == False) & (test_anomaly_df['Ground_Truth'] == 0)).sum()
    
    # Calculate precision, recall, F1-score and accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(test_anomaly_df)
    
    app.layout = html.Div([
        html.H1("LSTM-based Sequential Anomaly Detection (Test Set Evaluation)", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.Label("Select Feature for Visualization:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': i} for i, feature in enumerate(features) 
                         if i < test_predictions.shape[1]],
                value=0,
                multi=False,
                style={'marginBottom': '20px'}
            )
        ]),
        
        dcc.Graph(id='trend-graph'),
        
        dcc.Graph(id='error-graph'),
        
        html.Div([
            html.H3("Test Set Performance Metrics", style={'color': '#2c3e50'}),
            html.Div([
                html.Div([
                    html.P("Test Set Size:", style={'fontWeight': 'bold'}),
                    html.P(f"{len(test_anomaly_df)}")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Ground Truth Anomalies:", style={'fontWeight': 'bold'}),
                    html.P(f"{test_anomaly_df['Ground_Truth'].sum()}")
                ], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([
                    html.P("Detected Anomalies:", style={'fontWeight': 'bold'}),
                    html.P(f"{test_anomalies.sum()}")
                ], style={'width': '33%', 'display': 'inline-block'})
            ], style={'display': 'flex'}),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.P("True Positives:", style={'fontWeight': 'bold'}),
                    html.P(f"{true_positives}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("False Positives:", style={'fontWeight': 'bold'}),
                    html.P(f"{false_positives}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("False Negatives:", style={'fontWeight': 'bold'}),
                    html.P(f"{false_negatives}")
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([
                    html.P("True Negatives:", style={'fontWeight': 'bold'}),
                    html.P(f"{true_negatives}")
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
            ], style={'display': 'flex'}),
            
            html.Hr(),
            
            html.P(f"Error Threshold: {test_threshold:.4f}", style={'fontWeight': 'bold'})
            
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '30px'})
    ])
    
    @app.callback(
        [Output('trend-graph', 'figure'),
         Output('error-graph', 'figure')],
        [Input('feature-dropdown', 'value')]
    )
    def update_graphs(selected_feature_idx):
        selected_feature = features[selected_feature_idx]
        
        # Create trend visualization
        trend_fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Add actual values
        trend_fig.add_trace(
            go.Scatter(
                x=test_anomaly_df['Timestamp'],
                y=test_anomaly_df[f'{selected_feature}_Actual'],
                mode='lines',
                name='Actual Values',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add predicted values
        trend_fig.add_trace(
            go.Scatter(
                x=test_anomaly_df['Timestamp'],
                y=test_anomaly_df[f'{selected_feature}_Predicted'],
                mode='lines',
                name='LSTM Predictions',
                line=dict(color='green', width=1.5, dash='dash')
            )
        )
        
        # Add LSTM-detected anomalies
        lstm_anomaly_points = test_anomaly_df[test_anomaly_df['Is_Anomaly']]
        trend_fig.add_trace(
            go.Scatter(
                x=lstm_anomaly_points['Timestamp'],
                y=lstm_anomaly_points[f'{selected_feature}_Actual'],
                mode='markers',
                name='LSTM Detected Anomalies',
                marker=dict(color='red', size=10, symbol='circle')
            )
        )
        
        # Add ground truth anomalies
        ground_truth_points = test_anomaly_df[test_anomaly_df['Ground_Truth'] == 1]
        trend_fig.add_trace(
            go.Scatter(
                x=ground_truth_points['Timestamp'],
                y=ground_truth_points[f'{selected_feature}_Actual'],
                mode='markers',
                name='Ground Truth Anomalies',
                marker=dict(color='orange', size=12, symbol='x')
            )
        )
        
        trend_fig.update_layout(
            title=f"Test Set Analysis: {selected_feature} (Actual vs. Predicted)",
            xaxis_title="Timestamp",
            yaxis_title=selected_feature,
            legend=dict(x=0, y=1, traceorder="normal"),
            height=500,
            template="plotly_white",
            hovermode="closest"
        )
        
        # Create error visualization
        error_fig = go.Figure()
        
        # Add error values
        error_fig.add_trace(
            go.Scatter(
                x=test_anomaly_df['Timestamp'],
                y=test_anomaly_df['Error'],
                mode='lines',
                name='Prediction Error',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add threshold line
        error_fig.add_trace(
            go.Scatter(
                x=test_anomaly_df['Timestamp'],
                y=[test_threshold] * len(test_anomaly_df),
                mode='lines',
                name='Anomaly Threshold',
                line=dict(color='red', dash='dash', width=1.5)
            )
        )
        
        # Mark ground truth anomalies on error graph
        error_fig.add_trace(
            go.Scatter(
                x=ground_truth_points['Timestamp'],
                y=ground_truth_points['Error'],
                mode='markers',
                name='Ground Truth Anomalies',
                marker=dict(color='orange', size=12, symbol='x')
            )
        )
        
        error_fig.update_layout(
            title="Test Set Prediction Error Analysis",
            xaxis_title="Timestamp",
            yaxis_title="Error Magnitude",
            legend=dict(x=0, y=1, traceorder="normal"),
            height=400,
            template="plotly_white",
            hovermode="closest"
        )
        
        return trend_fig, error_fig
    
    return app

# Main function
def main():
    print("Loading and preprocessing data...")
    # Load and preprocess data
    file_path = 'UP-1_Anomaly_Detection_SynData.csv'
    df = load_and_preprocess_data(file_path)
    
    # Define features for LSTM - include all relevant numerical features
    features = ['Speed_mps', 'Battery_Level_Percent', 'Signal_Strength_Percent', 'Heading_deg']
    
    # Add rate of change features
    for feature in features.copy():
        df[f'{feature}_Change'] = df[feature].diff().fillna(0)
        features.append(f'{feature}_Change')
    
    # Set sequence length to capture patterns (12 represents 1 hour with 5-minute intervals)
    sequence_length = 12
    
    print("Preparing data with proper train-test split...")
    # Prepare data with proper train-test split - FIXED: Pass the entire dataframe
    (X_train, X_test, y_train, y_test, scaler, data_scaled,
     train_timestamps, test_timestamps, train_ground_truth, test_ground_truth) = prepare_lstm_data(
        df, features, sequence_length, train_ratio=0.8  # Pass entire df, not df[features]
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    print("Building and training LSTM model...")
    # Build LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    
    # Train model with early stopping
    history = model.fit(
        X_train, y_train, 
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    print("Detecting anomalies on test set...")
    # Detect anomalies on test set only
    train_size = len(X_train)
    test_anomalies, test_errors, test_threshold, test_predictions, test_actuals = detect_anomalies_lstm(
        model, 
        X_test, 
        data_scaled, 
        scaler, 
        sequence_length,
        start_idx=train_size,
        threshold_percentile=95  # Adjusted for better sensitivity
    )
    
    # Print test set summary statistics
    print("\n=== TEST SET EVALUATION ===")
    print(f"Test set size: {len(test_anomalies)}")
    print(f"Detected anomalies: {test_anomalies.sum()}")
    print(f"Detection rate: {(test_anomalies.sum() / len(test_anomalies)) * 100:.2f}%")
    print(f"Error threshold: {test_threshold:.4f}")
    print(f"Ground truth anomalies in test set: {test_ground_truth.sum()}")
    
    # Calculate performance metrics
    y_true = test_ground_truth.values
    y_pred = test_anomalies.astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    print("\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    print("\nLaunching visualization dashboard...")
    # Create and run dashboard with test set results
    app = create_lstm_dashboard(
        df, test_anomalies, test_errors, test_threshold,
        features, test_predictions, test_actuals, 
        test_timestamps, test_ground_truth
    )
    app.run_server(debug=True)

if __name__ == '__main__':
    main()