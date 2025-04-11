# Import required libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load and preprocess the dataset
file_path = 'UP-1_Anomaly_Detection_SynData.csv'
df = pd.read_csv(file_path)

# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract time-based features (optional, but useful for filtering in the dashboard)
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month

# Split data into first two days (X1) and last day (X2)
split_date = df['Timestamp'].max() - pd.DateOffset(days=1)
X1 = df[df['Timestamp'] <= split_date]
X2 = df[df['Timestamp'] > split_date]

# Drop the original 'Timestamp' column if not needed
X1 = X1.drop('Timestamp', axis=1)
X2 = X2.drop('Timestamp', axis=1)

# Step 2: Train the Isolation Forest model
# Prepare the feature set (exclude the 'Anomaly_Flag' column)
X1_train = X1.drop('Anomaly_Flag', axis=1)
X2_test = X2.drop('Anomaly_Flag', axis=1)

# Initialize the Isolation Forest model
# Set contamination to the expected proportion of anomalies in the data
model = IsolationForest(contamination=0.04, random_state=42)

# Fit the model
model.fit(X1_train)

# Predict anomalies (-1 for anomalies, 1 for normal)
X2['Anomaly_Prediction'] = model.predict(X2_test)

# Convert predictions to binary (1 for normal, 0 for anomaly)
X2['Anomaly_Prediction'] = X2['Anomaly_Prediction'].apply(lambda x: 0 if x == -1 else 1)

# Convert 'Anomaly_Flag' to binary (1 for normal, 0 for anomaly)
X2['Anomaly_Flag'] = X2['Anomaly_Flag'].apply(lambda x: 1 if x == 'Normal' else 0)

# Print prediction array for X2
# prediction_array = ''.join(X2['Anomaly_Prediction'].astype(str).values)
# print("\nX2 Anomaly Predictions Array:")
# print(prediction_array)

# Print Anomaly_Prediction and Anomaly_Flag value counts for comparison
print("\nAnomaly_Flag Value Counts:")
print(X2['Anomaly_Flag'].value_counts())
print("\nAnomaly_Prediction Value Counts:")
print(X2['Anomaly_Prediction'].value_counts())

# Print classification report
print("\nClassification Report:")
print(classification_report(X2['Anomaly_Flag'], X2['Anomaly_Prediction']))

# Calculate prediction accuracy
correct_predictions = (X2['Anomaly_Flag'] == X2['Anomaly_Prediction']).sum()
total_predictions = len(X2)
prediction_accuracy = (correct_predictions / total_predictions) * 100
print(f"\nPrediction Accuracy: {prediction_accuracy:.2f}%")

# Step 3: Create an interactive dashboard
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard (X2 Predictions)", style={'textAlign': 'center'}),
    
    # Dropdown for filtering by location (Latitude and Longitude)
    html.Label("Filter by Location:"),
    html.Div([
        html.Label("Latitude Range:"),
        dcc.RangeSlider(
            id='latitude-slider',
            min=X2['Latitude'].min(),
            max=X2['Latitude'].max(),
            step=0.01,
            marks={i: f'{i:.2f}' for i in range(int(X2['Latitude'].min()), int(X2['Latitude'].max()) + 1)},
            value=[X2['Latitude'].min(), X2['Latitude'].max()]
        ),
        html.Label("Longitude Range:"),
        dcc.RangeSlider(
            id='longitude-slider',
            min=X2['Longitude'].min(),
            max=X2['Longitude'].max(),
            step=0.01,
            marks={i: f'{i:.2f}' for i in range(int(X2['Longitude'].min()), int(X2['Longitude'].max()) + 1)},
            value=[X2['Longitude'].min(), X2['Longitude'].max()]
        ),
    ]),
    
    # Dropdown for filtering by time (Hour, Day, Month)
    html.Label("Filter by Time:"),
    dcc.Dropdown(
        id='time-filter',
        options=[
            {'label': 'Hour', 'value': 'Hour'},
            {'label': 'Day', 'value': 'Day'},
            {'label': 'Month', 'value': 'Month'}
        ],
        value='Hour',
        multi=False
    ),
    
    # Graph to display anomalies
    dcc.Graph(id='anomaly-graph')
])

# Define callback to update the graph based on filters
@app.callback(
    Output('anomaly-graph', 'figure'),
    [Input('latitude-slider', 'value'),
     Input('longitude-slider', 'value'),
     Input('time-filter', 'value')]
)
def update_graph(latitude_range, longitude_range, time_filter):
    # Filter data based on latitude and longitude
    filtered_df = X2[
        (X2['Latitude'] >= latitude_range[0]) & (X2['Latitude'] <= latitude_range[1]) &
        (X2['Longitude'] >= longitude_range[0]) & (X2['Longitude'] <= longitude_range[1])
    ]
    
    # Group data by the selected time filter and count anomalies (0)
    grouped_df = filtered_df.groupby(time_filter)['Anomaly_Prediction'].apply(lambda x: (x == 0).sum()).reset_index()
    
    # Create a bar chart to show anomalies over time
    fig = px.bar(
        grouped_df,
        x=time_filter,
        y='Anomaly_Prediction',
        title=f'X2 Anomalies by {time_filter}',
        labels={'Anomaly_Prediction': 'Number of Anomalies'}
    )
    
    return fig

# Step 4: Run the app
if __name__ == '__main__':
    app.run_server(debug=True)