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

# Drop the original 'Timestamp' column if not needed
df.drop('Timestamp', axis=1, inplace=True)

# Step 2: Train the Isolation Forest model
# Prepare the feature set (exclude the 'Anomaly_Flag' column)
X = df.drop('Anomaly_Flag', axis=1)

# Initialize the Isolation Forest model
# Set contamination to the expected proportion of anomalies in the data
model = IsolationForest(contamination=0.04, random_state=42)

# Fit the model
model.fit(X)

# Predict anomalies (-1 for anomalies, 1 for normal)
df['Anomaly_Prediction'] = model.predict(X)

# Convert predictions to binary (1 for normal, 0 for anomaly)
df['Anomaly_Prediction'] = df['Anomaly_Prediction'].apply(lambda x: 0 if x == -1 else 1)

# Convert 'Anomaly_Flag' to binary (1 for normal, 0 for anomaly)
df['Anomaly_Flag'] = df['Anomaly_Flag'].apply(lambda x: 1 if x == 'Normal' else 0)

# print Anomaly_Prediction and Anomaly_Flag value counts for comparison
print(df['Anomaly_Flag'].value_counts())
print(df['Anomaly_Prediction'].value_counts())

print("Classification Report:")
print(classification_report(df['Anomaly_Flag'], df['Anomaly_Prediction']))

# Calculate prediction accuracy
correct_predictions = (df['Anomaly_Flag'] == df['Anomaly_Prediction']).sum()
total_predictions = len(df)
prediction_accuracy = (correct_predictions / total_predictions) * 100
print(f"Prediction Accuracy: {prediction_accuracy:.2f}%")


# Step 3: Create an interactive dashboard
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown for filtering by location (Latitude and Longitude)
    html.Label("Filter by Location:"),
    html.Div([
        html.Label("Latitude Range:"),
        dcc.RangeSlider(
            id='latitude-slider',
            min=df['Latitude'].min(),
            max=df['Latitude'].max(),
            step=0.01,
            marks={i: f'{i:.2f}' for i in range(int(df['Latitude'].min()), int(df['Latitude'].max()) + 1)},
            value=[df['Latitude'].min(), df['Latitude'].max()]
        ),
        html.Label("Longitude Range:"),
        dcc.RangeSlider(
            id='longitude-slider',
            min=df['Longitude'].min(),
            max=df['Longitude'].max(),
            step=0.01,
            marks={i: f'{i:.2f}' for i in range(int(df['Longitude'].min()), int(df['Longitude'].max()) + 1)},
            value=[df['Longitude'].min(), df['Longitude'].max()]
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
    filtered_df = df[
        (df['Latitude'] >= latitude_range[0]) & (df['Latitude'] <= latitude_range[1]) &
        (df['Longitude'] >= longitude_range[0]) & (df['Longitude'] <= longitude_range[1])
    ]
    
    # Group data by the selected time filter and count anomalies (0)
    grouped_df = filtered_df.groupby(time_filter)['Anomaly_Prediction'].apply(lambda x: (x == 0).sum()).reset_index()
    
    # Create a bar chart to show anomalies over time
    fig = px.bar(
        grouped_df,
        x=time_filter,
        y='Anomaly_Prediction',
        title=f'Anomalies by {time_filter}',
        labels={'Anomaly_Prediction': 'Number of Anomalies'}
    )
    
    return fig

# Step 4: Run the app
if __name__ == '__main__':
    app.run_server(debug=True)