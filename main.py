# ENGINEERING DATA ANALYTICS SYSTEM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

#creating the outputs folder

os.makedirs("outputs", exist_ok=True)

#loading the dataset

# Define the dataset path using your absolute file path 
dataset_path = r"C:\Users\User\Documents\EDS_25-0598_Echano\data\dataset_original.csv" 

# Load dataset 
df = pd.read_csv(dataset_path, encoding="utf-8")

def load_data(file_path):

    try:

        df = pd.read_csv(dataset_path, encoding="utf-8")

        print("Dataset Loaded Successfully!")
        print(df.head())

        return df

    except FileNotFoundError:

        print("ERROR: File not found.")
        return None

    except Exception as e:

        print("ERROR:", e)
        return None


#This cleans the data

def clean_data(df):

    try:

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        df = df.ffill()

        # Convert numeric columns
        numeric_cols = df.columns.drop('timestamp')

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove remaining nulls
        df = df.dropna()

        print("Data Cleaned Successfully!")

        return df

    except Exception as e:

        print("Cleaning Error:", e)
        return None
    
# Remove remaining nulls
df = df.dropna()

# Saving the clean data
df.to_csv(
    "data/dataset_cleaned.csv",
    index=False
)

print("Data Cleaned Successfully!")
print("Cleaned dataset saved!")

# Generate the Statistical Table

def generate_statistics_table(df):

    try:

        numeric_df = df.select_dtypes(include=np.number)

        stats_table = pd.DataFrame({

            'Mean': numeric_df.mean(),
            'Median': numeric_df.median(),
            'Std Dev': numeric_df.std(),
            'Variance': numeric_df.var(),
            'Minimum': numeric_df.min(),
            'Maximum': numeric_df.max(),
            'Skewness': numeric_df.skew()

        })

        print("\n===== STATISTICAL TABLE =====")
        print(stats_table)

        # Save table
        stats_table.to_csv(
            "outputs/statistics_table.csv"
        )

        print("\nStatistics Table Saved!")

        return stats_table

    except Exception as e:

        print("Statistics Error:", e)

# Engineering Interpretation

def engineering_interpretation(df):

    try:

        vibration_mean = df['vibration_x'].mean()

        vibration_std = df['vibration_x'].std()

        print("\n===== ENGINEERING INTERPRETATION =====")

        if vibration_mean > 0.05:

            print(
                "High vibration levels may indicate gearbox imbalance, shaft misalignment, or bearing wear."
            )

        else:

            print(
                "Low vibration levels indicate stable gearbox operation."
            )

        if vibration_std > 0.02:

            print(
                "Large vibration variation suggests unstable rotating conditions."
            )

        else:

            print(
                "Stable vibration distribution indicates healthy machine behavior."
            )

    except Exception as e:

        print("Interpretation Error:", e)


#Correlation Analysis

def correlation_analysis(df):

    try:

        numeric_df = df.select_dtypes(include=np.number)

        correlation_matrix = numeric_df.corr()

        print("\n===== CORRELATION MATRIX =====")
        print(correlation_matrix)

        return correlation_matrix

    except Exception as e:

        print("Correlation Error:", e)


# Comparative Analysis
def comparative_analysis(df):

    try:

        low_temp = df[
            df['gearbox_bearing_temp'] <= 70
        ]['vibration_x']

        high_temp = df[
            df['gearbox_bearing_temp'] > 70
        ]['vibration_x']

        print("\n===== COMPARATIVE ANALYSIS =====")

        print(
            "Low Temperature Mean Vibration:",
            np.mean(low_temp)
        )

        print(
            "High Temperature Mean Vibration:",
            np.mean(high_temp)
        )

        if np.mean(high_temp) > np.mean(low_temp):

            print(
                "Higher temperatures increase vibration amplitude."
            )

        else:

            print(
                "Temperature has minimal effect on vibration."
            )

    except Exception as e:

        print("Comparative Analysis Error:", e)

# Anomaly Detection

def anomaly_detection(df):

    try:

        features = df.select_dtypes(include=np.number)

        model = IsolationForest(
            contamination=0.03,
            random_state=42
        )

        df['anomaly'] = model.fit_predict(features)

        df['anomaly_label'] = df['anomaly'].map({

            1: 'Normal',
            -1: 'Anomaly'

        })

        print("Anomaly Detection Completed!")

        return df

    except Exception as e:

        print("Anomaly Error:", e)
        return df


# STATIC VISUALIZATIONS
def static_graphs(df):

    try:

        # 1. TIME SERIES GRAPH
        plt.figure(figsize=(12, 5))

        plt.plot(
            df['timestamp'],
            df['vibration_x'],
            label='Vibration X'
        )

        plt.plot(
            df['timestamp'],
            df['vibration_y'],
            label='Vibration Y'
        )

        plt.plot(
            df['timestamp'],
            df['vibration_z'],
            label='Vibration Z'
        )

        plt.title("Gearbox Vibration Trends")

        plt.xlabel("Timestamp")
        plt.ylabel("Vibration")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            "outputs/time_series.png"
        )

        plt.show()

        # 2. HEATMAP
        plt.figure(figsize=(10, 6))

        sns.heatmap(
            df.select_dtypes(include=np.number).corr(),
            annot=True,
            cmap='coolwarm'
        )

        plt.title("Correlation Heatmap")

        plt.savefig(
            "outputs/heatmap.png"
        )

        plt.show()

        # 3. BOXPLOT
        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df[
                [
                    'vibration_x',
                    'vibration_y',
                    'vibration_z'
                ]
            ]
        )

        plt.title("Vibration Boxplot")

        plt.savefig(
            "outputs/boxplot.png"
        )

        plt.show()

        print("Static Graphs Generated!")

    except Exception as e:

        print("Visualization Error:", e)


# ANIMATED GRAPHS
def animated_graphs(df):

    try:
        import plotly.express as px

        # OPTIMIZATION

        # Avoid modifying original dataframe
        data = df.copy()

        # Reduce total animation frames
        # (larger number = fewer frames = faster)
        data['frame'] = data.index // 50

        # Keep only necessary columns
        line_df = data[
            [
                'timestamp',
                'vibration_x',
                'vibration_y',
                'vibration_z',
                'frame'
            ]
        ]

        scatter_df = data[
            [
                'gearbox_bearing_temp',
                'vibration_x',
                'anomaly_label',
                'frame'
            ]
        ]

        # INTERACTIVE SCROLLABLE GEARBOX MONITORING

        import plotly.graph_objects as go

        # Prepare data
        line_data = df.copy()

        line_data['timestamp'] = pd.to_datetime(
            line_data['timestamp']
        )

        line_data = line_data.sort_values('timestamp')

        # PERFORMANCE OPTIMIZATION
        # 'D' = Daily
        # 'W' = Weekly
        # 'M' = Monthly

        line_data = line_data.set_index('timestamp')

        line_data = line_data.resample('W').mean(
            numeric_only=True
        )

        line_data.reset_index(inplace=True)

        # CREATE INTERACTIVE GRAPH

        fig1 = go.Figure()

        # Vibration X
        fig1.add_trace(
            go.Scatter(
                x=line_data['timestamp'],
                y=line_data['vibration_x'],
                mode='lines',
                name='Vibration X'
            )
        )

        # Vibration Y
        fig1.add_trace(
            go.Scatter(
                x=line_data['timestamp'],
                y=line_data['vibration_y'],
                mode='lines',
                name='Vibration Y'
            )
        )

        # Vibration Z
        fig1.add_trace(
            go.Scatter(
                x=line_data['timestamp'],
                y=line_data['vibration_z'],
                mode='lines',
                name='Vibration Z'
            )
        )

        # RANGE SLIDER + TIME BUTTONS

        fig1.update_layout(

            title='Interactive Gearbox Monitoring',

            template='plotly_dark',

            hovermode='x unified',

            height=700,

            xaxis=dict(

                title='Timestamp',

                rangeslider=dict(
                    visible=True
                ),

                rangeselector=dict(

                    buttons=list([

                        dict(
                            count=7,
                            label="1W",
                            step="day",
                            stepmode="backward"
                        ),

                        dict(
                            count=1,
                            label="1M",
                            step="month",
                            stepmode="backward"
                        ),

                        dict(
                            count=6,
                            label="6M",
                            step="month",
                            stepmode="backward"
                        ),

                        dict(
                            count=1,
                            label="1Y",
                            step="year",
                            stepmode="backward"
                        ),

                        dict(
                            step="all",
                            label="ALL"
                        )

                    ])
                )
            ),

            yaxis=dict(
                title='Vibration'
            )
        )

        # Save interactive graph
        fig1.write_html(
            "outputs/interactive_gearbox_monitoring.html"
        )

        fig1.show()
        
        # FAST ANIMATED SCATTER GRAPH

        fig2 = px.scatter(
            scatter_df,
            x='gearbox_bearing_temp',
            y='vibration_x',
            color='anomaly_label',
            animation_frame='frame',
            title='Animated Anomaly Detection',
            template='plotly_dark'
        )

        # Smaller markers = faster rendering
        fig2.update_traces(marker=dict(size=6))

        # Disable transitions
        fig2.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0
        fig2.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 0

        fig2.show()

        print("Fast Animated Graphs Loaded!")

    except Exception as e:

        print("Animation Error:", e)
# MAIN PROGRAM

def main():

    # CHANGE FILE PATH HERE
    file_path = "data/dataset_original.csv"

    # LOAD
    df = load_data(file_path)

    if df is not None:

        # CLEAN
        df = clean_data(df)

        # STATISTICS TABLE
        generate_statistics_table(df)

        # ENGINEERING INTERPRETATION
        engineering_interpretation(df)

        # CORRELATION
        correlation_analysis(df)

        # COMPARATIVE ANALYSIS
        comparative_analysis(df)

        # ANOMALY DETECTION
        df = anomaly_detection(df)

        # STATIC GRAPHS
        static_graphs(df)

        # ANIMATED GRAPHS
        animated_graphs(df)

        print("\nALL TASKS COMPLETED!")
        print("Check the OUTPUTS folder.")

# RUN PROGRAM

if __name__ == "__main__":

    main()