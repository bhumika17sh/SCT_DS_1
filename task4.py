import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

# Load the dataset
data = pd.read_csv('/Users/bhumikahiremath/documents/IdeaProjects/task4/data.csv')

# Data Preprocessing: Convert relevant columns to datetime and categorize them
data['Accident_Time'] = pd.to_datetime(data['Time'], format='%H:%M', errors='coerce').dt.hour
data['Road_Conditions'] = data['Road_Surface_Conditions'].astype('category')
data['Weather_Conditions'] = data['Weather_Conditions'].astype('category')

# Check for missing values
print(data.isnull().sum())

# Analyze Accidents by Time of Day
accident_time_dist = data.groupby('Accident_Time').size()
plt.figure(figsize=(10,6))
sns.barplot(x=accident_time_dist.index, y=accident_time_dist.values, palette='viridis', hue=None)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(range(24))  # Show all hour ticks
plt.grid(axis='y')
plt.show()

# Visualize weather-related accidents
plt.figure(figsize=(10,6))
sns.countplot(x='Weather_Conditions', data=data, order=data['Weather_Conditions'].value_counts().index, palette='magma', hue=None)
plt.title('Accidents by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualize road-related accidents
plt.figure(figsize=(10,6))
sns.countplot(x='Road_Conditions', data=data, order=data['Road_Conditions'].value_counts().index, palette='plasma', hue=None)
plt.title('Accidents by Road Conditions')
plt.xlabel('Road Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Create a heatmap of accident locations
if data['Latitude'].notnull().any() and data['Longitude'].notnull().any():
    # Filter valid latitude and longitude data
    valid_data = data.dropna(subset=['Latitude', 'Longitude'])

    if not valid_data.empty:
        accidents_map = folium.Map(location=[valid_data['Latitude'].mean(), valid_data['Longitude'].mean()], zoom_start=12)
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in valid_data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(accidents_map)
        accidents_map.save('accident_hotspots.html')
        print("Heatmap saved as 'accident_hotspots.html'")
    else:
        print("No valid accident data for the heatmap.")
else:
    print("No valid latitude or longitude data available.")

# Analyze accident severity based on road and weather conditions
plt.figure(figsize=(10,6))
sns.boxplot(x='Weather_Conditions', y='Accident_Severity', data=data, palette='coolwarm', hue=None)
plt.title('Accident Severity by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Road_Conditions', y='Accident_Severity', data=data, palette='coolwarm', hue=None)
plt.title('Accident Severity by Road Conditions')
plt.xlabel('Road Conditions')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

print(data.columns)
