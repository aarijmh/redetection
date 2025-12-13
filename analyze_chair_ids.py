import pandas as pd

# Read the CSV file
df = pd.read_csv('office_final_exact.csv')

print('Unique chair IDs:', sorted(df['id'].unique()))
print('Number of unique chairs:', len(df['id'].unique()))
print('\nChair positions:')

for chair_id in sorted(df['id'].unique()):
    chair_data = df[df['id'] == chair_id]
    avg_x1 = chair_data['x1'].mean()
    avg_y1 = chair_data['y1'].mean()
    avg_x2 = chair_data['x2'].mean()
    avg_y2 = chair_data['y2'].mean()
    center_x = (avg_x1 + avg_x2) / 2
    center_y = (avg_y1 + avg_y2) / 2
    print(f'Chair {chair_id}: center ({center_x:.0f}, {center_y:.0f}), {len(chair_data)} detections')
