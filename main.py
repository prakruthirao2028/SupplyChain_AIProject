
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Suppy_Chain_Shipment_Data.csv')

##wanted columns
## removes the unecessary columns filtered through excel sheets and analyzing kaggle parameters

columns_used = [
    'country',
    'fulfill via',
    'shipment mode',
    'scheduled delivery date',
    'delivered to client date',
    'vendor',
    'dosage form',
    'manufacturing site'
]
## initial cleaning process

##uses wanted columns only, drops all others

df_filtered = df[columns_used].copy()
## converts to datetime for math processing

df_filtered['scheduled delivery date'] = pd.to_datetime(df_filtered['scheduled delivery date'])
df_filtered['delivered to client date'] = pd.to_datetime(df_filtered['delivered to client date'])

## now making new column with boolean condition for the above ^

df_filtered["delivered late"] = df_filtered["delivered to client date"] > df['scheduled delivery date']

##new csv for filtered data
## syntax things: removing index prevents row index # from becoming an extra column within new dataset
df_filtered.to_csv('Filtered_Shipment_Date.csv', index = False)

print(df_filtered.head())