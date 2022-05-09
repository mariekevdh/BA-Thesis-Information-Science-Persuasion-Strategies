# Adds gold labels to annotations
# If at least two annotators agree = gold standard
# Else gold label will be left empy so it can be added by hand.

import pandas as pd


df_marieke = pd.read_csv('standardized_annotations/marieke_s.csv')
df_friso = pd.read_csv('standardized_annotations/friso_s.csv')
df_robin = pd.read_csv('standardized_annotations/robin_s.csv')

df_merged = pd.concat([df_marieke['thread_id'],
                       df_marieke['comment_id'],
                       df_marieke['sentence'],
                       df_marieke['evidence'],
                       df_friso['evidence'],
                       df_robin['evidence']],
                       axis=1)
df_merged['gold_label'] = ""
df_merged.columns = ['thread_id', 'comment_id', 'sentence', 'a1', 'a2', 'a3', 'gold_label']

for index, row in df_merged.iterrows():
    gold_label = ""
    a1 = row['a1']
    a2 = row['a2']
    a3 = row['a3']
    if a1 == 'Common ground':
        a1 = 'Assumption'
    if a2 == 'Common ground':
        a2 = 'Assumption'
    if a3 == 'Common ground':
        a3 = 'Assumption'
    if a1 == a2 or a1 == a3:
        gold_label = a1
    elif a2 == a3:
        gold_label = a2
    df_merged.at[index, 'gold_label'] = gold_label

df_merged.to_csv('10threads.csv', index=False)