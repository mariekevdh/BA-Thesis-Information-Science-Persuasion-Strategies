import pandas as pd
import json


def get_labels(csvfile):
    df = pd.read_csv(csvfile)
    prev_label = df['evidence'][0]

    for index, row in df.iterrows():
        current_label = row['evidence']
        new_label = None
        if 'choices' in str(current_label):
            current_label = json.loads(row['evidence'])['choices'][0]
        if current_label == 'Continue':
            new_label = prev_label
        if current_label == 'Common ground':
            new_label = 'Assumption'
        # if current_label == 'Definition':
        #     new_label = 'Anecdote'
        if current_label == 'Other':
            new_label = 'None'
        if new_label:
            df.at[index, 'evidence'] = new_label
            prev_label = new_label
        else:
            prev_label = current_label
    return df


def main():
    df1 = get_labels('standardized_annotations/marieke_s.csv')
    df2 = get_labels('standardized_annotations/friso_s.csv')
    df3 = get_labels('standardized_annotations/robin_s.csv')

    combined_df = pd.concat([df1['evidence'], df2['evidence'], df3['evidence']], axis=1)
    combined_df.columns = ['a1', 'a2', 'a3']
    combined_df.to_csv('merged_no_other.csv', index=False)


main()