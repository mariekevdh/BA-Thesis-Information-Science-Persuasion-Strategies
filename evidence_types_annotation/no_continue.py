# Replaces continue labels with previous label

import pandas as pd

def add_no_continue(csvfile):
    df = pd.read_csv(csvfile)
    df['gold_label_no_continue'] = df['gold_label']
    prev_label = df['gold_label'][0]

    for index, row in df.iterrows():
        current_label = row['gold_label']
        new_label = None
        if current_label == 'Continue':
            new_label = prev_label
        if new_label:
            df.at[index, 'gold_label_no_continue'] = new_label
            prev_label = new_label
        else:
            prev_label = current_label
    return df


def main():
    df = add_no_continue('merged_annotations_gold.csv')

    df.to_csv('merged_annotations_gold_no_continue.csv', index=False)


main()