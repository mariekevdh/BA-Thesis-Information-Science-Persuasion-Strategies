from operator import index
import pandas as pd

def add_no_continue(df):
    df['evidence_no_continue'] = df['evidence']
    prev_label = df['evidence'][0]

    for index, row in df.iterrows():
        current_label = row['evidence']
        new_label = None
        if current_label == 'Continue':
            new_label = prev_label
        if new_label:
            df.at[index, 'evidence_no_continue'] = new_label
            prev_label = new_label
        else:
            prev_label = current_label
    return df

def main():

    first10 = pd.read_csv('merged_annotations_gold.csv')
    marieke = pd.read_csv('processed_annotations/marieke_final.csv')
    friso = pd.read_csv('processed_annotations/friso_final.csv')
    robin = pd.read_csv('processed_annotations/robin_final.csv')

    first10_final = first10[['sentence', 'gold_label', 'thread_id', 'comment_id']]
    first10_final.columns = ['sentence', 'evidence', 'thread_id', 'comment_id']
    marieke_final = marieke[['sentence', 'evidence', 'thread_id', 'comment_id']]
    friso_final = friso[['sentence', 'evidence', 'thread_id', 'comment_id']]
    robin_final = robin[['sentence', 'evidence', 'thread_id', 'comment_id']]

    final_dataset = pd.concat([first10_final, marieke_final, friso_final, robin_final])
    final_dataset.reset_index(inplace=True,drop=True)

    final_dataset_no_continue = add_no_continue(final_dataset)
    final_dataset_no_continue.to_csv('final_dataset.csv', index=False)

    print(final_dataset.shape)
    print(final_dataset.drop_duplicates(subset=['thread_id', 'sentence']).shape)

main()