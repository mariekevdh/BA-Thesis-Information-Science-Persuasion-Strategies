import pandas as pd
import statsmodels
from statsmodels.stats.inter_rater import fleiss_kappa


def get_labels(csvfiles,
               keep_continue=True,
               keep_common_ground=True,
               keep_definition=True,
               keep_other=True):
    # Create empty dataframe to store labels in
    label_frame = pd.DataFrame()
    for i, file in enumerate(csvfiles, start=1):
        df = pd.read_csv(file)
        prev_label = df['evidence'][0]
        for index, row in df.iterrows():
            current_label = row['evidence']
            new_label = None
            if not keep_continue:
                if current_label == 'Continue':
                    new_label = prev_label
            if not keep_common_ground:
                if current_label == 'Common ground':
                    new_label = 'Assumption'
            if not keep_definition:
                if current_label == 'Definition':
                    new_label = 'Anecdote'
            if not keep_other:
                if current_label == 'Other':
                    new_label = 'None'
            if new_label:
                df.at[index, 'evidence'] = new_label
                prev_label = new_label
            else:
                prev_label = current_label
        label_frame['a' + str(i)] = df['evidence']
    return label_frame


def print_fleiss_kappa(df, details=True):
    # Get annotation labels
    labels = set()
    for c in df.columns:
        labels.update((df[c].unique()))

    # Add number of annotations per label per sentence
    for label in labels:
        df[label] = (df.iloc[:, 0:len(df.columns)] == label).sum(axis=1)

    # Select only these newly added counts
    labels_iaa = df.iloc[:, 3:]

    print('Fleiss kappa: {}'.format(round(fleiss_kappa(labels_iaa), 3)))

    if details:
        print('\nPer label:')
        print('{:>26s}'.format('Kappa'))
        for i, _ in enumerate(labels):
            # Select label
            selected_label = labels_iaa.iloc[:, i]
            # Sum count of all other labels
            sum_other_labels = (labels_iaa.iloc[:, :]).sum(axis=1) - selected_label
            # Create new dataframe for calculating fleiss kappa
            sub_df = pd.concat([selected_label, sum_other_labels], axis=1)

            print('{:20s} {:.3f}'.format(selected_label.name, round(fleiss_kappa(sub_df), 3)))


def main():
    csvfiles = ['standardized_annotations/marieke_s.csv',
                'standardized_annotations/friso_s.csv',
                'standardized_annotations/robin_s.csv']

    print("Original labels:")
    print_fleiss_kappa(get_labels(csvfiles,
                                  keep_continue=True,
                                  keep_common_ground=True,
                                  keep_definition=True,
                                  keep_other=True))
    print("\n--------------------------")
    print("With 'Common ground' merged into 'Assumption':\n")
    print_fleiss_kappa(get_labels(csvfiles,
                                  keep_continue=True,
                                  keep_common_ground=False,
                                  keep_definition=True,
                                  keep_other=True))
    print("\n--------------------------")
    print("Previous changes + 'Continue' changed to actual label:\n")
    print_fleiss_kappa(get_labels(csvfiles,
                                  keep_continue=False,
                                  keep_common_ground=False,
                                  keep_definition=True,
                                  keep_other=True))
    print("\n--------------------------")
    print("Previous changes + 'Other' merged into 'None':\n")
    print_fleiss_kappa(get_labels(csvfiles,
                                  keep_continue=False,
                                  keep_common_ground=False,
                                  keep_definition=True,
                                  keep_other=False))
    print("\n--------------------------")
    print("Previous changes + 'Definition' merged into 'Anecdote':\n")
    print_fleiss_kappa(get_labels(csvfiles,
                                  keep_continue=False,
                                  keep_common_ground=False,
                                  keep_definition=False,
                                  keep_other=False))


if __name__ == "__main__":
    main()
