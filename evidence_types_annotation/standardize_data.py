from concurrent.futures import thread
import pandas as pd

def select_sentences(csvfile, thread_ids, nr=130):
    df = pd.read_csv(csvfile)
    threads = []
    for thread_id in thread_ids:
        if thread_id == 't3_6tsx1p':
            # one annotater had 129 sentences in this thread
            threads.append(df[df['thread_id']==thread_id][:129])
            print(len(df[df['thread_id']==thread_id][:129]))
        else:
            threads.append(df[df['thread_id']==thread_id][:nr])
            print(len(df[df['thread_id']==thread_id][:nr]))

    return pd.concat(threads)

def main():
    df = pd.read_csv('marieke.csv')
    thread_ids = list(set(df['thread_id'].tolist()))
    
    select_sentences('original_annotations/marieke.csv', thread_ids).to_csv('marieke_s.csv', index=False)
    select_sentences('original_annotations/friso.csv', thread_ids).to_csv('friso_s.csv', index=False)
    select_sentences('original_annotations/robin.csv', thread_ids).to_csv('robin_s.csv', index=False)

main()