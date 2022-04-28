# This program filters out data from the 2020 Webis ChangeMyView Corpus.
# It selects threads that have at least 50 first-level comments,
# (and after filtering comments at least 35)
# is created in 2016 or later and has received at least one delta.
# For these threads it filters out comments that have 30-300 words
# and are not written by a moderator or bot.
# The selected threads are saved in a jsonl file called 'filtered_threads.jsonl'

import jsonlines


def clean_up_meta_data(discussion, filtered_comments):
    """Filters out non-relevant meta data"""

    clean_comments = [{'body': comment['body'],
                       'author': comment['author'],
                       'controversiality': comment['controversiality'],
                       'score': comment['score'],
                       'created_utc': comment['created_utc'],
                       'urls': comment['urls'],
                       'id': comment['id']}
                      for comment in filtered_comments]

    clean_discussion = {'title': discussion['title'],
                        'comments': clean_comments,
                        'score': discussion['score'],
                        'created_utc': int(discussion['created_utc']),
                        'author': discussion['author'],
                        'url': discussion['url'],
                        'id': discussion['id']}

    return clean_discussion


def check_and_select(discussion):
    """checks criteria on discussion and comment level,
    and selects comments that meet criteria"""

    filtered_comments = []

    # filter out archived / banned threads
    if 'created_utc' in discussion.keys():
        if (
            # has at least 50 first level comments
            (len(discussion['comments']) > 50)
            # was created in 2016 or later (unix time)
            and (int(discussion['created_utc']) >= 1451602800)
            # has received at least one delta
            and discussion['delta']
        ):
            for comment in discussion['comments']:
                comment_length = len(comment['body'].split())
                if (
                    # length between 30-300 words
                    (comment_length >= 30 and comment_length <= 300)
                    # is not written by a moderator
                    and (comment['distinguished'] != 'moderator')
                    # is not a written by a bot
                    and not any(x in comment['body'] for x in ["I'm a bot", '#bot'])
                ):
                    filtered_comments.append(comment)

    # has at least 35 comments after filtering
    if len(filtered_comments) >= 35:
        return clean_up_meta_data(discussion, filtered_comments)


def main():
    discussion_count = 0
    comment_count = 0
    word_count = 0
    filtered_discussions = []
    with jsonlines.open('threads.jsonl') as reader:
        for discussion in reader:
            if discussion_count < 100:
                checked_discussion = check_and_select(discussion)
                if checked_discussion:
                    filtered_discussions.append(checked_discussion)
                    discussion_count += 1
                    # for printing averages
                    comment_count += len(checked_discussion['comments'])
                    word_count += sum([len(comment['body'].split()) for comment
                                      in checked_discussion['comments']])
            else:
                break

    with jsonlines.open('filtered_threads.jsonl', mode='w') as writer:
        writer.write_all(filtered_discussions)

    print('done.')
    print('nr of threads found: {}'.format(discussion_count))
    print('average nr of comments per thread: {}'.format(comment_count//discussion_count))
    print('average nr of words per comment: {}'.format(word_count//comment_count))


if __name__ == "__main__":
    main()
