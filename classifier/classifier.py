# Written using a tutorial from curiousily.com

from concurrent.futures import thread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

import transformers
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class EvidenceTypeDataset(Dataset):

    def __init__(self, thread_id, comment_id, sentence, target, tokenizer, max_len):
        self.thread_id = thread_id
        self.comment_id = comment_id
        self.sentence = sentence
        self.targets = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = str(self.sentence[item])
        thread_id = str(self.thread_id[item])
        comment_id = str(self.comment_id[item])

        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return {
            'thread_id': thread_id,
            'comment_id': comment_id,
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets[item], dtype=torch.long)
        }


class EvidenceTypeClassifier(nn.Module):

    def __init__(self, num_classes):
        super(EvidenceTypeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def create_data_loader(df, tokenizer, max_len, batch_size):

    return DataLoader(
        EvidenceTypeDataset(
            thread_id=df.thread_id.to_numpy(),
            comment_id=df.comment_id.to_numpy(),
            sentence=df.sentence.to_numpy(),
            target=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        ),
        batch_size=batch_size,
        num_workers=4
    )


def train_model(model, data_loader, loss_function, optimizer, scheduler, num_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        targets = d['targets']

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / num_examples, np.mean(losses)


def evaluate_model(model, data_loader, loss_function, num_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            targets = d['targets']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / num_examples, np.mean(losses)


def find_best_model(model, loss_function, df_train, df_val, tokenizer, max_len, batch_size, num_epochs, print_graph=True, save_file_name='best_model_state.bin'):

    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False, no_deprecation_warning=True)

    total_steps = len(train_data_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 40)

        training_accuracy, training_loss = train_model(
            model,
            train_data_loader,
            loss_function,
            optimizer,
            scheduler,
            len(df_train)
        )

        print('Training:   loss {:.3f} - accuracy {:.3f}'.format(training_loss, training_accuracy))

        validation_accuracy, validation_loss = evaluate_model(
            model,
            val_data_loader,
            loss_function,
            len(df_val)
        )

        print('Validation: loss {:.3f} - accuracy {:.3f}\n'.format(validation_loss, validation_accuracy))

        history['training_accuracy'].append(training_accuracy)
        history['training_loss'].append(training_loss)

        history['validation_accuracy'].append(validation_accuracy)
        history['validation_loss'].append(validation_loss)

        if validation_accuracy > best_accuracy:
            torch.save(model.state_dict(), save_file_name)
            best_accuracy = validation_accuracy

    if print_graph:
        plt.plot(history['training_accuracy'], label='training accuracy')
        plt.plot(history['validation_accuracy'], label='validation accuracy')
        plt.title('Training history')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()


def get_predictions(model, data_loader, label_encoder, save_predictions=True, save_file_name='predictions.csv'):
    model = model.eval()
    thread_id_list = []
    comment_id_list = []
    sentence_texts = []
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            thread_ids = d['thread_id']
            comment_ids = d['comment_id']
            texts = d["sentence_text"]
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            thread_id_list.extend(thread_ids)
            comment_id_list.extend(comment_ids)
            sentence_texts.extend(texts)
            predictions.extend(preds)
            real_values.extend(targets)

    if save_predictions:
        df_pred = pd.DataFrame()
        df_pred['thread_id'] = thread_id_list
        df_pred['comment_id'] = comment_id_list
        df_pred['sentence'] = sentence_texts
        df_pred['pred_label'] = label_encoder.inverse_transform(predictions)
        df_pred['real_label'] = label_encoder.inverse_transform(real_values)

        df_pred.to_csv(save_file_name, index=False)

    return thread_id_list, comment_id_list, sentence_texts, predictions, real_values

def main():
    sns.set(style='whitegrid')
    sns.set_palette(sns.color_palette("rocket"))

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    df = pd.read_csv('data_10threads_no_continue.csv')

    # Transform labels to integers
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['gold_label_no_continue'])

    # # check if dataset is balanced
    # sns.countplot(df.gold_label_no_continue)
    # plt.show()

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # # Check max length
    # token_lens = []
    # for txt in df.sentence:
    #     tokens = tokenizer.encode(txt, max_length=512)
    #     token_lens.append(len(tokens))

    # sns.distplot(token_lens)
    # plt.show()

    num_classes = len(df['label'].unique())
    loss_function = nn.CrossEntropyLoss()

    max_len = 80
    batch_size = 16
    num_epochs = 5

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed)

    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    # model = EvidenceTypeClassifier(num_classes)
    # model = model
    # find_best_model(model, loss_function, df_train, df_val, tokenizer, max_len, batch_size, num_epochs)

    model = EvidenceTypeClassifier(num_classes)
    model.load_state_dict(torch.load('best_model.bin'))
    model = model

    thread_ids, comment_ids, y_sentence_texts, y_pred, y_test = get_predictions(
    model,
    test_data_loader,
    label_encoder,
    save_file_name='pred1.csv'
    )

    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()