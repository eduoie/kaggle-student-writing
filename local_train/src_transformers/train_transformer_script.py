# Based on the code from: https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615

import sys
import subprocess
import argparse
import os
import logging
import pandas as pd
from ast import literal_eval

import torch
import gc
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# the supposed correct way is installing dependencies before, but some people commented it wasn't working.
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.15.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece==0.1.96"])
from transformers import AutoTokenizer
import transformers

LABEL_ALL_SUBTOKENS = True

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids  # for validation

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                                  is_split_into_words=True,
                                  # return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        word_ids = encoding.word_ids()

        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(labels_to_ids[word_labels[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)

        return item

    def __len__(self):
        return self.len


def train_epoch(epoch, model, optimizer, config, training_loader):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()

    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 10 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_labels.extend(labels)
        # tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 3 == 0:
            print('Stopping here!. This is slow (and expensive), is just used for demo purposes...')
            break

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def _train(args):
    print('received args: ', args)

    # TODO: this will be used to avoid downloading the whole model each time the training has to run
    preloaded = False
    if preloaded:
        # config_model = transformers.AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')
        # model = AutoModelForTokenClassification.from_pretrained(
        #                    DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)
        # model.to(config['device'])
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])
        pass
    else:
        pass

    config = {'model_name': args.model_name,
              'max_length': args.max_length,
              'train_batch_size': args.batch_size,
              'valid_batch_size': args.batch_size,
              'epochs': args.epochs,
              'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
              'max_grad_norm': 10,
              'device': 'cuda' if cuda.is_available() else 'cpu'}

    # prepare model for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    config_model = transformers.AutoConfig.from_pretrained(args.model_name)
    config_model.num_labels = 15
    model = transformers.AutoModelForTokenClassification.from_pretrained(args.model_name, config=config_model)

    model.to(config['device'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])

    # read and prepare data
    input_files = [file for file in os.listdir(args.data_dir) if file.endswith('.csv')]
    print('found training files:', input_files)

    train_dataset = pd.read_csv(os.path.join(args.data_dir, 'train_NER.csv'))
    valid_dataset = pd.read_csv(os.path.join(args.data_dir, 'valid_NER.csv'))

    # pandas saves lists as string, we must convert back
    train_dataset.entities = train_dataset.entities.apply(lambda x: literal_eval(x))
    valid_dataset.entities = valid_dataset.entities.apply(lambda x: literal_eval(x))

    training_set = dataset(train_dataset, tokenizer, args.max_length, False)
    valid_set = dataset(valid_dataset, tokenizer, args.max_length, True)

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    # TODO: temp, originally set to 2, but is a fix to a bug when testing the script outside the docker environment
                    'num_workers': 0,
                    'pin_memory': True
                    }

    valid_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 0,
                    'pin_memory': True
                    }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **valid_params)

    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups:
            g['lr'] = config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')

        train_epoch(epoch, model, optimizer, config, training_loader)
        torch.cuda.empty_cache()
        gc.collect()

    path = os.path.join(args.model_dir, 'bigbird_v1.pt')
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of total epochs to run (default: 1)')
    # NOTE: a batch size of 4 will require at least 20GB of RAM for the Docker image to avoid OutOfMemory errors
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='batch size (default: 2)')
    parser.add_argument('--max-length', type=int, default=1024, metavar='BS',
                        help='batch size (default: 1024)')
    parser.add_argument('--model-name', type=str, default='google/bigbird-roberta-base',
                        help='transformers model (default "google/bigbird-roberta-base")')

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--data-dir', type=str, default='./processed_data_transformer')


    _train(parser.parse_args())
