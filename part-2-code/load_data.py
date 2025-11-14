import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small', legacy=False)

        encoder_inputs, decoder_labels = self.process_data(data_folder, split, self.tokenizer)

        self.encoder_inputs = encoder_inputs
        self.decoder_labels = decoder_labels
        self.bos_token_id = self.tokenizer.pad_token_id
        
        print(f"Dataset initialized: {len(self.encoder_inputs)} examples")
        
    def process_data(self, data_folder, split, tokenizer):
        # TODO
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]

        sql_path = os.path.join(data_folder, f'{split}.sql')
        if os.path.exists(sql_path):
            with open(sql_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f]
            assert len(questions) == len(queries), \
                f"Mismatch: {len(questions)} questions vs {len(queries)} queries in {split} set"
            has_labels = True
        else:
            print(f"No SQL file found for {split} set (this is expected for test set)")
            queries = [""] * len(questions)
            has_labels = False
        
        print(f"Loading {len(questions)} examples from {split} set")
        
        encoder_inputs = []
        decoder_labels = []
        
        for question, sql in tqdm(zip(questions, queries), total=len(questions), desc=f"Processing {split}"):
            input_text = f"translate English to SQL: {question}"
            
            # Tokenize encoder input
            encoder_input = tokenizer(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            if has_labels and sql:
                decoder_output = tokenizer(
                    sql,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                labels = decoder_output['input_ids'].squeeze(0).clone()
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = None
            
            encoder_inputs.append({
                'input_ids': encoder_input['input_ids'].squeeze(0),
                'attention_mask': encoder_input['attention_mask'].squeeze(0)
            })
            
            decoder_labels.append(labels)
        
        print(f"Finished processing {split} set: {len(encoder_inputs)} examples")
        return encoder_inputs, decoder_labels
        
    def __len__(self):
        # TODO
        return len(self.encoder_inputs)
    def __getitem__(self, idx):
        # TODO
        item = {
            'encoder_input': self.encoder_inputs[idx]['input_ids'],
            'encoder_mask': self.encoder_inputs[idx]['attention_mask'],
            'bos_token_id': self.bos_token_id,
        }
        
        if self.decoder_labels[idx] is not None:
            item['labels'] = self.decoder_labels[idx]
        
        return item

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_inputs = torch.stack([item['encoder_input'] for item in batch])
    encoder_masks = torch.stack([item['encoder_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    

    return encoder_inputs, encoder_masks, labels
def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    from transformers import T5Tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small', legacy=False)
    
    bos_token_id = tokenizer.pad_token_id  

    encoder_input = torch.stack([item['encoder_input'] for item in batch])
    encoder_mask = torch.stack([item['encoder_mask'] for item in batch])

    batch_size = encoder_input.shape[0]
    initial_decoder_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
    
    return encoder_input, encoder_mask, initial_decoder_input

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x
    
