import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertForSequenceClassification

class Bert_dataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    def __getitem__(self,index):
        # get the sentence from the dataframe
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = 128,
            pad_to_max_length= True,
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        # These are torch tensors already
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        #Convert the target to a torch tensor
        target = torch.tensor(self.df.loc[index, 'label'], dtype=torch.int64)

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample

    def __len__(self):
        return len(self.df)
