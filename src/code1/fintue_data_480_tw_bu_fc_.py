#首尾截断 + 补
import json
import random
import zipfile
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
from category_id_map_ import category_id_to_lv2id ,CATEGORY_ID_LIST
#from DeBERTa import deberta
import json
import jieba.analyse

def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    '''train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))'''
    ss= StratifiedShuffleSplit(n_splits=1,test_size=0.1,train_size=0.9,
                                                               random_state=args.seed)
    a,b=[],[]
    #print(dataset.label())
    for train_idx, val_idx in ss.split(dataset,dataset.label()):
        a=train_idx
        b=val_idx

    train_dataset= MultiModalDataset(args, args.train_annotation, args.train_zip_feats,idxx=a)
    val_dataset=MultiModalDataset(args, args.train_annotation, args.train_zip_feats,idxx=b)
    #train_sampler = RandomSampler(dataset.idxx(x=a))
    #val_sampler = SequentialSampler(dataset.idxx(x=b))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  #sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                #sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False,idxx:list=[],):
        self.idxx=idxx
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]

        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.a=[]
        if not self.test_mode:
            for i in self.anns:
                self.a.append(i['category_id'])
        #
        '''self.keywords = jieba.analyse.extract_tags((self.anns), topK=50,
                                              withWeight=True,
                                              allowPOS=(['n', 'v']))  # 只提取前10个名词和动词'''
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        #self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        # deberta tokenizer
        #vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
        #self.tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    def label(self):
        return self.a
    '''def idxx(self,x):
        self.idxx=x'''
    def __len__(self) -> int:
        if self.idxx==[]:
            length =len(self.anns)
        else:length=len(self.idxx)
        return length

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats  # 填充补零
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        #encoded_inputs = self.tokenizer(text, max_length=300, padding='max_length', truncation=True)
        encoded_inputs = self.tokenizer(text, max_length=480, padding='max_length', truncation=True,return_attention_mask = True)
        '''tokens = self.tokenizer.tokenize(text)
        max_seq_len = 400
        tokens = tokens[:max_seq_len - 2]
        # Add special tokens to the `tokens`
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # padding
        paddings = max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * paddings
        input_mask = input_mask + [0] * paddings'''
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        #input_ids=torch.tensor(input_ids, dtype=torch.int),
        #input_mask= torch.tensor(input_mask, dtype=torch.int)
        #position_ids=np.array(encoded_inputs['position_ids'])
        token_type_ids=np.array(encoded_inputs['token_type_ids'])
        return input_ids, mask,token_type_ids

    def __getitem__(self, idx: int) -> dict:
        if self.idxx==[]:
            idx =idx
        else:idx=self.idxx[idx]

        # Step 1, loa5d visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title ocr asr  tokens
        a=[]
        for i in self.anns[idx]['ocr']:
            a.extend(i['text'])
        len_0 = 60   # title
        len_1 = 260  # ocr
        len_2 = 160   # asr
        all_len = len_0 + len_1 + len_2
        self.length = all_len
        #title
        if len(self.anns[idx]['title'])>=len_0:
            all_text0=self.anns[idx]['title'][:len_0//2]+self.anns[idx]['title'][-len_0//2:]
        else:all_text0=self.anns[idx]['title']
        # ocr
        if len(str(a)) >= len_1:
            all_text1 = str(a)[:len_1//2] + str(a)[-len_1//2:]
        else:
            all_text1 = str(a)
        # asr
        if len(self.anns[idx]['asr']) >= len_2:
            all_text2 = self.anns[idx]['asr'][:len_2 // 2] + self.anns[idx]['asr'][-len_2 // 2:]
        else:
            all_text2 = self.anns[idx]['asr']
        #all
        #all_text = self.anns[idx]['title'] + str(a) + self.anns[idx]['asr']
        all_text = all_text0+all_text1+all_text2
        #print('gg',len(all_text))
       #补
        self.length=all_len
        if len(all_text)<=all_len and len(all_text0)>= len_0 and len(all_text1)>= len_1 and len(all_text2)>= len_2:
            all_text = all_text0 + all_text1 +all_text2+ self.anns[idx]['title'][len_0//2:-len_0//2]+str(a)[len_1//2:-len_1//2]+self.anns[idx]['asr'][len_2//2:-len_2//2]
        elif len(all_text)<=all_len and len(all_text0)>= len_0 and len(all_text1)>= len_1:
            all_text = all_text0 + all_text1 + all_text2+self.anns[idx]['title'][len_0//2:-len_0//2]+str(a)[len_1//2:-len_1//2]
        elif len(all_text)<=all_len and len(all_text0)>= len_0 and len(all_text2)>= len_2:
            all_text = all_text0 + all_text1 + all_text2+self.anns[idx]['title'][len_0//2:-len_0//2]+self.anns[idx]['asr'][len_2//2:-len_2//2]
        elif len(all_text)<=all_len and len(all_text1)>= len_1 and len(all_text2)>= len_2:
            all_text = all_text0 + all_text1 + all_text2+str(a)[len_1//2:-len_1//2]+self.anns[idx]['asr'][len_2//2:-len_2//2]
        elif len(all_text)<=all_len and len(all_text0)>= len_0 :
            all_text = all_text0 + all_text1 + all_text2+self.anns[idx]['title'][len_0//2:-len_0//2]
        elif len(all_text) <= all_len and len(all_text1) >= len_1:
            all_text = all_text0 + all_text1 + all_text2 +str(a)[len_1//2:-len_1//2]
        elif len(all_text)<=all_len and len(all_text2)>= len_2 :
            all_text = all_text0 + all_text1 + all_text2+ self.anns[idx]['asr'][len_2//2:-len_2//2]
        else:all_text =all_text
        if len(all_text)>=all_len:
            all_text=all_text[:all_len//2]+all_text[-all_len//2:]
        else:all_text=all_text
        #print(len(all_text))
        #print(all_text)
        # tfifdf
        keys = []
        '''
        # append(self.anns[idx]['title'] + self.anns[idx]['asr'] + str(a))
        keywords = jieba.analyse.extract_tags((self.anns[idx]['title'] + self.anns[idx]['asr'] + str(a)), topK=50,
                                              withWeight=True,
                                              allowPOS=(['n', 'v']))  # 只提取前10个名词和动词
    
        for j in keywords:
            keys.append(j[0])'''
        #print(self.keywords[idx])
        #title_input, title_mask,token_type_ids = self.tokenize_text(self.keywords[idx]) #all_text0+all_text1+all_text2
        #title_input, title_mask, token_type_ids = self.tokenize_text(self.anns[idx]['title'][:60]+self.anns[idx]['asr'][:60]+str(a)[:60])
        title_input, title_mask, token_type_ids = self.tokenize_text(all_text)
        #keys_input, keys_mask,_= self.tokenize_text(str(keys))
        #print(len(title_input),len(title_mask),len(token_type_ids))
        #print(title_input.type)
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            #position=position,
            #keys_input=keys_input,
            #keys_mask=keys_mask,
            token_type_ids=token_type_ids,
        )
        #print('|',frame_input.shape,frame_mask.shape,title_input.shape,title_mask.shape,'|')
        #| torch.Size([32, 768]) torch.Size([32]) (50,) (50,) |
        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
        # pretrain
        '''if not self.test_mode:
            label = category_id_to_lv2id0(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])'''

        return data
