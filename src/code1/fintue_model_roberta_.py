#vlbert2
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,VisualBertPreTrainedModel,VisualBertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from category_id_map_ import CATEGORY_ID_LIST
class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        task = ['mlm','itm']#'mlm','itm'
        self.task=task
        model_path = args.bert_dir
        #print(uni_bert_cfg)
        NUM_CLASSES=200
        weidu = 768
        self.video_fc = torch.nn.Linear(768, weidu)
        self.gelu = nn.GELU()
        self.Bert = BertModel.from_pretrained(args.bert_dir)
        self.classifier = nn.Linear(weidu * 2, len(CATEGORY_ID_LIST))  # 212
        uni_bert_cfg = self.Bert.config
        if 'tag' in task:
            # self.newfc_tag = torch.nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
            self.newfc_tag = torch.nn.Linear(weidu, NUM_CLASSES)

        if 'mlm' in task:

            self.cls = BertOnlyMLMHead(uni_bert_cfg)
            self.lm = MaskLM(tokenizer_path=model_path)
            self.num_class = NUM_CLASSES
            self.vocab_size = uni_bert_cfg.vocab_size

        '''if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)'''

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(weidu, 1)
        #self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
    def forward(self, inputs, inference=False,target = None,return_mlm = False):
        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}


        loss, pred, pred_tag = 0, None, None
        # sample_task=self.task
        sample_task=[]
        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']
        # position_ids = inputs['position_ids'],
        token_type_ids = inputs['token_type_ids'],
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu().long())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)  # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True

        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)

        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)


        text_emb = self.Bert.embeddings(input_ids=inputs['title_input'])
        #text input is [CLS][SEP] t e x t [SEP]
        '''cls_emb = text_emb[:, 0:1, :]  # batch * 单词 * 维度
        text_emb = text_emb[:, 1:, :]
        text_mask = inputs['title_mask']
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]'''
        video_feature = inputs['frame_input']
        frame_mask = inputs['frame_mask']
        video_feature = self.Bert.embeddings(inputs_embeds=self.gelu(self.video_fc(video_feature)))
        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([video_feature, text_emb], 1)
        mask = torch.cat([frame_mask,inputs['title_mask']], 1)
        mask0 = mask[:, None, None, :]
        mask0 = (1.0 - mask0) * -10000.0  # embedding_output torch.Size([2, 160=128+32, 312]) torch.Size([2, 1, 1, 160]) 没有映射的没有embedding
        encoder_outputs= self.Bert.encoder(embedding_output,mask0)['last_hidden_state'] # torch.Size([2, 332, 768])  BaseModelOutput(last_hidden_state=tensor([[[-0.0577, -0.3963, -0.3105,
        pooler_out=encoder_outputs
        #print(out, out.shape, pooler_out.shape, lm_prediction_scores.shape)  # torch.Size([212, 768])
        features=pooler_out
        embed_mean=(pooler_out*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1)
        embed_mean=embed_mean.float()
        embed_max=pooler_out+(1-mask).unsqueeze(-1)*(-1e10)
        embed_max=embed_max.max(1)[0].float()
        pooler_out = torch.cat([embed_mean, embed_max], -1) #1 torch.Size([2, 2, 160, 1536])
        #prediction = self.classifier(pooler_out)
        #pooler_out = torch.mean(pooler_out, 1)
        #print(pooler_out.shape)  #
        prediction = self.classifier(pooler_out)
        # compute pretrain task loss
        if 'mlm' in sample_task:
            lm_prediction_scores=self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :]
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss / 1.25 / len(sample_task)

        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input,
                                                     video_mask, video_label, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)

        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / 100 / len(sample_task)

        if 'tag' in sample_task:
            pred_tag = self.newfc_tag(torch.relu(features[:, 0, :]))
            if target is not None:
                tagloss = nn.BCEWithLogitsLoss(reduction="mean")(pred_tag.view(-1), target.view(-1)) / len(sample_task)
                loss += tagloss * 1250
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(loss,prediction, inputs['label'])

    @staticmethod
    def cal_loss(loss, prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label) + loss  # 预训练 ： 1注释此处 2 'mlm','mfm','itm','tag'
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    # calc mfm loss
    def calculate_mfm_loss(self, video_feature_output, video_feature_input,
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(nn.Module):
    def __init__(self,args, config):
        super().__init__()
        self.bert = UniBert(args,config)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask,token_type_ids, gather_index=None, return_mlm=False):
        mask0,encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask,token_type_ids)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :],mask0
        else:
            return encoder_outputs, None,mask0


class UniBert(nn.Module):
    def __init__(self, args,config):
        super().__init__()
        self.config = config

        #self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, 768)
        #self.video_embeddings = BertEmbeddings(config)
        #self.encoder = BertEncoder(config)
        self.bert=BertModel.from_pretrained(args)
        self.gelu=nn.GELU()
        # self.init_weights()
        self.pos_encoder_src0 = PositionalEncoding(d_model=768)
        self.pos_encoder_src = LearnedPositionEncoding(d_model=768)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask,token_type_ids, gather_index=None):
        #print(token_type_ids)
        text_emb = self.bert.embeddings(input_ids=text_input_ids,token_type_ids=token_type_ids[0])
        #print(text_emb.shape)
        #text_emb=self.pos_encoder_src(text_emb) #第一种位置id
        #text_emb = self.pos_encoder_src0(text_emb)  # 第一种位置id
        # text input is [CLS][SEP] t e x t [SEP]
        '''cls_emb = text_emb[:, 0:1, :]  # batch * 单词 * 维度
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]'''

        # reduce frame feature dimensions : 1536 -> 1024
        # print('video_feature',video_feature.shape)  # torch.Size([2, 32, 1536]) 相当于文本里的32个单词 每个单词1536维
        video_emb = self.video_fc(video_feature)
        video_feature=self.gelu(video_emb)
        # print('video_feature1', video_feature.shape)  # torch.Size([2, 32, 768])
        video_emb = self.bert.embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([video_emb, text_emb], 1)
        #print(embedding_output.shape) torch.Size([2, 32+200, 768])
        mask0 = torch.cat([video_mask, text_mask], 1)
        mask = mask0[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        out = self.bert.encoder(embedding_output,mask)['last_hidden_state']

        return mask0,out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        #print(x.shape,weight[:x.size(0),:].shape)  # torch.Size([2, 384, 768]) torch.Size([2, 1, 768])
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class MaskLM(object):
    def __init__(self, tokenizer_path='bert-base-chinese', mlm_probability=0.15):
        self.mlm_probability = 0.15
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class MaskVideo(object):
    def __init__(self, mlm_probability=0.15):
        self.mlm_probability = 0.15

    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask

        masked_indices = torch.bernoulli(probability_matrix).bool()

        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1,
                                                                                              video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2))

        return inputs, video_labels_index

class ShuffleVideo(object):
    def __init__(self):
        pass

    def torch_shuf_video(self, video_feature):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        return video_feature, label





