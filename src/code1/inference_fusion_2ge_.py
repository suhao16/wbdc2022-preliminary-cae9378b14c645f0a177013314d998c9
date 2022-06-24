import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy
# from config_f_pgd import parse_args
from config_f_pgd_ import parse_args
# from data_helper import MultiModalDataset
# from pretrain_data_helper60260160_tw_bu_fc import MultiModalDataset  # 480  不分层
from fintue_data_480_tw_bu_fc_ import MultiModalDataset
# from fintue_data_480_tw_bu_fc import MultiModalDataset

# from data_helper60_fc import MultiModalDataset
from category_id_map_ import lv2id_to_category_id

from fintue_model_roberta_fusion_ import MultiModal
# from fintue_model_roberta_smart import MultiModal

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModal(args)
    # checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    checkpoint = torch.load(
        'src/code1/save/Best_EMA_PGD_480_bs32_lr5e-5_preepoch33_quan.bin',
        map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    model2 = MultiModal(args)
    # checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    checkpoint = torch.load(
        'src/code1/save/Best_EMA_PGD_480_bs32_lr5e-5_preepoch33.bin',
        map_location='cpu')
    model2.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model2 = torch.nn.parallel.DataParallel(model2.cuda())
    model2.eval()

    '''
    model3 = MultiModal(args)
    # checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    checkpoint = torch.load(
        '/home/test/liuluyao/wxy/match/dspfl/challenge-main/save/fintue_model/best_fintue_epoch2_quanliang_model.bin',
        map_location='cpu')
    model3.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model3 = torch.nn.parallel.DataParallel(model3.cuda())
    model3.eval()
    '''

    '''
    predictions_68 = numpy.load("save/fintue_model/without_argmax.npy", allow_pickle=True)  # 68.1373
    # predictions_68 = numpy.reshape(predictions_68,(-1,200))
    print(predictions_68.shape)   # (25000,)
    print(len(predictions_68))
    print(predictions_68[1].shape)

    predictions_677_256 = numpy.load("data/result_0.67_0.668_256.npy", allow_pickle=True)   # 67.7157
    print(predictions_677_256.shape)  # (25000,)
    print(len(predictions_677_256))
    print(predictions_677_256[1].shape)

    # pred_label_id_all = torch.argmax((predictions_68+predictions_677_256)/2, dim=1)
    # print(pred_label_id_all.shape)

    # predictions_677=numpy.load('data/result_0.67_0.668.csv.npy', allow_pickle=True)  # 保存为.npy格式
    # print(predictions_677.shape)  # (50000,)
    # print(len(predictions_677))
    # print(predictions_677[1].shape)

    # # predictions_68 = predictions_68.tolist()
    '''
    # # 3. inference
    predictions = []
    r=numpy.load("src/code1/save/12weight.npy", allow_pickle=True)
    print(len(r))
    print(r[1].shape)
    xx=0
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id,prediction = model(batch, inference=True)
            # print(prediction.shape) # [256, 200]  [bs,200]
            pred_label_id2,prediction2 = model2(batch, inference=True)
            # pred_label_id3, prediction3 = model2(batch, inference=True)
            
            pred_label_id_all =torch.argmax(((prediction.cpu())*0.25+(prediction2.cpu())*0.25+torch.tensor(r[xx]).cpu()*0.5), dim=1)
            xx=xx+1
            # pred_label_id_all =torch.argmax((prediction), dim=1)
            # pred_label_id_all =prediction*0.5+prediction2*0.5
            # pred_label_id = model(batch)
            predictions.extend(pred_label_id_all.cpu().numpy())
            # predictions.append(prediction.cpu().numpy())
    # torch.save(predictions, 'save/fintue_model/with_argmax.npy')

    # predictions_ = numpy.array(predictions)
    # numpy.save('data/resultB_2+9.npy', predictions_)  # 保存为.npy格式
    # print(predictions_.shape)
    # 读取

    # predictions = []
    # predictions_677_256 = numpy.load("data/result_0.67_0.668_256.npy", allow_pickle=True)  # 67.7157
    # predictions_681 = numpy.load("save/fintue_model/without_argmax.npy", allow_pickle=True)  # 68.1373
    # # for i in range(97):
    # # predictions.extend(torch.argmax(torch.tensor(predictions_681.astype(float)), dim=1))
    # predictions.extend(torch.argmax(torch.tensor(predictions_677_256.tolist()), dim=1))
    # # print(predictions.shape)

    # 4. dump results
    # with open(args.test_output_csv, 'w') as f:
    with open('data/result.csv', 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
