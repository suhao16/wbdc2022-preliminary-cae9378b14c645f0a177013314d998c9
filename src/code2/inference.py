import torch
from torch.utils.data import SequentialSampler, DataLoader
import pickle
from config import parse_args
from dataset.data_helper import MultiModalDataset
from dataset.category_id_map import lv2id_to_category_id
from models.singlestream_model import SingleStream_model
from fintue_model_roberta_fusion import MultiModal
from config_f_pgd import parse_args as parse_args1
import numpy
def inference():
    args = parse_args()
    args1=parse_args1
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
    # model = MultiModal(args)
#     a=np.load("save/result_epoch2.npy",allow_pickle=True)
    target=[]
    data=['autodl-tmp/model_epoch_2_mean_f1_0.8806.bin','autodl-tmp/model_epoch_2_mean_f1_0.8564.bin','autodl-tmp/model_epoch_2_mean_f1_0.8775.bin','autodl-tmp/model_epoch_2_mean_f1_0.8706.bin','autodl-tmp/model_epoch_2_mean_f1_0.8774.bin','autodl-tmp/model_epoch_2_mean_f1_0.8709.bin','autodl-tmp/model_epoch_2_mean_f1_0.8753.bin','autodl-tmp/model_epoch_2_mean_f1_0.8758.bin','autodl-tmp/model_epoch_2_mean_f1_0.8776.bin','autodl-tmp/model_epoch_2_mean_f1_0.8785.bin','autodl-tmp/model_epoch_2_mean_f1_0.87531.bin','autodl-tmp/model_epoch_2_mean_f1_0.8722.bin']
    
    
    
    
    
    model0 = SingleStream_model(args)
    model1 = SingleStream_model(args)
    model2 = SingleStream_model(args)
    model3 = SingleStream_model(args)
    model4 = SingleStream_model(args)
    model5 = SingleStream_model(args)
    model6 = SingleStream_model(args)
    model7 = SingleStream_model(args)
    model8 = SingleStream_model(args)
#     model9 = SingleStream_model(args)
#     model10 = SingleStream_model(args)
#     model11 = SingleStream_model(args)
    
    

    checkpoint = torch.load(data[0], map_location='cpu')
    model0.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[1], map_location='cpu')
    model1.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[2], map_location='cpu')
    model2.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[3], map_location='cpu')
    model3.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[4], map_location='cpu')
    model4.load_state_dict(checkpoint['model_state_dict'])

    
    checkpoint = torch.load(data[5], map_location='cpu')
    model5.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[6], map_location='cpu')
    model6.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[7], map_location='cpu')
    model7.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(data[8], map_location='cpu')
    model8.load_state_dict(checkpoint['model_state_dict'])
    
#     checkpoint = torch.load(data[9], map_location='cpu')
#     model9.load_state_dict(checkpoint['model_state_dict'])
#     checkpoint = torch.load(data[10], map_location='cpu')
#     model10.load_state_dict(checkpoint['model_state_dict'])
#     checkpoint = torch.load(data[11], map_location='cpu')
#     model11.load_state_dict(checkpoint['model_state_dict'])
    
    
    
    if torch.cuda.is_available():
        model0 = torch.nn.parallel.DataParallel(model0.cuda())
        model1 = torch.nn.parallel.DataParallel(model1.cuda())
        model2 = torch.nn.parallel.DataParallel(model2.cuda())
        model3 = torch.nn.parallel.DataParallel(model3.cuda())
        model4 = torch.nn.parallel.DataParallel(model4.cuda())
        model5 = torch.nn.parallel.DataParallel(model5.cuda())
        model6 = torch.nn.parallel.DataParallel(model6.cuda())
        model7 = torch.nn.parallel.DataParallel(model7.cuda())
        model8 = torch.nn.parallel.DataParallel(model8.cuda())
#         model9 = torch.nn.parallel.DataParallel(model9.cuda())
#         model10 = torch.nn.parallel.DataParallel(model10.cuda())
#         model11 = torch.nn.parallel.DataParallel(model11.cuda())
        
        
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    model8.eval()
#     model9.eval()
#     model10.eval()
#     model11.eval()
    # 3. inference
    w=1
    predictions = []
    p=[]
    re=[]
#     r=numpy.load ( 'without_argmax.npy',allow_pickle=True)
    xx=0
    with torch.no_grad():
        for batch in dataloader:
            pred0= model0(batch, inference=True)
            pred1= model1(batch, inference=True)
            pred2= model2(batch, inference=True)
            pred3= model3(batch, inference=True)
            pred4= model4(batch, inference=True)
            pred5= model5(batch, inference=True)
            pred6= model6(batch, inference=True)
            pred7= model7(batch, inference=True)
            pred8= model8(batch, inference=True)
#             pred9= model9(batch, inference=True)
#             pred10= model10(batch, inference=True)
#             pred11= model11(batch, inference=True)
            
            pred =(pred0+pred1+pred2+pred3+pred4+pred5+pred6+pred7+pred8)/9
            re.append(pred.cpu().numpy())
            pred_label_id=torch.argmax(pred, dim=1)
            print(w)
            w=w+1
            predictions.extend(pred_label_id.cpu().numpy())
#     target.append(predictions)
#     with open('data'+str(i)+'.bin', 'wb') as f:
#         pickle.dump( predictions, f)
    predictions = numpy.array(predictions)
    numpy.save ( 'with_argmax.npy', predictions)
    re=numpy.array(re)
    numpy.save ( 'src/code1/save/12weight.npy',re)
    predictions=list(predictions)
# 4. dump results
    with open('result.csv', 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
