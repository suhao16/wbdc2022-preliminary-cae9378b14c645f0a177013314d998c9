import logging
import os
import time
import torch

from config import parse_args
#from data_helper_model7 import create_dataloaders
#from data_helper60260160_tw_fc import create_dataloaders
#from data_helper60_fc import create_dataloaders
#
from pretrain_data_helper60260160_tw_bu_fc import create_dataloaders
#from data_helper_tfidf import create_dataloaders
#from model_top1_fc_gelu_shareembedding_best1 import MultiModal
#from model9 import MultiModal
#from bert_top1_fc_gelu_shareembedding_meanpool_best1 import MultiModal
#from bert_nextvlad import MultiModal
#
from pretrain_roberta4_noleaky import MultiModal
#from bert_double import MultiModal
#from vlbert_pretrain import MultiModal
#from vlbert_twoembedding_meanpooler import MultiModal
from imp import reload
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        # logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.FileHandler(f"src/code1/log_file/pretrain_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    #
    model = MultiModal(args)
    #model = SMARTBertClassificationLoss2(args)
    #
    #model = SMARTClassificationModel(args)
    #
    print(model)
    #model.load_state_dict(torch.load(args.pre_model))
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = 100
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.sum()
            #accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.5f}")

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.5f}, {results}")
        # 5. save checkpoint
        # mean_f1 = loss
        # if mean_f1 < best_score:
        #     best_score = mean_f1
        #     if epoch>10 and epoch%3==0:
        #         torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'loss_train_epoch': mean_f1},
        #                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        mean_f1 = loss
        if mean_f1 < best_score:
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/best_pretrain_roberta_base.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
