import logging
import os
import time
import torch

from config import parse_args
from dataset.data_helper import create_dataloaders

from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, increment_path, MetricLogger, EMA
from models.singlestream_model import SingleStream_model
from utils.adv_train import FGM

import yaml


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
    # 1. load data and save yaml
    train_dataloader, val_dataloader = create_dataloaders(args)

    save_path = increment_path(args.savedmodel_path)
    os.makedirs(save_path, exist_ok=True)  # make dir

    with open(save_path / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    metric_logger = MetricLogger()
    txt_res, img_res = save_path / 'res.txt', save_path / 'res.png'

    # 2. build model and optimizers
    # model = MultiModal(args)
    model = SingleStream_model(args)

    ema = EMA(model, 0.999, device=args.device)  # EMA初始化
    ema.register()

    fgm = FGM(model)  # 对抗训练
    # pgd = PGD(model)
    # K = 3

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    train_loss, val_loss, train_acc = 0, 0, 0
    for epoch in range(3):
        for batch in train_dataloader:
            model.train()
            train_loss, train_acc, _, _ = model(batch)
            train_loss = train_loss.mean()
            train_acc = train_acc.mean()
            train_loss.backward()

            '''fgm'''
            fgm.attack()  # 在embedding上添加对抗扰动
            adv_loss, _, _, _ = model(batch)
            adv_loss = adv_loss.mean()
            adv_loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            '''pgd'''
            # pgd.backup_grad()
            # adv_loss = None
            # for t in range(K):
            #     pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
            #     if t != K - 1:
            #         model.zero_grad()
            #     else:
            #         pgd.restore_grad()
            #     adv_loss, _, _, _ = model(batch)
            #     adv_loss = adv_loss.mean()
            #     adv_loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # pgd.restore()  # 恢复embedding参数

            optimizer.step()  # 梯度下降，更新参数
            ema.update()  # 更新EMA参数

            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Train: Epoch {epoch} step {step} eta {remaining_time}: train_loss {train_loss:.3f}, "
                             # f"train_acc {train_acc:.3f}")
                             f"adv_loss {adv_loss:.3f}, train_acc {train_acc:.3f}")

        # 4. validation
        ema.apply_shadow()  # EMA验证
        val_loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Validate: Epoch {epoch} step {step}: val_loss {val_loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{save_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

        metric_logger.update(lr=scheduler.get_last_lr()[0], train_loss=train_loss, val_loss=val_loss,
                             train_acc=train_acc, mean_f1=mean_f1)
        ema.restore()  # 保存EMA

    # 6. save and plot
    metric_logger.output_csv(txt_res)
    metric_logger.plot(img_res, 3)


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    seed=[111111,,111,333,33333,4444,4456,177,896,66,7456,31,478,96143,7896,1610，1,3,76,5]#加入不同随机数
    for i in seed:
        setup_seed(args,i)

    # os.makedirs(args.savedmodel_path, exist_ok=True)
        logging.info("Training/evaluation parameters: %s", args)

        train_and_validate(args)


if __name__ == '__main__':
    main()
