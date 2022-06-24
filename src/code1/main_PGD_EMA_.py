import logging
import os
import time
import torch

from config_f_pgd_ import parse_args
# from data_helper_model7 import create_dataloaders
# from data_helper_128_3_swbq import create_dataloaders
from fintue_data_480_tw_bu_fc_ import create_dataloaders
# from fintue_data_480_tw_bu_fc_quan import create_dataloaders

# from model1 import MultiModal
# from model_MLM_MFM_allloss import MultiModal # 没有mlm，mfm
from fintue_model_roberta_ import MultiModal

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

from imp import reload
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        # logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.FileHandler(f"src/code1/log_file/train_EMA_PGD480_bs32_lr5e-5_preepoch33_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = self.shadow[name].to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='BertEmbeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='BertEmbeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self,emb_name='BertEmbeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.grad_backup
                param.grad = self.grad_backup[name]

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
    #print(model)
    checkpoint = torch.load(args.pre_model)
    # checkpoint = torch.load('/home/test/liuluyao/wxy/match/dspfl/challenge-main/save/fintue_model/Best_EMA_PGD480_bs32_lr5e-5_preepoch33_down0.673_up0.668.bin')
    model.load_state_dict(checkpoint['model_state_dict'])
    pgd = PGD(model)
    K = 5
    ema = EMA(model, 0.999)  # # EMA初始化
    ema.register()
    #model = SMARTBertClassificationLoss2(args)
    #
    #model = SMARTClassificationModel(args)

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            #pgd.backup_grad()  # 保存正常的grad
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()  # 恢复正常的grad
                loss, accuracy, _, _  = model(batch)
                loss.mean().backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()  # 训练过程中，更新完参数后，同步update shadow weights

            step += 1
            # if step % args.print_steps == 0:
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss.item():.3f}, accuracy {accuracy:.3f}")

                # # 4. validation
                # ema.apply_shadow()  # model应用shadow weights，进行验证、保存模型
                # loss, results = validate(model, val_dataloader)
                # results = {k: round(v, 4) for k, v in results.items()}
                # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 4. validation
        ema.apply_shadow()  # model应用shadow weights，进行验证、保存模型
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        # mean_f1 = results['mean_f1']
        # torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
        #            f'save/fintue_model/best_fintue_epoch{epoch}_ql_480_bs32_lr5e-5_preepoch27_seed42_model.bin')

        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            # torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
            #            f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/Best_EMA_PGD_480_bs32_lr5e-5_preepoch33.bin')
        ema.restore()  # 下一次训练之前，恢复模型参数  ##在此之前 保存模型


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
