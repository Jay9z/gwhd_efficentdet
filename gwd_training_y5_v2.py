import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import datetime
import glob
import random
import cv2
import math
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset,RandomSampler,SequentialSampler
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

import warnings
warnings.filterwarnings("ignore")
TRAIN_ROOT_PATH = r'../input/global-wheat-detection/train'

class TrainGlobalConfig:
    num_workers = 2
    accumulate_steps = 32
    batch_size = 2
    n_epochs = 15  # n_epochs = 40
    lr = 0.0004 # 0.0004
    momentum = 0.937
    weight_decay = 5e-4
    mixed_precision = False
    folder = 'effdet5-from-coco_y5optim_2'

    # -------------------
    verbose = True
    verbose_step = 10
    # -------------------

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    #     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    #     scheduler_params = dict(
    #         max_lr=0.001,
    #         epochs=n_epochs,
    #         steps_per_epoch=int(len(train_dataset) / batch_size),
    #         pct_start=0.1,
    #         anneal_strategy='cos',
    #         final_div_factor=10**5
    #     )

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=3,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )


try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for fast er mixed precision training: https://github.com/NVIDIA/apex')
    TrainGlobalConfig.mixed_precision = False  # not installed


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        oppt = random.random()
        if self.test or oppt > 0.75:
            image, boxes = self.load_image_and_boxes(index)
        # elif oppt > 0.25:
        #     image, boxes = self.load_mixup_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    break
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_mixup_image_and_boxes(self, index):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        return (image + r_image) / 2, np.vstack((boxes, r_boxes)).astype(np.int32)

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.step_finished = 0
        self.bs = config.batch_size
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        ## Original optimizer version
        # param_optimizer = list(self.model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        ##Configed by TrainGlobalConfig
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.lr)
        ##self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        #self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        ## leverage yolov5 optimizer
        self.n_burn = 1e3  # burn-in iterations, max(3 epochs, 1k iterations)
        ##FIXME: do cosine optimizer by TrainGlobalConfig
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        adam = False
        self.optimizer = optim.Adam(pg0, lr=config.lr) if adam else \
            optim.SGD(pg0, lr=config.lr, momentum=config.momentum, nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': config.weight_decay})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / config.n_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        #scheduler.last_epoch = start_epoch - 1  # do not move
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)
        
        # Mixed precision training https://github.com/NVIDIA/apex
        if self.config.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        self.n_burn = max(3 * len(train_loader), 1e3)
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.datetime.now().utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            train_summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {train_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            val_summary_loss = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {val_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if val_summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                # for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                #     os.remove(path)

            # if self.config.validation_scheduler:
            #     self.scheduler.step(metrics=val_summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        mloss = torch.zeros(4, device=self.device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        accumulate = self.config.accumulate_steps  # accumulate loss before optimizing
        #step = 0
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
                    
            # Burn-in / warm-up
            ni,n_burn  = self.step_finished, self.n_burn
            if ni <= n_burn:
                xi = [0, self.n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, self.config.accumulate_steps]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(self.config.n_epochs)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, self.config.momentum])

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            loss, _, _ = self.model(images, boxes, labels)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ')
                return results

            # Backward
            if self.config.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if self.step_finished % accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                #ema.update(model)            
                summary_loss.update(loss.detach().item(), batch_size)

            if self.config.step_scheduler:
                self.scheduler.step()

            self.step_finished += 1

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(650, 1024), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.68, sat_shift_limit= 0.68,
                                      val_shift_limit=0.1, p=0.9),
                A.RandomGamma(p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.1,
                                            contrast_limit=0.1, p=0.9),
            ],p=0.9),
           # A.CLAHE(p=1.0),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.CoarseDropout(max_holes=20, max_height=32, max_width=32, fill_value=0, p=0.25),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
          #  A.CLAHE(p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def get_net():
    # GWD restart
    config = get_efficientdet_config('tf_efficientdet_d5')
    print("det config: ",config)
    config.num_classes = 1
    config.image_size = 512
    net = EfficientDet(config, pretrained_backbone=False)
    # checkpoint = torch.load(r'effdet5-from-coco_y5optim/best-checkpoint-039epoch.bin')
    # net.load_state_dict(checkpoint['model_state_dict'])

    # ##From coco baseline
    # config = get_efficientdet_config('tf_efficientdet_d5')
    # print("det config: ",config)
    # net = EfficientDet(config, pretrained_backbone=False)
    # #checkpoint = torch.load(r'../input/eff_checkpoint/tf_efficientdet_d5-ef44aea8.pth')
    # net.load_state_dict(checkpoint)
    # config.num_classes = 1
    # config.image_size = 512

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)


def run_training():
    ##KFold images by source and bbox number
    marking = pd.read_csv(r'../input/global-wheat-detection/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking.drop(columns=['bbox'], inplace=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    # print("group: ",np.unique(df_folds.stratify_group))
    # stratify_count = df_folds[['stratify_group']].groupby('stratify_group').count()
    # print("stratify distribution：",stratify_count)
    # # 这么做的目的来源于两个方面。
    # # 1. 需要保证划分的多折训练集中数据来源占比一致。
    # # 2. 需要保证划分的多折训练集中 bbox 分布大致一致。
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    ## Form dataset
    #fold_numbers = [0, 1, 2, 3, 4]
    fold_numbers = [2,3,4]

    for fold_number in fold_numbers:

        train_dataset = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
            marking=marking,
            transforms=get_train_transforms(),
            test=False,
        )

        validation_dataset = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
            marking=marking,
            transforms=get_valid_transforms(),
            test=True,
        )

        ## Show image 1
        image, target, image_id = train_dataset[1]
        boxes = target['boxes'].cpu().numpy().astype(np.int32)

        numpy_image = image.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for box in boxes:
            cv2.rectangle(numpy_image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2)

        ax.set_axis_off()
        ax.imshow(numpy_image)

        # device = torch.device('cuda:0')
        # net.to(device)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        net.to(device)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=TrainGlobalConfig.batch_size,
            num_workers=TrainGlobalConfig.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            sampler=SequentialSampler(validation_dataset),
            batch_size=TrainGlobalConfig.batch_size,
            num_workers=TrainGlobalConfig.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=collate_fn,
        )

        fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
        path_pattern = r'effdet5-from-coco_y5optim_2/best-checkpoint-*epoch.bin'
        paths = glob.glob(path_pattern)
        path = sorted(paths,reverse=True)[0]
        print("weight path： ",path)
        fitter.load(path)
        fitter.fit(train_loader, val_loader)


if __name__ == '__main__':

    ## Select GPU device if there are multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ## Train model
    net = get_net()

    run_training()