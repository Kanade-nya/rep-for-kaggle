# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
# import models
import pvt
import pvt_v2
from my_pvt import re_pvt
import SPVT
import utils
import collections


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--savepath', default='' )

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # 用于初始化分布式训练模式。分布式训练是指将训练任务分配给多个计算设备或多个计算节点进行并行计算，提高训练效率。
    utils.init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 这一段代码设置了随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # 这行代码启用了CuDNN（CUDA深度神经网络库）的自动优化功能，以提高训练速度。CuDNN会根据输入数据的大小和模型结构自动选择最优的卷积实现方式。
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed: 分布式训练
        # 获取分布式训练中的总任务数（即计算节点数）。
        num_tasks = utils.get_world_size()
        # 获取当前计算节点的全局排名。
        global_rank = utils.get_rank()
        # 根据args.repeated_aug参数的值选择不同的采样器（sampler）进行训练数据集的采样
        if args.repeated_aug:
            # RASampler是一个自定义的采样器，根据参数设置对训练数据集进行重复增强采样。它接受dataset_train作为数据集输入，
            # 使用num_replicas指定总任务数，rank指定当前计算节点的排名，shuffle=True表示采样时进行数据集的洗牌（随机顺序）。
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            # DistributedSampler是PyTorch提供的分布式训练采样器，它接受dataset_train作为数据集输入，
            # 使用num_replicas指定总任务数（这里设置为0），rank指定当前计算节点的排名，shuffle=True表示采样时进行数据集的洗牌。
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=True
            )
        # 根据args.dist_eval参数的值选择不同的采样器（sampler）进行验证数据集的采样
        if args.dist_eval:
            # 用于检查验证数据集的长度是否能够被任务数（即计算节点数）整除。
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                # num_replicas=num_tasks,
                num_replicas=0,
                rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    # 创建数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 存储混合数据增强函数（mixup）。
    mixup_fn = None
    # 判断使用什么数据增强 只要有一个条件为真，即表示混合数据增强处于活动状态，将mixup_active设置为True。
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 创建模型
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    # 进行模型的微调（fine-tuning）。
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    # 更换设备
    model.to(device)

    # 指数滑动平均
    model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')


    # /* 1 */
    '''
    这段代码主要完成了以下几个任务：
    对于分布式训练，使用DistributedDataParallel对模型进行并行处理，将模型进行分布式数据并行处理，并获取没有进行数据并行处理的模型副本。
    计算模型的参数总数，并打印出来。
    根据线性比例缩放规则，计算并更新学习率。
    创建优化器对象和损失
    '''

    # 将当前的模型赋值给model_without_ddp，此变量用于存储没有进行分布式数据并行处理（DistributedDataParallel，DDP）的模型副本。
    model_without_ddp = model

    # 如果args.distributed为True，表示使用分布式训练，执行以下代码块。
    if args.distributed:
        # 使用DistributedDataParallel将模型进行分布式数据并行处理。device_ids=[args.gpu]指定了模型在哪些GPU设备上进行并行处理。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 获取没有进行数据并行处理的模型，即去除了DistributedDataParallel的封装。
        model_without_ddp = model.module
    # 计算模型中需要梯度更新的参数总数。通过遍历模型的参数，对于每个具有requires_grad=True的参数，
    # 使用.numel()方法获取参数的元素数量，然后将这些数量相加，得到总的参数数量。
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)
    # 根据线性比例缩放规则计算学习率。
    # 这里将初始学习率(args.lr)与批次大小(args.batch_size)、
    # 分布式训练中的进程数量(utils.get_world_size())以及标准化的参考批次大小(512)相乘，得到缩放后的学习率。
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    #  将缩放后的学习率更新为args.lr，即更新了参数中的学习率。
    args.lr = linear_scaled_lr
    # 函数创建优化器对象，该函数根据给定的参数和模型创建相应的优化器。
    optimizer = create_optimizer(args, model_without_ddp)
    # 创建一个名为loss_scaler的NativeScaler对象，用于进行损失缩放（loss scaling）操作。
    loss_scaler = NativeScaler()
    # 使用create_scheduler函数创建学习率调度器对象，该函数根据给定的参数和优化器创建相应的学习率调度器。此处使用_来接收第二个返回值，但未使用。
    lr_scheduler, _ = create_scheduler(args, optimizer)


    # LabelSmoothingCrossEntropy的损失函数对象，并将其赋值给criterion变量。这个损失函数通常用于在分类任务中对标签进行平滑化处理。
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        # 使用SoftTargetCrossEntropy作为损失函数。这是因为在Mixup中，平滑化操作是通过混合的标签转换来处理的。
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        # 启用了标签平滑化），则使用带有指定平滑化参数的LabelSmoothingCrossEntropy作为损失函数。
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # 多类别分类任务中常用的损失函数。
        criterion = torch.nn.CrossEntropyLoss()

    # 下面是蒸馏？不太懂，没问
    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )

    '''
    # 通过将原始的损失函数和其他参数传递给DistillationLoss，可以在需要的情况下将知识蒸馏功能添加到损失函数中。
    # 这可以用于模型训练过程中的知识蒸馏任务，其中学生模型（当前模型）通过参考教师模型进行训练。
    # 然而，在这段代码中，将'none'和0的值传递给DistillationLoss，表示没有进行知识蒸馏，
    # 即仅使用原始的损失函数进行训练，没有教师模型、权重或温度参数的影响。
    '''

    # DistillationLoss是一种用于知识蒸馏（knowledge distillation）的损失函数。
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )

    # output_dir = Path(args.output_dir)
    # output_dir = Path('/content/drive/MyDrive/PVT')
    output_dir = Path(args.savepath)
    print(f'now dir is {output_dir}')

    # 模型恢复训练
    if args.resume:
        if args.resume.startswith('https'):
            # 函数从指定的URL加载模型的检查点，
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 表示从本地加载预训练模型的检查点。执行以下代码块以加载检查点。
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 检查点中存在键 'model'，表示检查点中保存了模型的状态字典。执行以下代码块。
        if 'model' in checkpoint:
            # 将检查点中的模型状态字典加载到没有进行数据并行处理的模型副本
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            # 表示检查点中直接保存了模型的状态字典
            msg = model_without_ddp.load_state_dict(checkpoint)
        print(msg)
        # 表示检查点中保存了优化器、学习率调度器和训练的起始轮数。执行以下代码块。
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # 如果args.eval为True，表示进行模型评估，执行以下代码块。
    if args.eval:
        # 调用evaluate函数对验证数据集进行评估，传入验证数据加载器、模型和设备，并将评估结果保存在test_stats变量中。
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return


    # 正式开始训练了
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0


    for epoch in range(args.start_epoch, args.epochs):
        # fp32恢复。这意味着在 epoch 大于初始 epoch + 1 时，将执行下面的逻辑。
        if args.fp32_resume and epoch > args.start_epoch + 1:
            # 表示不再执行 fp32 恢复。
            args.fp32_resume = False
        # GradScaler 是 PyTorch 的一种工具，用于控制混合精度训练中的梯度缩放。通过设置 enabled 参数为 True 或 False，可以启用或禁用梯度缩放。
        # args.fp32_resume 为 False，则梯度缩放被启用；如果 args.fp32_resume 为 True，则梯度缩放被禁用。
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        # 启用了分布式训练, 进程在每个 epoch 中使用不同的数据顺序进行训练。这有助于增加数据的随机性，避免每个进程在每个 epoch 中都使用相同的数据。
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 训练一个epoch 没有具体看
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume
        )

        # 学习率递减
        lr_scheduler.step(epoch)

        #if args.output_dir:
        if args.savepath:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 保存检查点
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # 训练后进行评估
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # 当前的最大精度
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # 保存训练的log log.txt
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        #if args.output_dir and utils.is_main_process():
        #    with (output_dir / "log.txt").open("a") as f:
        #       f.write(json.dumps(log_stats) + "\n")
        if args.savepath and utils.is_main_process():
            with (output_dir / "log74.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 输出训练时长
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    #if args.output_dir:
    #    Path('/content/drive/MyDrive/PVT').mkdir(parents=True, exist_ok=True)
    #    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.savepath:
        Path(args.savepath).mkdir(parents=True, exist_ok=True)
    main(args)
