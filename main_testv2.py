import os
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 
import numpy as np
import logging 
from mobilenet import MobileNet
from utils import plot_loss_acc, plot_lr




def get_train_valid_loader(dataset_dir, batch_size, parameter, seed, save_images):
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # whitening data
    transform_train = v2.Compose(
    [v2.PILToTensor(),
     v2.ToDtype(torch.float, scale=True),
     v2.Normalize([0.5068, 0.4861, 0.4403], [0.2671, 0.2563, 0.2759]),
     ])
    
    # get dataset 
    dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=parameter, download=save_images, transform=transform_train)
    
    num_training = int(0.8 * len(dataset))
    num_validation = len(dataset) - num_training
    
    train, valid = torch.utils.data.random_split(dataset=dataset ,lengths=[num_training, num_validation], generator=generator)

    # set the mini-batch training
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader
    
def get_test_loader(dataset_dir, batch_size):
    transform_test = v2.Compose([v2.PILToTensor(), 
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize([0.5068, 0.4861, 0.4403], [0.2671, 0.2563, 0.2759])]) 
    testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=2)
    return test_loader


def main(args):
    os.makedirs("log", exist_ok=True)

    # logging
    log_file = f"testv2_log/lr{args.lr}_wd{args.wd}_eps{args.epochs}_scheduler{args.lr_scheduler}_mixup{args.mixup}_alpha{args.alpha}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders youself
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=args.save_images) 
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    if args.mixup:
        mixup = v2.MixUp(num_classes=100, alpha=args.alpha)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model = MobileNet(100)
    print(model)
    model.to(device)
    
        
    # criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)
    


    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    stat_lr = []

    transform_train = v2.Compose(
    [v2.PILToTensor(),
     v2.ToDtype(torch.float, scale=True),
     v2.Normalize([0.5068, 0.4861, 0.4403], [0.2671, 0.2563, 0.2759]),
     v2.RandomHorizontalFlip(p=0.5),
     v2.RandomCrop((32,32), padding=4),
     ])
    
    
    for epoch in range(args.epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        for imgs, labels in train_loader:
            # data augmentation
            imgs = transform_train(imgs)
            imgs = imgs.cuda()
            labels = labels.cuda()

            if args.mixup:
                imgs, labels= mixup(imgs, labels)

            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            logits = model.forward(imgs)

            loss = criterion(logits, labels)  # + (1 - lam) * criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            
            _, top_class = logits.topk(1, dim=1)
            if args.mixup:
            # equals = top_class == labels.cuda().view(*top_class.shape)
                equals = top_class == torch.argmax(labels, dim=1).view(top_class.shape)
            else:
                equals = top_class == labels.cuda().view(*top_class.shape)

            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size
        assert training_samples==40000
        
        # validation
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        stat_lr.append(scheduler.get_lr()[0])
        # print
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        logging.info(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        # lr scheduler
        scheduler.step()
    # plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)

    # plot_prob_density(alpha)
    # test
    if args.test:
        test_loss = 0
        test_acc = 0
        test_samples = 0
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            test_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss += batch_size * test_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        print('Test loss: ', test_loss/test_samples)
        print('Test acc: ', test_acc/test_samples)
        logging.info(f"Test loss: {(test_loss/test_samples):.4f}.. Test acc: {(test_acc/test_samples):.4f}")
     
    lr_fig = f"lr{args.lr}_wd{args.wd}_alpha{args.alpha}learning_curve.png"
    plot_lr(stat_lr, lr_fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    parser.set_defaults(mixup=False)
    parser.add_argument('--alpha', type=float, help='')
    parser.set_defaults(alpha=0.0)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()
    print(args)
    main(args)



    