## what is deepfake?
Deepfake is fake media especially in the form of video and audio, produced by AI technology.

## key steps of baselines
1. define a model: create a resnet18 model by timm <br>
2. load train/test data: use DataLoader in pytorch
3. train and validate
4. evaluate the model's performance
5. submit the prediction results

## understand the code
``` Python
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):  # zero out all the parameters
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # update parameters in AverageMeter
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):  # define the form to print
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""


    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
```
In this part, we define two classes 'AverageMeter' and 'ProgressMeter' in order to monitor the train process
and displays progress and performance indicators.

```Python
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm_notebook(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        return top1

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm_notebook(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = F.softmax(output, dim=1)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)
```
### validate
初始化 AverageMeter 实例来记录时间、损失和准确率。
将模型切换到评估模式。
在没有梯度计算的情况下，遍历验证数据集：
将数据和标签移至 GPU。
计算模型的输出和损失。
计算准确率并更新 AverageMeter 实例。
记录每个批次处理的时间。
打印验证集上的准确率。
返回 top1，即准确率 AverageMeter

### predict
将模型切换到评估模式。
对于每个 TTA 迭代：
初始化一个列表来存储预测结果。
在没有梯度计算的情况下，遍历测试数据集：
将数据移至 GPU。
计算模型的输出，应用 softmax 函数，并将结果移至 CPU。
将预测结果添加到列表中。
将所有批次的预测结果堆叠成一个数组。
如果是第一次迭代，则直接存储结果；否则，将结果累加到 test_pred_tta。
返回累加的预测结果。

### train
初始化 AverageMeter 实例来记录时间、损失和准确率。
将模型切换到训练模式。
遍历训练数据集：
将数据和标签移至 GPU。
计算模型的输出和损失。
更新损失和准确率的 AverageMeter 实例。
清零优化器的梯度。
反向传播损失并更新模型参数。
记录每个批次处理的时间。
每隔 100 个批次，使用 ProgressMeter 打印训练进度。

``` Python
class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(np.array(self.img_label[index]))
    
    def __len__(self):
        return len(self.img_path)
```
### __getitem__
使用 Image.open 打开索引对应的图像路径，并将其转换为 RGB 格式。
如果 self.transform 不是 None，则应用这个变换到图像上。
将图像标签转换为 NumPy 数组，然后转换为 PyTorch 张量。
返回变换后的图像和标签。

``` Python
train_loader = torch.utils.data.DataLoader(
    FFDIDataset(train_label['path'].head(1000), train_label['target'].head(1000), 
            transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ), batch_size=40, shuffle=True, num_workers=4, pin_memory=True
)
```
使用 DataLoader 创建一个训练数据加载器，它将 FFDIDataset 实例作为输入，并应用了一系列变换，包括调整大小、随机水平翻转、随机垂直翻转、转换为张量以及标准化。
只取前 1000 条记录进行训练。
批大小为 40，数据在每个 epoch 期间会被打乱。
使用 4 个工作进程来加载数据。
pin_memory=True 可以加速数据传输到 GPU。
