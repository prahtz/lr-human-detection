import torch.distributed as dist
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from tqdm import tqdm
from models.efficientnet import load_efficientnet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch
from datasets.thermal_person_classification import Negatives, Positives, ConcatDataset, load_batch_collator
import utils

utils.init_distributed()
utils.set_seed(42)

local_rank, global_rank, num_replicas = utils.get_ranks_and_replicas()

num_epochs = 100
lr = 5e-5
accumulation_steps = 1
batch_size = 16


pos = Positives('data/positives', 'train')
neg = Negatives('data/negatives', 'train')

train_dataset = ConcatDataset([neg, pos])
sampler = utils.DistributedMultiplexerSampler([neg, pos], len(train_dataset), weights=[0.5, 0.5], rank=global_rank, num_replicas=num_replicas)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=load_batch_collator(), sampler=sampler)

model = load_efficientnet('efficientnet_b3')
if torch.cuda.is_available():
    model.cuda()
    
if dist.is_torchelastic_launched():
    device_ids = [local_rank] if torch.cuda.is_available() else None
    model = DDP(model, device_ids=device_ids)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()


log_dir = 'logs/'
training_writer = SummaryWriter(log_dir)
training_log = defaultdict(float)
iter_epochs = range(num_epochs)
if global_rank == 0:
    iter_epochs = tqdm(range(num_epochs), total=num_epochs)

for epoch in iter_epochs:
    model.train()
    optimizer.zero_grad()
    training_log['train/loss'] = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        logits = model(inputs)
        loss = loss_fn(logits, labels) / accumulation_steps
        loss.backward()
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        training_log['train/loss'] += (loss.detach() / len(train_loader))

    with torch.no_grad():
        if dist.is_torchelastic_launched():
            tensor_list = utils.gather_tensor(training_log['train/loss'], global_rank, num_replicas, 0)
            if global_rank == 0:
                metric = torch.mean(torch.stack(tensor_list)).item()
                training_writer.add_scalar('train/loss', metric, epoch)