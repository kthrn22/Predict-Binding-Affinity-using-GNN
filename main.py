from model import *
from dataset import *
from trainer import *
from utils import *
from config import *

config = Config()

dataset = Ligand_Protein_Dataset(config.root, config.data_dir, config.affinity_file)

val_size = int(config.val_split * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

if config.test_split > 0:
    test_size = int(config.test_split * len(val_dataset))
    val_size = len(val_dataset) - test_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

model = Binding_Affinity_Predictor(config.in_channels, config.num_gnn_layers, config.num_linear_layers, config.linear_out_channels)

optimizer = Adam(model.parameters(), lr = config.learning_rate)
if config.use_scheduler:
    scheduler = StepLR(optimizer, step_size = config.step_size, gamma = config.gamma)
else:
    scheduler = None

criterion = torch.nn.L1Loss()

train_dataloader = DataLoader(train_dataset, batch_size = config.train_batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = config.val_batch_size, shuffle = True)

if config.test_split > 0:
    test_dataloader = DataLoader(test_dataset, batch_size = config.test_batch_size, shuffle = True)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

trainer = Trainer(model, config.device, criterion, optimizer, scheduler = None)

model = model.to(config.device)

trained_model, best_model = trainer.train(config.num_epochs, train_dataloader, val_dataloader, config.early_stop, config.patience)

print("Saving...")
torch.save(best_model, "best_model.pt")
torch.save(trained_model, "model.pt")
torch.save(optimizer, "optimizer.pt")
if config.use_scheduler:
    torch.save(scheduler, "scheduler.pt")