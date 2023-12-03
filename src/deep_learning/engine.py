import torch
from tqdm import tqdm

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    correct_predictions = 0
    correct_predictions = torch.zeros(1, dtype=torch.float, device=device)
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        # calculate accuracy
        preds = torch.round(outputs)
        correct_predictions += torch.sum((preds == target).int()).double()

    train_loss = train_running_loss / counter
    train_accuracy = correct_predictions.item() / len(train_data)
    return train_loss, train_accuracy

# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    correct_predictions = torch.zeros(1, dtype=torch.float, device=device)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            # calculate accuracy
            preds = torch.round(outputs)
            correct_predictions += torch.sum((preds == target).int()).double()

        val_loss = val_running_loss / counter
        val_accuracy = correct_predictions.item() / len(val_data)
        return val_loss, val_accuracy