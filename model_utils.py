import torch

from sklearn.metrics import confusion_matrix

def accuracy(y_prob, y_true):
    _, preds = torch.max(y_prob, dim=-1)
    return torch.sum(preds==y_true).numpy() / len(y_true)

def get_confusion(y_prob, y_true):
    _, preds = torch.max(y_prob, dim=-1)
    matrix = confusion_matrix(y_true.numpy(), preds.numpy())
    return matrix

def train_model(model, train_loader, optimizer, scheduler, criterion, epoch, waypoint_count=2, f=None):
    model.train()

    y_prob = []
    y_true = []
    running_loss = 0.0

    waypoint = int(len(train_loader)/waypoint_count)

    for i, datas in enumerate(train_loader, 0):
        inputs, _labels, _ = datas
        inputs = inputs.cuda()
        _labels = _labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, _labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_t = _labels.cpu().numpy().tolist()
        y_true += y_t
        y_p = outputs.cpu().detach().numpy().tolist()
        y_prob += y_p

        #scheduler.step()

        if i % waypoint == waypoint-1:
            y_prob = torch.tensor(y_prob, device='cpu')
            y_true = torch.tensor(y_true, device='cpu')
            acc = accuracy(y_prob, y_true)
            avg_loss = running_loss / waypoint

            if f is not None:
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc), file=f)
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc))
            else:
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc))
            y_prob = []
            y_true = []
            running_loss = 0.0

    return avg_loss, acc

def train_model2(model, train_loader, optimizer, scheduler, criterion, epoch, waypoint_count=2, f=None):
    model.train()

    y_prob = []
    y_true = []
    running_loss = 0.0

    waypoint = int(len(train_loader)/waypoint_count)

    for i, datas in enumerate(train_loader, 0):
        inputs, _labels, _, other_features = datas
        inputs = inputs.cuda()
        _labels = _labels.cuda()
        other_features = other_features.cuda()
        
        optimizer.zero_grad()

        outputs = model(inputs, other_features)

        loss = criterion(outputs, _labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_t = _labels.cpu().numpy().tolist()
        y_true += y_t
        y_p = outputs.cpu().detach().numpy().tolist()
        y_prob += y_p

        #scheduler.step()

        if i % waypoint == waypoint-1:
            y_prob = torch.tensor(y_prob, device='cpu')
            y_true = torch.tensor(y_true, device='cpu')
            acc = accuracy(y_prob, y_true)
            avg_loss = running_loss / waypoint

            if f is not None:
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc), file=f)
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc))
            else:
                print('[{}:{}: Train] loss: {:.4f} acc: {:.4f}'.format(
                                    epoch + 1, i,
                                    avg_loss,
                                    acc))
            y_prob = []
            y_true = []
            running_loss = 0.0

    return avg_loss, acc

def test_model(model, test_loader, criterion, phase='valid', f=None):
    model.eval()
    running_loss = 0.0
    n = 0
    names = []
    y_prob = []
    y_true = []
    for inputs, _labels, name in test_loader:
        inputs = inputs.cuda()
        _labels = _labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, _labels)
        running_loss += loss.item()
        n += 1

        y_t = _labels.cpu().numpy().tolist()
        y_true += y_t
        y_p = outputs.cpu().detach().numpy().tolist()
        y_prob += y_p
        names += list(name)

    y_prob = torch.tensor(y_prob, device='cpu')
    y_true = torch.tensor(y_true, device='cpu')
    matrix = get_confusion(y_prob, y_true)
    acc = accuracy(y_prob, y_true)
    avg_loss = running_loss / n

    if f is not None:
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc), file=f)
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc))
        print(matrix, file=f)
    else:
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc))
        print(matrix)
    return avg_loss, acc, names, y_prob, y_true
    
def test_model2(model, test_loader, criterion, phase='valid', f=None):
    model.eval()
    running_loss = 0.0
    n = 0
    names = []
    y_prob = []
    y_true = []
    for inputs, _labels, name, other_features in test_loader:
        inputs = inputs.cuda()
        _labels = _labels.cuda()
        other_features = other_features.cuda()
        outputs = model(inputs, other_features)
        loss = criterion(outputs, _labels)
        running_loss += loss.item()
        n += 1

        y_t = _labels.cpu().numpy().tolist()
        y_true += y_t
        y_p = outputs.cpu().detach().numpy().tolist()
        y_prob += y_p
        names += list(name)

    y_prob = torch.tensor(y_prob, device='cpu')
    y_true = torch.tensor(y_true, device='cpu')
    matrix = get_confusion(y_prob, y_true)
    acc = accuracy(y_prob, y_true)
    avg_loss = running_loss / n

    if f is not None:
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc), file=f)
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc))
        print(matrix, file=f)
    else:
        print('[{}] loss: {:.4f} acc: {:.4f}'.format(phase,
                                                  avg_loss,
                                                  acc))
        print(matrix)
    return avg_loss, acc, names, y_prob, y_true

def predict(model, test_loader, phase='predict', f=None):
    model.eval()

    names = []
    y_prob = []
    y_true = []
    for inputs, _labels, name in test_loader:
        inputs = inputs.cuda()
        _labels = _labels.cuda()
        outputs = model(inputs)

        y_t = _labels.cpu().numpy().tolist()
        y_true += y_t
        y_p = outputs.cpu().detach().numpy().tolist()
        y_prob += y_p
        names += list(name)

    y_prob = torch.tensor(y_prob, device='cpu')
    y_true = torch.tensor(y_true, device='cpu')
    matrix = get_confusion(y_prob, y_true)
    acc = accuracy(y_prob, y_true)

    if f is not None:
        print('[{}] acc: {:.4f}'.format(phase, acc), file=f)
        print('[{}] acc: {:.4f}'.format(phase, acc))
        print(matrix, file=f)
    else:
        print('[{}] acc: {:.4f}'.format(phase, acc))
        print(matrix)
    return acc, names, y_prob, y_true