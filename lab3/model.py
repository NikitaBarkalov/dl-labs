import numpy as np
import torch
import torchvision

def build_model_and_optimizer(regime, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = None if regime == 'scratch' else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights = weights)
    model.fc = torch.nn.Linear(512, 6)

    if regime == 'scratch':
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-3, weight_decay = 1e-4)

    elif regime == 'frozen':
        for p in model.parameters():
            p.requires_grad = False

        for p in model.fc.parameters():
            p.requires_grad = True

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr = 3e-3, weight_decay = 1e-4)

    elif regime == 'unfreeze':
        for p in model.parameters():
            p.requires_grad = False

        for p in model.layer4.parameters():
            p.requires_grad = True

        for p in model.fc.parameters():
            p.requires_grad = True

        model = model.to(device)
        optimizer = torch.optim.AdamW([
            {'params': model.fc.parameters(), 'lr': 3e-3},
            {'params': model.layer4.parameters(), 'lr': 3e-4}
        ], weight_decay = 1e-4)

    else:
        raise ValueError(f'Unknown regime: {regime}')

    return model, optimizer

def set_train_mode(model, regime):
    model.train()

    if regime == 'frozen':
        eval_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
        for m in eval_modules:
            m.eval()

    elif regime == 'unfreeze':
        eval_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
        for m in eval_modules:
            m.eval()
