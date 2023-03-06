import torchvision
import torch
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from models.aug_simclr import SimCLRModel
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
train_dataset = datasets.ImageFolder(root='../imagenet20',transform=data_transform)
test_dataset = datasets.ImageFolder(root='../imagenet20_val',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4, shuffle=True,num_workers=4)
@torch.no_grad()
def evaluation(model):
    
    network = deepcopy(model.resnet)
    # network.projection = nn.Identity()
    network.eval()
    network.to('cuda:1')
    
    X_train_feature=[]
    y_train=[]
    X_test_feature=[]
    y_test=[]
    
    for data in train_loader:
        x, y = data
        x = x.to('cuda:1')
        features = network(x)
        X_train_feature.extend(features.cpu().detach().numpy())
        y_train.extend(y.cpu().detach().numpy())
    for data in test_loader:
        x, y = data
        x = x.to('cuda:1')
        features = network(x)
        X_test_feature.extend(features.cpu().detach().numpy())
        y_test.extend(y.cpu().detach().numpy())
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_feature)
    X_train_feature = scaler.transform(X_train_feature)
    X_test_feature = scaler.transform(X_test_feature)
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', C=1.0)# Multinomial Loss
    clf.fit(X_train_feature, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train_feature, y_train))
    print("Test score:", clf.score(X_test_feature, y_test))
model = SimCLRModel.load_from_checkpoint('./simclr/xi4tsgxl/checkpoints/epoch=16-step=3451.ckpt')
evaluation(model)