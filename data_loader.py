import torch
from torchvision import transforms, datasets


def get_loader(name_dataset, batch_size, train=True, data_dir=None):

    # Computed with compute_mean_std.py
    mean_std = {
        'amazon': {
            'mean': [0.79235494, 0.7862071, 0.78418255],
            'std':  [0.31496558, 0.3174693, 0.3193569]
        },
        'dslr': {
            'mean': [0.47086468, 0.44865608, 0.40637794],
            'std':  [0.20395322, 0.19204104, 0.1996422]
        },
        'webcam': {
            'mean': [0.6119875 , 0.6187739, 0.61730677],
            'std':  [0.25063968, 0.25554898, 0.25773206]
        },
        'ub': {
            'mean': [0.38512646, 0.36533064, 0.36375676],
            'std': [0.09447344, 0.08681861, 0.10410806]
        },
        'ntcir': {
            'mean': [0.42066872, 0.40827047, 0.40907493],
            'std': [0.08447598, 0.0840078,  0.0850569 ]
        },
        'thomaz': {
            'mean': [0.38777892, 0.3505929, 0.29683589],
            'std': [0.11271215, 0.09815634, 0.09124211]
        }
    }

    data_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[name_dataset]['mean'],
                                 std=mean_std[name_dataset]['std'])
        ])

    if data_dir:
        dataset = datasets.ImageFolder(root=data_dir,
                                       transform=data_transform)
    else:
        dataset = datasets.ImageFolder(root='./data/%s/images' % name_dataset,
                                       transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=4)
    return dataset_loader

