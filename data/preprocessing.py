import os
import torchvision



def relabel_dataset( data_root, labels):

    dataset = torchvision.datasets.ImageFolder(data_root)

    unlabeled_set, labeled_set = [], []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            labeled_set.append([path, label_idx])
            del labels[filename]
        else:
            unlabeled_set.append(path)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))


    return labeled_set, unlabeled_set
