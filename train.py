import torch
from dataset import SSTDataset, word_to_ix
from model import PretrainedEncoder, Classifier 
import os
import torch.nn as nn

def train_one_epoch(encoder, classifier, lossfn, optimizer, dataset, device, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    encoder.train()
    classifier.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in generator:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(encoder(batch))
        err = lossfn(logits, labels)
        err.backward()
        optimizer.step()

        train_loss += err.item()
        pred_labels = torch.argmax(logits, dim=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(encoder, classifier, lossfn, optimizer, dataset, device, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    encoder.eval()
    classifier.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(device), labels.to(device)
            logits = classifier(encoder(batch))
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, dim=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc

def train(params):
    trainset = SSTDataset(phrase=params["phrase"])
    devset = SSTDataset(split="dev", phrase=params["phrase"])
    testset = SSTDataset(split="test", phrase=params["phrase"])

    encoder = PretrainedEncoder(len(word_to_ix))
    classifier = Classifier(512, 256, 5)

    optimizer = torch.optim.Adam(classifier.parameters(),
                                    lr=params["lr"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device", device)
    os.makedirs(params["logdir"], exist_ok=True)
    os.makedirs(params["savedir"], exist_ok=True)
    logfile = open(params["logfile"], "w")
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    for epoch in range(params["epochs"]):
        print("epoch", epoch, flush=True)
        logfile.write(f"epoch {epoch}\n")
        tn_loss, tn_acc = train_one_epoch(encoder, classifier,
                                        nn.CrossEntropyLoss(),
                                        optimizer,
                                        trainset,
                                        device,
                                        batch_size=params["batch_size"])
        
        val_loss, val_acc = evaluate_one_epoch(encoder, classifier,
                                        nn.CrossEntropyLoss(),
                                        optimizer,
                                        devset,
                                        device,
                                        batch_size=params["batch_size"])
        
        tst_loss, tst_acc = evaluate_one_epoch(encoder, classifier,
                                        nn.CrossEntropyLoss(),
                                        optimizer,
                                        testset,
                                        device,
                                        batch_size=params["batch_size"])
        # logfile.write(f"train {tn_loss} {tn_acc*100}% \t val {val_loss} {val_acc*100}% \t test {tst_loss} {tst_acc*100}%\n")
        logfile.write("train {:.4} {:.4}% \t val {:.4} {:.4}% \t test {:.4} {:.4}%".format(tn_loss, tn_acc*100, val_loss, val_acc*100, tst_loss, tst_acc*100))
        print("train {:.4} {:.4}% \t val {:.4} {:.4}% \t test {:.4} {:.4}%".format(tn_loss, tn_acc*100, val_loss, val_acc*100, tst_loss, tst_acc*100))
        if ((epoch+1)%8 == 0) or epoch == params["epochs"]-1:
            torch.save(classifier.state_dict(), params["savedir"] + f"{epoch}.tar")
    logfile.close()
