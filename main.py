from train import train
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=256, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate with which to train")
parser.add_argument("--batch_size", type=int, default=256, help="size of one batch in one epoch")
parser.add_argument("--save_epoch", type=int, default=16, help="save classifier after this much epochs")
parser.add_argument("--savedir", default="save/", help="dir in which to save classifier")
parser.add_argument("--logdir", default="log/", help="dir in which to log training")
parser.add_argument("--logfile", default="log/log_0", help="file in which to log training")
parser.add_argument('--phrase', action='store_true', help="Use phrases in the dataset")

args = parser.parse_args()

if __name__ == "__main__":
    params = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "save_epoch": args.save_epoch,
        "logdir": args.logdir,
        "logfile": args.logfile,
        "savedir": args.savedir,
        "phrase": args.phrase
    }
    train(params)
