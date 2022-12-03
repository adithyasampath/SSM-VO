import argparse
from ppnet_trainer import PPNetTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--log_freq', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default="./ppnet_weights")
    parser.add_argument('--data_dir', type=str, default="../dataset/poses")
    parser.add_argument('--reduction', type=str, default="mean")
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument('--model_type', type=str, default="sgru")
    parser.add_argument('--input_size', type=int, default=6)
    parser.add_argument('--output_size', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=12)
    parser.add_argument('--num_data_loader_workers', type=int, default=0)
    parser.add_argument('--seq', type=int, default=20)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--batch_first', action='store_true', default=True)
    args = parser.parse_args()

    trainer = PPNetTrainer(args)
    val_loss = trainer.train()
    print(f"Model validation loss after {args.epochs} epochs: {val_loss}")

# [1] 15107 
# [2] 15143 
# [3] 15180 
# [4] 15217 