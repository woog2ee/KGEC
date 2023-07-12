import time
import argparse
from dataset import BERTDataset
from torch.utils.data import DataLoader
from detect_model.bert import BERT
from detect_trainer import BERTTrainer



def train():
    parser = argparse.ArgumentParser()

    # Dataset & output path
    parser.add_argument('--dataset_path', required=True, type=str,
                        default='/HDD/kyohoon1/data_split/',
                        help='')
    parser.add_argument('--sow_token', type=bool, default=True,
                        help='')
    parser.add_argument('--eow_token', type=bool, default=True,
                        help='')
    parser.add_argument('--output_path', required=True, type=str,
                        default='/home/seunguk/KGEC/AAAI2023/output/',
                        help='')
    
    # Training hyper parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--epochs', type=int, default=30,
                        help='')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='')
    parser.add_argument('--max_char_len', type=int, default=20,
                        help='')
    parser.add_argument('--max_word_len', type=int, default=150,
                        help='')
    
    # BERT hyper parameters
    parser.add_argument('--n_heads', type=int, default=8,
                        help='')
    parser.add_argument('--bert_d_model', type=int, default=256,
                        help='')
    parser.add_argument('--bert_layers', type=int, default=6,
                        help='')
    parser.add_argument('--bert_dropout', type=float, default=0.1,
                        help='')                    
    
    # Bi-LSTM hyper parameters
    parser.add_argument('--embed_size', type=int, default=64,
                        help='')
    parser.add_argument('--lstm_d_model', type=int, default=256,
                        help='')
    parser.add_argument('--lstm_layers', type=int, default=3,
                        help='')
    parser.add_argument('--lstm_bias', type=bool, default=True,
                        help='')
    parser.add_argument('--lstm_dropout', type=float, default=0.1,
                        help='')
    
    # Adam hyper parameters
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='')
    parser.add_argument('--adam_weight_decay', type=float, default=0.01,
                        help='')
    parser.add_argument('--adam_warmup_steps', type=int, default=10000,
                        help='')
    
    print('Loading All Parse Arguments\n')
    args = parser.parse_args()
    
    
    print('Building BERT Model\n')
    bert = BERT(n_heads      = args.n_heads,
                bert_d_model = args.bert_d_model,
                bert_layers  = args.bert_layers,
                bert_dropout = args.bert_dropout,
                embed_size   = args.embed_size,
                lstm_d_model = args.lstm_d_model,
                lstm_layers  = args.lstm_layers,
                lstm_bias    = args.lstm_bias,
                lstm_dropout = args.lstm_dropout,
                sow_token    = args.sow_token,
                eow_token    = args.eow_token)
    
    print('Creating BERT Trainer\n')
    trainer = BERTTrainer(bert         = bert,
                          lr           = args.lr,
                          betas        = (args.adam_beta1, args.adam_beta2),
                          weight_decay = args.adam_weight_decay,
                          warmup_steps = args.adam_warmup_steps)
    

    print('Training Start\n')
    for epoch in range(args.epochs):

        # Load different Dataset at each epoch for curriculum learning
        dataset     = BERTDataset(args.dataset_path, epoch,
                                  args.sow_token, args.eow_token,
                                  args.max_char_len, args.max_word_len)
        
        # Load diffrent DataLoader at each epoch for curriculum learning
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        print(f'Epoch {epoch} DataLoader Created!\n')


        # Train
        start_time = time.time()
        train_loss = trainer.do(mode        = 'train',
                                epoch       = epoch,
                                data_loader = data_loader)
        # Test
        test_loss = trainer.do(mode        = 'test',
                               epoch       = epoch,
                               data_loader = data_loader)
        
        print(f'\nEpoch: {epoch} | Time: {time.time() - start_time}')
        print(f'Train Loss: {train_loss} | Test Loss: {test_loss}\n')

        # Save
        trainer.save(epoch, args.output_path)



if __name__ == '__main__':
    train()