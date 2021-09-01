import pickle
import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, Alphabet, build_pretrain_embedding, my_collate_fn, lr_decay
import time
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_model, evaluate

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ceramic Coatings Named Entity Recognition')
    parser.add_argument('--mog', type=int, default=4)
    parser.add_argument('--charmog', type=int, default=4)
    parser.add_argument('--char_embedding_dim', type=int, default=10)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use_char', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--number_normalized', type=bool, default=False)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--train_path', default='data/')
    parser.add_argument('--dev_path', default='data/')
    parser.add_argument('--test_path', default='data/')
    parser.add_argument('--pretrain_embed_path', default='data/skipgram_phrases_200d.txt')
    parser.add_argument('--savedir', default='data/model/')
    parser.add_argument('--character_embedding', default='charbilstm')
    parser.add_argument('--feature_extractor',  default='BiMogLSTM')


    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    eval_path = "eval"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file1 = eval_temp + '/predeva.txt'
    score_file1 = eval_temp + '/scoreeva.txt'
    pred_file2 = eval_temp + '/predtest.txt'
    score_file2 = eval_temp + '/scoretest.txt'

    model_name = args.savedir + '/' + args.character_embedding + args.feature_extractor
    word_vocab = WordVocabulary(args.train_path, args.dev_path, args.test_path, args.number_normalized)
    with open('wordvocab.pkl', 'wb') as f:
        pickle.dump(word_vocab, f)
    label_vocab = LabelVocabulary(args.train_path)
    with open('labelvocab.pkl', 'wb') as f:
        pickle.dump(label_vocab, f)
    alphabet = Alphabet(args.train_path, args.dev_path, args.test_path)
    with open('alphabet.pkl', 'wb') as f:
        pickle.dump(alphabet, f)

    pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    np.savetxt("pretrain_word_embedding.txt", pretrain_word_embedding)

    train_dataset = MyDataset(args.train_path, word_vocab, label_vocab, alphabet, args.number_normalized)
    dev_dataset = MyDataset(args.dev_path, word_vocab, label_vocab, alphabet, args.number_normalized)
    test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, alphabet, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim, alphabet.size(),
                             args.char_embedding_dim, args.char_hidden_dim,
                             args.character_embedding, args.feature_extractor, label_vocab.size(), args.dropout,
                             args.mog, args.charmog,
                             pretrain_embed=pretrain_word_embedding, use_char=args.use_char, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_begin = time.time()
    print('----------begin training-----------')
    writer = SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0
    bestepoch = 0

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate(dev_dataloader, model, word_vocab, label_vocab, pred_file1, score_file1, eval_script, use_gpu)
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set:', best_f1)
            bestepoch = epoch + 1
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    print()
    print()
    print('---------end-----------')
    print('best epoch on dev set:', bestepoch)
    print('best F1-score on DEV:', best_f1)

    model.load_state_dict(torch.load(model_name))
    test_acc = evaluate(test_dataloader, model, word_vocab, label_vocab, pred_file2, score_file2, eval_script, use_gpu)
    print('Test acc on TEST:', test_acc)

