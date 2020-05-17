import pickle
import torch
import torchtext.datasets as datasets
from torchtext import datasets
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import TranslationDataset
from pathlib import Path


def get_data(args):
    # batch
    batch_size = args.batch
    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"


    # set up fields
    src = Field(sequential=True,tokenize=str.split,
                use_vocab=True,
                lower=True,
                include_lengths=False,
                fix_length=args.max_length, # fix max length
                batch_first=True)
    trg = Field(sequential=True,tokenize=str.split,
                use_vocab=True,
                init_token='<s>',
                eos_token='</s>',
                lower=True,
                fix_length=args.max_length,  # fix max length
                batch_first=True)

    print('set up fields ... done')


    if  args.data_type == "koen":

        train, valid, test = TranslationDataset .splits( ('.ko', '.en'),
                                         (src, trg),train='train',validation='valid',test='test',path = args.root_dir)


        # build the vocabulary
        src.build_vocab(train.src, min_freq=args.min_freq)
        trg.build_vocab(train.trg, min_freq=args.min_freq)

        # save the voabulary
        src_vocabs = src.vocab.stoi
        trg_vocabs = trg.vocab.stoi

        with open('./src_vocabs.pkl', 'wb') as f :
            pickle.dump(src_vocabs, f, pickle.HIGHEST_PROTOCOL)
        with open('./trg_vocabs.pkl', 'wb') as f :
            pickle.dump(trg_vocabs, f, pickle.HIGHEST_PROTOCOL)

    else:
        assert False, "Please Insert data_type"



    train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), batch_sizes=([batch_size]*3), device=device)



    return (src, trg), (train, valid, test), (train_iter, valid_iter, test_iter)
