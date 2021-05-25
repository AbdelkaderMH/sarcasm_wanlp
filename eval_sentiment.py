import argparse
import numpy as np
import torch
import torch.nn as nn

import preprocessing
import modeling
from barbar import Bar
import random

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(base_model, mt_classifier, iterator):
    all_sentiment_outputs = []
    all_sentiment_labels = []

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            output, pooled = base_model(**data_input)
            sentiment_logits = mt_classifier(output, pooled)

            sentiment_probs = nn.Softmax(dim=1)(sentiment_logits).to(device)
            _, predicted_sentiment = torch.max(sentiment_probs, 1)
            all_sentiment_outputs.extend(predicted_sentiment.squeeze().int().cpu().numpy())

    return all_sentiment_outputs


def eval_full(config, test_loader):
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    base_model.load_state_dict(torch.load("./ckpts/best_basemodel_sentiment_"+config["lm"]+".pth"))
    base_model.to(device)

    mtl_classifier = modeling.CLSClassifier(base_model.output_num(),class_num=3)
    mtl_classifier.load_state_dict(torch.load("./ckpts/best_mtl_cls_sentiment_"+config["lm"]+".pth"))
    mtl_classifier.to(device)
    all_sentiment_outputs = evaluate(base_model, mtl_classifier, test_loader)
    return all_sentiment_outputs


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--lm_pretrained', type=str, default='arabert',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_mult', type=float, default=1, help="dicriminator learning rate multiplier")

    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)


    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()


    config = {}
    config['args'] = args
    config["output_for_test"] = True
    config['epochs'] = args.epochs
    config["class_num"] = 1
    config["lr"] = args.lr
    config['lr_mult'] = args.lr_mult
    config['batch_size'] = args.batch_size
    config['lm'] = args.lm_pretrained

    dosegmentation = False
    if args.lm_pretrained == 'arbert':
        config['pretrained_path'] = "UBC-NLP/ARBERT"
    elif args.lm_pretrained == 'marbert':
        config['pretrained_path'] = "UBC-NLP/MARBERT"
    elif args.lm_pretrained == 'larabert':
        config['pretrained_path'] = "aubmindlab/bert-large-arabertv02"
        dosegmentation = True
    else:
        config['pretrained_path'] = 'aubmindlab/bert-base-arabertv02'
        dosegmentation = True

    label_dict = {
        0: "NEG",
        1: "NEU",
        2: "POS"
    }
    seeds = [12345]#, 12346, 12347, 12348, 12349]
    for RANDOM_SEED in seeds:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        test_loader = preprocessing.loadTestData(batchsize=args.batch_size, num_worker= 1, pretraine_path=config['pretrained_path'])
        all_sentiment = eval_full(config, test_loader)
        submission = pd.DataFrame(columns=['Sentiment'])
        submission["Sentiment"] = all_sentiment
        submission["Sentiment"].replace(label_dict, inplace=True)
        submission.to_csv("results/sentiment/CS-UM6P_Subtask_2_MARBERT_CLSATT.csv", index=False, header=False)

