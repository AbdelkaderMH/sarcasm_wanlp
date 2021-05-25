import argparse
import numpy as np
import torch
import torch.nn as nn

import preprocessing
import modeling
from barbar import Bar
import random


from sklearn.metrics import f1_score, accuracy_score, classification_report
from losses import ModelMultitaskLoss
import utils
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(base_model, mt_classifier, iterator, optimizer, sent_criterion, sar_criterion, multi_task_loss, scheduler):

    # set the model in eval phase
    base_model.train(True)
    mt_classifier.train(True)
    acc_sentiment = 0
    acc_sarcasm= 0
    loss_sent = 0
    loss_sarc= 0
    mtl_loss = 0

    for data_input, label_input  in Bar(iterator):

        for k, v in data_input.items():
            data_input[k] = v.to(device)

        for k, v in label_input.items():
            label_input[k] = v.to(device)

        optimizer.zero_grad()


        #forward pass

        sentiment_target = label_input['sentiment']
        sarcasm_target = label_input['sarcasm']

        # forward pass

        output, pooled = base_model(**data_input)
        sarcasm_logits, sentiment_logits = mt_classifier(output, pooled)
        sentiment_probs = nn.Softmax(dim=1)(sentiment_logits).to(device)
        sarcasm_probs = torch.sigmoid(sarcasm_logits).to(device)

        # compute the loss
        loss_sentiment = sent_criterion(sentiment_logits, sentiment_target)
        loss_sarcasm = sar_criterion(sarcasm_probs.squeeze(), sarcasm_target)
        total_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
        #total_loss = loss_sentiment + loss_sarcasm
        loss_sarc += loss_sarcasm.item()
        loss_sent += loss_sentiment.item()
        mtl_loss += total_loss.item()

        # backpropage the loss and compute the gradients
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        acc_sentiment += utils.calc_accuracy(sentiment_probs, sentiment_target)
        acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)

    accuracies = {'Sentiment' : acc_sentiment / len(iterator) , 'Sarcasm': acc_sarcasm / len(iterator)}
    losses = {'Sentiment' : loss_sent / len(iterator), 'Sarcasm': loss_sarc / len(iterator), 'total_loss': mtl_loss / len(iterator)}
    return accuracies, losses

def evaluate(base_model, mt_classifier, iterator, sent_criterion, sar_criterion, multi_task_loss):
    # initialize every epoch
    acc_sentiment = 0
    acc_sarcasm= 0
    loss_sent = 0
    loss_sarc= 0
    total_loss = 0
    mtl_loss = 0

    start = True
    all_sentiment_outputs = []
    all_sentiment_labels = []
    all_sarcasm_outputs = []
    all_sarcasm_labels = []

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        for data_input, label_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            for k, v in label_input.items():
                label_input[k] = v.to(device)


            sentiment_target = label_input['sentiment']
            sarcasm_target = label_input['sarcasm']

            # forward pass

            output, pooled = base_model(**data_input)
            sarcasm_logits, sentiment_logits = mt_classifier(output, pooled)

            sentiment_probs = nn.Softmax(dim=1)(sentiment_logits).to(device)
            sarcasm_probs = torch.sigmoid(sarcasm_logits).to(device)
            # compute the loss
            loss_sentiment = sent_criterion(sentiment_logits, sentiment_target)
            loss_sarcasm = sar_criterion(sarcasm_probs.squeeze(), sarcasm_target)
            mtl_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
            #mtl_loss = loss_sentiment  + loss_sarcasm

            # compute the running accuracy and losses
            acc_sentiment += utils.calc_accuracy(sentiment_probs, sentiment_target)
            acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)

            loss_sent += loss_sentiment.item()
            loss_sarc += loss_sarcasm.item()
            total_loss += mtl_loss.item()

            _, predicted_sentiment = torch.max(sentiment_probs, 1)
            all_sentiment_outputs.extend(predicted_sentiment.squeeze().int().cpu().numpy())
            all_sentiment_labels.extend(sentiment_target.squeeze().int().cpu().numpy())
            predicted_sarcasm = torch.round(sarcasm_probs)
            all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy())
            all_sarcasm_labels.extend(sarcasm_target.squeeze().int().cpu().numpy())

    fscore_sentiment = f1_score(all_sentiment_outputs, all_sentiment_labels, average='macro')
    fscore_sarcasm = f1_score(all_sarcasm_outputs, all_sarcasm_labels, pos_label=1, average='binary')
    report_sentiment = classification_report(all_sentiment_outputs, all_sentiment_labels, target_names=['Negative', 'Neutral', 'Positive'], digits=4)
    report_sarcasm = classification_report(all_sarcasm_outputs, all_sarcasm_labels, target_names=['False', 'True'], digits=4)


    accuracies = {'Sentiment' : acc_sentiment / len(iterator) , 'Sarcasm': acc_sarcasm / len(iterator), 'F1_sentiment' : fscore_sentiment, 'F1_sarcasm': fscore_sarcasm,
                  'Report_sarcasm': report_sarcasm, 'Report_sentiment': report_sentiment}
    losses = {'Sentiment' : loss_sent / len(iterator), 'Sarcasm': loss_sarc / len(iterator),
              'total_loss': total_loss / len(iterator)}
    return accuracies, losses

def train_full(config, train_loader, stest_loader):
    lr_o = config['lr_mult'] * config['lr']
    lr = config['lr']

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    mtl_classifier = modeling.MTClassifier2(base_model.output_num(), dropout_prob=0.2).to(device)
    cls = 'MTClassifier2'

    ## set optimizer and criterions

    sarc_criterion = nn.BCELoss().to(device)
    sent_criterion = nn.CrossEntropyLoss().to(device)
    multi_task_loss = ModelMultitaskLoss().to(device)
    params = [{'params':base_model.parameters(), 'lr':config['lr']}, {'params': mtl_classifier.parameters(), 'lr': lr_o}]#, {'params':multi_task_loss.parameters(), 'lr': 0.0005}]
    optimizer = AdamW(params, lr=config["lr"])
    train_data_size = len(train_loader)
    steps_per_epoch = int(train_data_size / config['batch_size'])
    num_train_steps = len(train_loader) * config['epochs']
    warmup_steps = int(config['epochs'] * train_data_size * 0.1 / config['batch_size'])
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=10000)
    # Train model
    best_sentiment_valid_accuracy, best_sarcasm_valid_accuracy = 0, 0
    best_total_val_acc = 0
    best_val_loss = float('+inf')
    report_sarcasm = None
    report_sentiment = None
    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, mtl_classifier, train_loader, optimizer, sent_criterion, sarc_criterion, multi_task_loss,scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, mtl_classifier, valid_loader, sent_criterion, sarc_criterion, multi_task_loss)
        #print(multi_task_loss.parameters())
        valid_total_loss = valid_losses['total_loss']
        total_val_acc = valid_accuracies['F1_sentiment'] + valid_accuracies['F1_sarcasm']
        if total_val_acc > best_total_val_acc:
        #if best_val_loss > valid_total_loss:
            epo = epoch
            best_total_val_acc = total_val_acc
            best_val_loss = valid_total_loss
            best_sentiment_valid_accuracy = valid_accuracies['F1_sentiment']
            best_sarcasm_valid_accuracy = valid_accuracies['F1_sarcasm']
            report_sarcasm = valid_accuracies['Report_sarcasm']
            report_sentiment = valid_accuracies['Report_sentiment']
            best_sentiment_loss = valid_losses['Sentiment']
            best_sarcasm_loss = valid_losses['Sarcasm']
            total_loss = valid_losses['total_loss']
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), "./ckpts/best_basemodel_"+config["lm"]+".pth")
            torch.save(mtl_classifier.state_dict(), "./ckpts/best_mtl_cls_"+config["lm"]+".pth")


        print('********************Train Epoch***********************\n')
        print("accuracies**********")
        for k , v in train_accuracies.items():
            print(k+f" : {v * 100:.2f}")
        print("losses**********")
        for k , v in train_losses.items():
            print(k+f": {v :.5f}\t")
        print('********************Validation***********************\n')
        print("accuracies**********")
        for k, v in valid_accuracies.items():
            if 'Report' not in k:
                print(k+f": {v * 100:.2f}")
        print("losses**********")
        for k, v in valid_losses.items():
            print(k + f": {v :.5f}\t")
        print('******************************************************\n')
    print(f"epoch of best results {epo}")
    with open(f'reports/report_MTL_model_{cls}_.txt', 'w') as f:
        f.write("Sarcasm report\n")
        f.write(report_sarcasm)
        f.write('\n Sentimment Report\n')
        f.write(report_sentiment)
    return best_sentiment_valid_accuracy, best_sarcasm_valid_accuracy, best_sentiment_loss, best_sarcasm_loss,total_loss
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
    parser.add_argument('--lr_mult', type=float, default=1, help="Classifier learning rate multiplier")

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

    seeds = [12345]#, 12346, 12347, 12348, 12349]
    for RANDOM_SEED in seeds:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_loader, valid_loader = preprocessing.loadTrainValData(batchsize=args.batch_size, num_worker= 1, pretraine_path=config['pretrained_path'])
        best_sentiment_acc, best_sarcasm_acc, best_sentiment_loss, best_sarcasm_loss, total_loss =train_full(config, train_loader, valid_loader)
        print(f'Val. Sentiment F1: {best_sentiment_acc * 100:.2f}% |  Val. Sarcasm F1: {best_sarcasm_acc * 100:.2f}% \t Val Sentiment Loss {best_sentiment_loss :.4f} \t Val Sarcasm Loss {best_sarcasm_loss :.4f} \t Val total Loss {total_loss :.4f}')
