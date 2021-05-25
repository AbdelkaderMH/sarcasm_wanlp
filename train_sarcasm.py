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


def train(base_model, mt_classifier, iterator, optimizer, sar_criterion, scheduler):

    # set the model in eval phase
    base_model.train(True)
    mt_classifier.train(True)
    acc_sarcasm= 0
    loss_sarc= 0

    for data_input, label_input  in Bar(iterator):

        for k, v in data_input.items():
            data_input[k] = v.to(device)

        for k, v in label_input.items():
            label_input[k] = v.to(device)

        optimizer.zero_grad()


        #forward pass

        sarcasm_target = label_input['sarcasm']

        # forward pass

        output, pooled = base_model(**data_input)
        sarcasm_logits = mt_classifier(output, pooled)
        sarcasm_probs = torch.sigmoid(sarcasm_logits).to(device)

        # compute the loss
        loss_sarcasm = sar_criterion(sarcasm_probs.squeeze(), sarcasm_target)
        #total_loss = multi_task_loss(loss_sentiment, loss_sarcasm)
        loss_sarc += loss_sarcasm.item()
        # backpropage the loss and compute the gradients
        loss_sarcasm.backward()
        optimizer.step()
        scheduler.step()
        acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)

    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator)}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses

def evaluate(base_model, mt_classifier, iterator, sar_criterion):
    # initialize every epoch
    acc_sarcasm= 0
    loss_sarc= 0

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


            sarcasm_target = label_input['sarcasm']

            # forward pass

            output, pooled = base_model(**data_input)
            sarcasm_logits = mt_classifier(output, pooled)

            sarcasm_probs = torch.sigmoid(sarcasm_logits).to(device)
            # compute the loss
            loss_sarcasm = sar_criterion(sarcasm_probs.squeeze(), sarcasm_target)
            #mtl_loss = multi_task_loss(loss_sentiment, loss_sarcasm)

            # compute the running accuracy and losses
            acc_sarcasm += utils.binary_accuracy(sarcasm_probs, sarcasm_target)


            loss_sarc += loss_sarcasm.item()

            predicted_sarcasm = torch.round(sarcasm_probs)
            all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy())
            all_sarcasm_labels.extend(sarcasm_target.squeeze().int().cpu().numpy())

    fscore_sarcasm = f1_score(all_sarcasm_outputs, all_sarcasm_labels, pos_label=1, average='binary')
    report_sarcasm = classification_report(all_sarcasm_outputs, all_sarcasm_labels, target_names=['False', 'True'],digits=4)


    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator), 'F1_sarcasm': fscore_sarcasm, 'Report_sarcasm': report_sarcasm}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses

def train_full(config, train_loader, stest_loader):
    lr_o = config['lr_mult'] * config['lr']
    lr = config['lr']

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    mtl_classifier = modeling.CLSlayer(base_model.output_num(), class_num=1).to(device)
    cls = 'CLSlayer'


    ## set optimizer and criterions

    sarc_criterion = nn.BCELoss().to(device)

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
    epo = 0
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, mtl_classifier, train_loader, optimizer, sarc_criterion,scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, mtl_classifier, valid_loader, sarc_criterion)
        #print(multi_task_loss.parameters())
        val_loss = valid_losses['Sarcasm']
        total_val_acc = valid_accuracies['F1_sarcasm']
        if total_val_acc > best_total_val_acc:
        #if best_val_loss > val_loss:
            epo = epoch
            best_val_loss = val_loss
            best_total_val_acc = total_val_acc
            best_sarcasm_valid_accuracy = valid_accuracies['F1_sarcasm']
            report_sarcasm = valid_accuracies['Report_sarcasm']
            best_sarcasm_loss = valid_losses['Sarcasm']
            print("save model's checkpoint")
            torch.save(base_model.state_dict(), "./ckpts/best_basemodel_sarcasm_"+config["lm"]+".pth")
            torch.save(mtl_classifier.state_dict(), "./ckpts/best_mtl_cls_sarcasm_"+config["lm"]+".pth")


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
    with open(f'reports/report_Sarcasm_ST_model_{cls}_v2.txt', 'w') as f:
        f.write("Sarcasm report\n")
        f.write(report_sarcasm)
    return best_sarcasm_valid_accuracy, best_sarcasm_loss
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
        best_sarcasm_acc, best_sarcasm_loss =train_full(config, train_loader, valid_loader)
        print(f'  Val. Sarcasm F1: {best_sarcasm_acc * 100:.2f}%  \t Val Sarcasm Loss {best_sarcasm_loss :.4f} ')
