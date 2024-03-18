import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from loss import CE, Align, Reconstruct,CM
from torch.optim.lr_scheduler import LambdaLR
from classification import fit_lr, get_rep_with_label,get_freqrep_with_label
from model.BrainVisModels import AlignNet,TimeFreqEncoder,FreqEncoder
import argparse

parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-mt','--model_type', default='FreqEncoder', help='')
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='lstm__subject0_epoch_900.pth', help="path to pre-trained net")

# Parse arguments
opt = parser.parse_args()

def l1_regularization(model, lambda_):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += param.abs().sum()
    l1_penalty = lambda_ * l1_norm
    return l1_penalty

class Trainer():
    def __init__(self, args, time_model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = time_model.to(torch.device(self.device))

        self.train_loader = train_loader
        #self.train_linear_loader = train_linear_loader
        self.train_linear_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.cr = CE(self.model)
        self.alpha = args.alpha
        self.beta = args.beta

        self.test_cr = torch.nn.CrossEntropyLoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_acc = 0
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        for epoch in range(self.num_epoch_pretrain):
            print('Epoch:' + str(epoch+1))
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad() 
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))

            if (epoch + 1) % 20 == 0:
                     torch.save(self.model.state_dict(), self.save_path + '/pretrain_model_epoch'+str(epoch+1)+'.pkl')

            if (epoch + 1) % 3 == 0:
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')
                    # It is worth noting that the highest pretraining accuracy does not mean the model is the
                    # best one for finetuning, so the one with larger training epoch should be used.

    def finetune(self):
        print('finetune')
        self.model.linear_proba = True
        #self.args.load_pretrained_model=False
        if self.args.load_pretrained_model:
            print('load pretrained model')
            state_dict = torch.load(self.save_path + '/pretrain_model_epoch300.pkl', map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except:
                model_state_dict = self.model.state_dict()
                for pretrain, random_intial in zip(state_dict, model_state_dict):
                    assert pretrain == random_intial
                    if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                    'predict_head.bias', 'position.pe.weight']:
                        state_dict[pretrain] = model_state_dict[pretrain]
                self.model.load_state_dict(state_dict)

        self.model.eval()
        train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
        test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
        clf = fit_lr(train_rep, train_label)
        acc = clf.score(test_rep, test_label)
        pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
        f1 = f1_score(test_label, pred_label, average='macro')
        print(acc, f1)

        self.model.linear_proba = False #If linear_proba = True, freeze pretrained model, train only classifier
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)

        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.print_process(
                'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))

            if (epoch + 1) % 5 == 0:
               torch.save(self.model.state_dict(),
                          self.save_path + '/finetune_model_epoch' + str(epoch + 1) + '.pkl')

        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader
        loss_sum = 0
        pos=0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            l1=l1_regularization(self.model,0.000003)
            loss = self.cr.computeft(batch)#+l1
            loss_sum += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            pos=pos+1
            self.step += 1
        # if self.step % self.eval_per_steps == 0:
        metric = self.eval_model()
        self.print_process(metric)

        if metric[self.metric] >= self.best_metric:
            torch.save(self.model.state_dict(), self.save_path + '/finetune_model.pkl')
            self.best_metric = metric[self.metric]
        self.model.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        print("aaa")
        print(len(label))
        confusion_mat = self._confusion_mat(label, pred)
        self.print_process(confusion_mat)
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        seqs, label, clip, clip_moreinf = batch
        lastrep, rep,scores = self.model(seqs)
        _, pred = torch.topk(scores, 1)
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def compute_metrics_freq(self, batch,model):
        #if len(batch) == 2:
        seqs, label,clip,clip_moreinf = batch
        lastrep, rep,scores = model(seqs)
        #else:
        #    seqs1, seqs2, label = batch
        #    lastrep, rep, scores = self.model((seqs1, seqs2))
        _, pred = torch.topk(scores, 1)
        #print(np.shape(scores))
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)

    def cont_pretrain(self):
        start_epoch=300
        state_dict = torch.load(self.save_path + '/pretrain_model_epoch300.pkl', map_location=self.device)
        eval_acc=0.0 # It should be modified.
        self.model.load_state_dict(state_dict)
        print('cont_pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()

        for epoch in range(self.num_epoch_pretrain):
            if(epoch<start_epoch):
                continue
            print('Epoch:' + str(epoch + 1))
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.save_path + '/pretrain_model_epoch'+str(epoch+1)+'.pkl')

            if (epoch + 1) % 3 == 0:
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')

    def finetune_CLIP(self):
        eval_cosine = 0.0
        freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                         (key, value) in [x.split("=") for x in opt.model_params]}
        freq_model = FreqEncoder(**freq_model_options)

        self.timefreq_model=TimeFreqEncoder(self.model,freq_model,self.args)
        self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

        freqtime_state_dict = torch.load(self.save_path + '/timefreqmodel.pkl', map_location=self.device)

        self.timefreq_model.load_state_dict(freqtime_state_dict)

        self.timefreq_model.to(torch.device("cpu"))

        freq_size=freq_model.output_size
        time_size=self.model.d
        clip_size=int(77*768)

        self.alignmodel=AlignNet(time_size,freq_size,clip_size,self.timefreq_model)
        self.alignmodel=self.alignmodel.to(torch.device(self.device))
        print('CLIP_finetune')
        self.optimizer = torch.optim.AdamW(self.alignmodel.parameters(), lr=self.args.lr)
        CLIPloss = CM()
        align=Align()

        for epoch in range(500):
            print('Epoch:' + str(epoch + 1))
            self.alignmodel.train()
            tqdm_dataloader = tqdm(self.train_loader)
            test_tqdm_dataloader=tqdm(self.test_loader)
            loss_clip=0
            loss_mse=0
            loss_clip_moreinf=0
            loss_mse_moreinf=0
            loss_sum = 0

            teloss_clip=0
            teloss_mse=0
            teloss_clip_moreinf=0
            teloss_mse_moreinf=0
            teloss_sum = 0

            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()

                clippred = self.alignmodel.forward(batch[0].float())
                CLIP_loss = CLIPloss.compute(clippred.float(), batch[2].float())
                CLIP_loss_moreinf=CLIPloss.compute(clippred.float(), batch[3].float())
                align_loss=align.compute(clippred.float(), batch[2].float())
                align_loss_moreinf = align.compute(clippred.float(), batch[3].float())
                All_CLIP_loss=CLIP_loss+CLIP_loss_moreinf
                All_align_loss=align_loss+align_loss_moreinf

                loss_clip+= CLIP_loss.item()
                loss_mse+= align_loss.item()
                loss_clip_moreinf+= CLIP_loss_moreinf.item()
                loss_mse_moreinf+= align_loss_moreinf.item() #MSE, due to numerical considerations
                # lambda_value = 0.000002
                # l1_penalty = l1_regularization(self.model, lambda_value)
                loss = All_align_loss+All_CLIP_loss#+l1_penalty
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            trloss=loss_sum / (idx + 1)
            trmse=loss_mse / (idx + 1)
            trmse_moreinf=loss_mse_moreinf / (idx + 1)
            trcosine=loss_clip / (idx + 1)
            trcosine_moreinf=loss_clip_moreinf / (idx + 1)

            for idxte, batch in enumerate(test_tqdm_dataloader):
                self.alignmodel.eval()
                batch = [x.to(self.device) for x in batch]
                clippred= self.alignmodel(batch[0])
                CLIP_loss = CLIPloss.compute(clippred, batch[2])
                CLIP_loss_moreinf = CLIPloss.compute(clippred.float(), batch[3].float())
                align_loss=align.compute(clippred, batch[2])
                align_loss_moreinf = align.compute(clippred.float(), batch[3].float())
                All_CLIP_loss = CLIP_loss + CLIP_loss_moreinf
                All_align_loss = align_loss + align_loss_moreinf

                teloss_clip+= CLIP_loss.item()
                teloss_mse+= align_loss.item()
                teloss_clip_moreinf+= CLIP_loss_moreinf.item()
                teloss_mse_moreinf+= align_loss_moreinf.item()
                teloss = All_align_loss+All_CLIP_loss
                teloss_sum += teloss.item()

            teloss = teloss_sum / (idxte + 1)
            temse = teloss_mse / (idxte + 1)
            tecosine = teloss_clip / (idxte + 1)
            temse_moreinf = teloss_mse_moreinf / (idxte + 1)
            tecosine_moreinf = teloss_clip_moreinf / (idxte + 1)

            print('clip_finetune epoch{0}, trloss{1}, trmse{2}, trcosine{3},trmse_moreinf{4},trcosine_moreinf{5}, '
                  'teloss{6}, temse{7}, tecosine{8},temse_moreinf{9}, tecosine_moreinf{10}'.format(epoch + 1,
             trloss,trmse,trcosine,trmse_moreinf,trcosine_moreinf,teloss,temse,tecosine,temse_moreinf,tecosine_moreinf))


            if (epoch + 1) % 10 == 0:
                torch.save(self.alignmodel.state_dict(),
                           self.save_path + '/clipfinetune_model_epoch' + str(epoch + 1) + 'MAEchange.pkl')

            if tecosine > eval_cosine:
                eval_cosine = tecosine
                torch.save(self.alignmodel.state_dict(), self.save_path + '/clipfinetune_model.pkl')

    def finetune_timefreq(self):
        time_state_dict = torch.load(self.save_path + '/finetune_model_epoch80.pkl',
                                map_location=self.device)
        print("freq_train")

        self.model.load_state_dict(time_state_dict)

        self.model.eval()
        self.model.to(torch.device("cuda"))
        train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
        test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
        clf = fit_lr(train_rep, train_label)
        acc = clf.score(test_rep, test_label)
        pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
        f1 = f1_score(test_label, pred_label, average='macro')
        print(acc, f1)
        self.model.train()
        self.model.to(torch.device("cpu"))

        freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                         (key, value) in [x.split("=") for x in opt.model_params]}
        freq_model = FreqEncoder(**freq_model_options)

        if opt.pretrained_net != '':
            freq_model = torch.load(opt.pretrained_net)

        self.timefreq_model=TimeFreqEncoder(self.model,freq_model,self.args)
        self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

        self.optimizer = torch.optim.AdamW(self.timefreq_model.parameters(), lr=self.args.lr)
        cr_freq = CE(self.timefreq_model)
        eval_acc = 0

        for epoch in range(50):
            print('Epoch:' + str(epoch + 1))
            self.timefreq_model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            test_tqdm_dataloader=tqdm(self.test_loader)
            loss_sum = 0

            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                loss=cr_freq.computefreq(batch)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            trloss=loss_sum / (idx + 1)

            metrics = {'acc': 0, 'f1': 0}
            pred = []
            label = []
            test_loss = 0

            for idxte, batch in enumerate(test_tqdm_dataloader):
                self.timefreq_model.eval()
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics_freq(batch,self.timefreq_model)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
            confusion_mat = self._confusion_mat(label, pred)
            self.print_process(confusion_mat)

            if self.args.num_class == 2:
                metrics['f1'] = f1_score(y_true=label, y_pred=pred)
                metrics['precision'] = precision_score(y_true=label, y_pred=pred)
                metrics['recall'] = recall_score(y_true=label, y_pred=pred)
            else:
                metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
                metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
            metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
            metrics['test_loss'] = test_loss / (idxte + 1)

            print('timefreq_finetune epoch{0}, trloss{1}, teloss{2},teacc{3}'.format(epoch + 1, trloss, metrics['test_loss'],metrics['acc']))

            if (epoch + 1) % 5 == 0:
                torch.save(self.timefreq_model.state_dict(),
                           self.save_path + '/timefreqmodel_epoch' + str(epoch + 1) + '.pkl')

            if metrics['acc'] > eval_acc:
                eval_acc = metrics['acc']
                torch.save(self.timefreq_model.state_dict(), self.save_path + '/timefreqmodel.pkl')
