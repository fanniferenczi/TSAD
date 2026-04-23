from data.data_loader_dad import (
    PSM,
    NASA_Anomaly,
    WADI,
    SWaT,
    GECCO
)
from exp.exp_basic import Exp_Basic
from models.gta import GTA

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from sklearn.metrics import classification_report

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_GTA_DAD(Exp_Basic):
    def __init__(self, args):
        super(Exp_GTA_DAD, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'gta':GTA,
        }
        if self.args.model=='gta':
            model = model_dict[self.args.model](
                self.args.num_nodes,
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.num_levels,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.data,
                self.args.activation,
                self.device
            )
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'SMAP':NASA_Anomaly,
            'MSL':NASA_Anomaly,
            'SMD': NASA_Anomaly,
            'WADI':WADI,
            'SWaT':SWaT,
            'PSM': PSM,
            'GECCO': GECCO,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_label) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)

            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # decoder input
            # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
            batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

                loss = criterion(outputs, batch_y) + \
                        torch.sum(torch.abs(self.model.gt_embedding.gc_module.logits[:, 0]))
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        labels = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_label) in enumerate(test_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                # dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:,-self.args.pred_len:,:].to(self.device)

                pred = outputs.detach().cpu().numpy()#.squeeze()
                true = batch_y.detach().cpu().numpy()#.squeeze()
                batch_label = batch_label.long().detach().numpy()
                
                preds.append(pred)
                trues.append(true)
                labels.append(batch_label)

        preds = np.array(preds)
        trues = np.array(trues)
        labels = np.array(labels)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path+'label.npy', labels)

        # --- Anomaly scoring and evaluation ---

        # Step 1: compute per-timestamp anomaly score (sum of squared errors across all sensors)
        # preds and trues shape: (n_timestamps, pred_len, n_sensors)
        # We take the last pred_len step and sum squared errors across sensors
        errors = np.sum((preds - trues) ** 2, axis=-1)  # (n_timestamps, pred_len)
        anomaly_scores = errors[:, -1]                   # take last predicted step -> (n_timestamps,)

        # labels shape: (n_timestamps, pred_len) - take last step to match
        anomaly_labels = labels[:, -1]                   # (n_timestamps,)

        # Step 2: point-adjust protocol (following Chen et al., 2022, §V-B-2)
        # If any point in a contiguous anomaly segment is detected, the whole segment is credited
        def point_adjust(scores, labels, threshold):
            preds_binary = (scores > threshold).astype(int)
            # find contiguous anomaly segments in ground truth
            adjusted = preds_binary.copy()
            in_anomaly = False
            segment_detected = False
            segment_start = 0
            for i in range(len(labels)):
                if labels[i] == 1 and not in_anomaly:
                    in_anomaly = True
                    segment_start = i
                    segment_detected = False
                if in_anomaly:
                    if preds_binary[i] == 1:
                        segment_detected = True
                    if labels[i] == 0 or i == len(labels) - 1:
                        if segment_detected:
                            adjusted[segment_start:i] = 1
                        in_anomaly = False
            return adjusted

        # Step 3: grid search over thresholds to find best F1
        from sklearn.metrics import precision_score, recall_score, f1_score

        thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
        best_f1 = 0
        best_thresh = 0
        best_prec = 0
        best_rec = 0

        for thresh in thresholds:
            adjusted_preds = point_adjust(anomaly_scores, anomaly_labels, thresh)
            f1 = f1_score(anomaly_labels, adjusted_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_prec = precision_score(anomaly_labels, adjusted_preds, zero_division=0)
                best_rec = recall_score(anomaly_labels, adjusted_preds, zero_division=0)

        # Step 4: compute AUC-ROC using raw anomaly scores
        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(anomaly_labels, anomaly_scores)
        except ValueError:
            # roc_auc_score fails if only one class present in labels
            auc = 0.0
            print('Warning: AUC could not be computed (only one class in labels)')

        print(f'AUC-ROC:   {auc:.4f}')
        print(f'Best threshold: {best_thresh:.6f}')
        print(f'Precision: {best_prec:.4f}')
        print(f'Recall:    {best_rec:.4f}')
        print(f'F1 Score:  {best_f1:.4f}')

        # save results
        np.save(folder_path+'anomaly_scores.npy', anomaly_scores)
        results_dict = {'precision': best_prec, 'recall': best_rec, 'f1': best_f1, 'threshold': best_thresh, 'auc': auc}
        print('Anomaly detection results:', results_dict)

        return