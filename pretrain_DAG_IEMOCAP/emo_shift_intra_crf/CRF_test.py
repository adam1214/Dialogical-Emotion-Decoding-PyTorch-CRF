import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from CRF_train import CRF
import os
import pdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_dialog(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def view_new_matrix(model):
    with torch.no_grad():
        multiplier_after_softmax = model.multiplier_softmax(model.multiplier)
        for i in range(0,4,1):
            multiplier_after_softmax[i][i] = -1.
        multiplier_after_softmax_6_6 = torch.zeros([6,6])
        for i in range(0,4,1):
            for j in range(0,4,1):
                multiplier_after_softmax_6_6[i][j] = multiplier_after_softmax[i][j]
                
        weight_for_emo_shift_in_activate = torch.cat([model.weight_for_emo_shift_in_activate.expand(1, 4), torch.zeros(1,2)], dim=1) #(1,6)
        weight_for_emo_with_shift_out_activate = torch.cat([model.weight_for_emo_with_shift_out_activate.expand(1, 4), torch.zeros(1,2)], dim=1)
        weight_for_emo_no_shift_out_activate = torch.cat([model.weight_for_emo_no_shift_out_activate.expand(1, 4), torch.zeros(1,2)], dim=1)
        
        weight_for_emo_shift_in_activate = torch.cat([weight_for_emo_shift_in_activate, weight_for_emo_shift_in_activate, weight_for_emo_shift_in_activate, weight_for_emo_shift_in_activate, torch.zeros(1,6), torch.zeros(1,6)], dim=0)
        weight_for_emo_with_shift_out_activate = torch.cat([weight_for_emo_with_shift_out_activate, weight_for_emo_with_shift_out_activate, weight_for_emo_with_shift_out_activate, weight_for_emo_with_shift_out_activate, torch.zeros(1,6), torch.zeros(1,6)], dim=0)
        weight_for_emo_no_shift_out_activate = torch.cat([weight_for_emo_no_shift_out_activate, weight_for_emo_no_shift_out_activate, weight_for_emo_no_shift_out_activate, weight_for_emo_no_shift_out_activate, torch.zeros(1,6), torch.zeros(1,6)], dim=0)
        
        new_matrix_case1 = model.transitions + weight_for_emo_no_shift_out_activate*model.activate_fun((0-0.5)*weight_for_emo_shift_in_activate)*multiplier_after_softmax_6_6
        print('#####INTRA:')
        print(pd.DataFrame((np.load('intra_trans.npy')[:4,:4])).round(2))
        print('#####CASE1: NO SHIFT')
        print(pd.DataFrame((np.array(new_matrix_case1.data)[:4,:4])).round(2))
        new_matrix_case2 = model.transitions + weight_for_emo_with_shift_out_activate*model.activate_fun((1-0.5)*weight_for_emo_shift_in_activate)*multiplier_after_softmax_6_6
        print('#####CASE2: WITH SHIFT')
        print(pd.DataFrame(np.array(new_matrix_case2.data)[:4,:4]).round(2))
        
        fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[1,1,1.25]))
        fig.set_size_inches(12, 3) 
        global_max = -100
        global_min = 100
        for matrix in [np.load('intra_trans.npy')[:4,:4], np.array(new_matrix_case1.data)[:4,:4], np.array(new_matrix_case2.data)[:4,:4]]:
            local_min = matrix.min()
            local_max = matrix.max()
            if global_max < local_max:
                global_max = local_max
            if global_min > local_min:
                global_min = local_min
        sns.set(font_scale=3)
        sns.heatmap(np.load('intra_trans.npy')[:4,:4], annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=['ang', 'hap', 'neu', 'sad'], ax=axs[0], cbar=False, vmin=global_min, vmax=global_max, square=True, cmap="gray_r", annot_kws={"size": 14})
        sns.heatmap(np.array(new_matrix_case1.data)[:4,:4], annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=False, ax=axs[1], cbar=False, vmin=global_min, vmax=global_max, square=True, cmap="gray_r", annot_kws={"size": 14})
        sns.heatmap(np.array(new_matrix_case2.data)[:4,:4], annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=False, ax=axs[2], cbar=True, vmin=global_min, vmax=global_max, square=True, cbar_kws={"shrink": 1}, cmap="gray_r", annot_kws={"size": 14})
        #plt.show()
        plt.savefig('matrix_heatmap.png')
        '''
        print('#####TERM_1 (How much to subtract/add in CASE1)')
        print(pd.DataFrame(np.array(weight_for_emo_no_shift_out_activate*model.activate_fun((0-0.5)*weight_for_emo_shift_in_activate).data).round(2)))
        print('#####TERM_1 (How much to subtract/add in CASE2)')
        print(pd.DataFrame(np.array(weight_for_emo_with_shift_out_activate*model.activate_fun((1-0.5)*weight_for_emo_shift_in_activate).data).round(2)))
        print('#####multiplier_after_softmax (Diagonal value (-1) is constant)')
        print(pd.DataFrame(np.array(multiplier_after_softmax_6_6)).round(2))
        print('#####weight_for_emo_shift_in_activate')
        print(pd.DataFrame(np.array(weight_for_emo_shift_in_activate.data)).round(2))
        print('#####weight_for_emo_no_shift_out_activate')
        print(pd.DataFrame(np.array(weight_for_emo_no_shift_out_activate.data)).round(2))
        print('#####weight_for_emo_with_shift_out_activate')
        print(pd.DataFrame(np.array(weight_for_emo_with_shift_out_activate.data)).round(2))
        '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use?\niemocap_original_4\niemocap_original_6\niemocap_U2U_4\niemocap_U2U_6\nmeld", default = 'iemocap_original_4')
    parser.add_argument("-s", "--seed", type=int, help="select torch seed", default = 1)
    parser.add_argument("-e", "--emo_shift", type=str, help="which emo_shift prob. to use?", default = 'model')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5
    
    if args.dataset == 'iemocap_original_4':
        emo_dict = joblib.load('../data/iemocap/four_type/emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/four_type/dialog_rearrange_4emo_iemocap.pkl')
        #emo_dict = joblib.load('../data/iemocap/emo_all_iemocap_4.pkl')
        #dias = joblib.load('../data/iemocap/dialog_edit.pkl')
    elif args.dataset == 'iemocap_original_6':
        emo_dict = joblib.load('../data/iemocap/new_audio_text/emo_all_6.pkl')
        dias = joblib.load('../data/iemocap/new_audio_text/dialog_rearrange_6_emo.pkl')
        #emo_dict = joblib.load('../data/iemocap/old_text/emo_all_iemocap_6.pkl')
        #dias = joblib.load('../data/iemocap/old_text/dialog_edit.pkl')
    elif args.dataset == 'iemocap_U2U_4':
        emo_dict = joblib.load('../data/iemocap/U2U_4emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/dialog.pkl')
    elif args.dataset == 'iemocap_U2U_6':
        emo_dict = joblib.load('../data/iemocap/U2U_6emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/dialog.pkl')
    elif args.dataset == 'meld':
        emo_dict = joblib.load('../data/meld/emo_all_meld.pkl')
        dias = joblib.load('../data/meld/dialog_meld.pkl')

    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U_6':
        emo_to_ix = {'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5, START_TAG: 6, STOP_TAG: 7}
    elif args.dataset == 'iemocap_original_4' or args.dataset == 'iemocap_U2U_4':
        emo_to_ix = {'hap':1, 'neu':2, 'sad':3, 'ang':0, START_TAG: 4, STOP_TAG: 5}
    elif args.dataset == 'meld':
        emo_to_ix = {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6, START_TAG: 7, STOP_TAG: 8}

    if args.dataset == 'iemocap_original_4' or args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U_4' or args.dataset == 'iemocap_U2U_6':
        out_dict = joblib.load('../data/iemocap/four_type/outputs_4_text_audio.pkl')
        #out_dict = joblib.load('../data/iemocap/old_text/outputs_6_text.pkl')
        
    elif args.dataset == 'meld':
        out_dict = {}
    
    if args.emo_shift == 'constant':
        spk_dialogs = utils.split_dialog(dias)
        bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
    else:
        if 'iemocap' in args.dataset:
            bias_dict_label = joblib.load('../data/iemocap/four_type/4emo_shift_all_rearrange.pkl')
            bias_dict = joblib.load('../data/iemocap/four_type/DAG_emo_shift_output_text_audio.pkl')
            #bias_dict_label = joblib.load('../data/iemocap/old_text/6emo_shift_all.pkl')
            #bias_dict = joblib.load('../data/iemocap/old_text/DAG_emo_shift_output_text.pkl')
            #bias_dict = bias_dict_label
            p, g = [], []
            for utt in bias_dict_label:
                if 'Ses05' in utt:
                    if bias_dict[utt] > 0.5:
                        p.append(1)
                        #emo_shift_list.append(bias_dict[utt])
                    else:
                        p.append(0)
                        #emo_no_shift_list.append(bias_dict[utt])
                    g.append(int(bias_dict_label[utt]))
        else:
            bias_dict = joblib.load('../data/meld/6emo_shift_all.pkl')
        
        print('## EMO_SHIFT MODEL PERFORMANCE ##')
        print(len(p), len(g))
        print('UAR:', round(recall_score(g, p, average='macro')*100, 2), '%')
        print('RECALL 2 type:', recall_score(g, p, average=None))
        print('precision 2 type:', precision_score(g, p, average=None))
        print('WEIGHTED F1:', round(f1_score(g, p, average='weighted')*100, 2), '%')
        print(confusion_matrix(g, p))
        print('##########')
        
    # Make up training data & testing data
    train_data = []
    val_data = []
    test_data = []
    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U' or args.dataset == 'iemocap_original_4':
        val_dias_set = joblib.load('../data/iemocap/val_dias_set.pkl')
        for dialog in dias:
            if dialog[4] == '5':
                test_data.append(([],[]))
                test_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        test_data[-2][0].append(utt)
                        test_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        test_data[-1][0].append(utt)
                        test_data[-1][1].append(emo_dict[utt])
                if len(test_data[-2][0]) == 0:
                    del test_data[-2]
                if len(test_data[-1][0]) == 0:
                    del test_data[-1]
                    
            elif dialog in val_dias_set:
                val_data.append(([],[]))
                val_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        val_data[-2][0].append(utt)
                        val_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        val_data[-1][0].append(utt)
                        val_data[-1][1].append(emo_dict[utt])
                if len(val_data[-2][0]) == 0:
                    del val_data[-2]
                if len(val_data[-1][0]) == 0:
                    del val_data[-1]
                    
            else:
                train_data.append(([],[]))
                train_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        train_data[-2][0].append(utt)
                        train_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        train_data[-1][0].append(utt)
                        train_data[-1][1].append(emo_dict[utt])
                if len(train_data[-2][0]) == 0:
                    del train_data[-2]
                if len(train_data[-1][0]) == 0:
                    del train_data[-1]
                    
    elif args.dataset == 'meld':
        for dialog in dias:
            if dialog.split('_')[0] == 'train':
                train_data.append((dias[dialog],[]))
                for utt in train_data[-1][0]:
                    train_data[-1][1].append(emo_dict[utt])
            elif dialog.split('_')[0] == 'val':
                val_data.append((dias[dialog],[]))
                for utt in val_data[-1][0]:
                    val_data[-1][1].append(emo_dict[utt])
            elif dialog.split('_')[0] == 'test':
                test_data.append((dias[dialog],[]))
                for utt in test_data[-1][0]:
                    test_data[-1][1].append(emo_dict[utt])
    
    utt_to_ix = {}
    for dialog, emos in train_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in val_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)

    ix_to_utt = {}
    for key in utt_to_ix:
        val = utt_to_ix[key]
        ix_to_utt[val] = key

    # Load model
    model = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model.to(device)
    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_original_4':
        checkpoint = torch.load('./model/iemocap/original/model' + str(args.seed) + '.pth')
    elif args.dataset == 'meld':
        checkpoint = torch.load('./model/meld/model' + str(args.seed) + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # inference
    predict = []
    pred_dict = {}
    with torch.no_grad():
        for i in range(0, len(test_data), 1):
            precheck_dia = prepare_dialog(test_data[i][0], utt_to_ix)
            tmp = model(precheck_dia, test_data[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data[i][0]):
                pred_dict[utt] = tmp[j]
                
    ori_emo_dict = joblib.load('../data/iemocap/four_type/emo_all_iemocap.pkl')
    #ori_emo_dict = joblib.load('../data/iemocap/new_audio_text/emo_all_6.pkl')
    #ori_emo_dict = joblib.load('../data/iemocap/old_text/emo_all_iemocap_6.pkl')
    label = []
    for dia_emos_tuple in test_data:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])

    for i in range(0, len(label), 1):
        if emo_to_ix.get(label[i]) != None:
            label[i] = emo_to_ix.get(label[i])
        else:
            label[i] = -1
    
    uar, acc, f1, conf = utils.evaluate(predict, label, final_test=1)
    #print('UAR:', uar)
    #print('ACC:', acc)
    print('WEIGHTED F1:', round(100*f1, 2), '%')
    print(conf)
    '''
    path = 'uar.txt'
    f = open(path, 'a')
    f.write(str(uar)+'\n')
    f.close()
    
    path = 'acc.txt'
    f = open(path, 'a')
    f.write(str(acc)+'\n')
    f.close()
    '''
    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_original_4':
        joblib.dump(pred_dict, './model/iemocap/original/preds_4.pkl')
    else:
        joblib.dump(pred_dict, './model/meld/original/preds_7.pkl')
    
    # ensure pretrained model performance
    labels = []
    predicts = []
    if 'iemocap' in args.dataset:
        dialogs_edit = joblib.load('../data/iemocap/four_type/dialog_rearrange_4emo_iemocap.pkl')
        emo_dict = joblib.load('../data/iemocap/four_type/emo_all_iemocap.pkl')
        #dialogs_edit = joblib.load('../data/iemocap/new_audio_text/dialog_rearrange_6_emo.pkl')
        #emo_dict = joblib.load('../data/iemocap/new_audio_text/emo_all_6.pkl')
        #dialogs_edit = joblib.load('../data/iemocap/old_text/dialog_edit.pkl')
        #emo_dict = joblib.load('../data/iemocap/old_text/emo_all_iemocap_6.pkl')
        #emo2num ={'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5, START_TAG: 6, STOP_TAG: 7}
        emo2num = {'neu':2, 'sad':3, 'hap':1, 'ang':0, START_TAG: 6, STOP_TAG: 7}
    else:
        pass
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            if utt[4] == '5':
                labels.append(emo2num[emo_dict[utt]])
                predicts.append(out_dict[utt].index(max(out_dict[utt])))
            
    #print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    #print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')
    print('pretrained WEIGHTED F1:', round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    
    #analysis: new matrix(case1: emo_shift prob.=0 & case2:emo_shift prob.=1)
    #view_new_matrix(model)