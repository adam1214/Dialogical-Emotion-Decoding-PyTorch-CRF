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

def view_new_matrix(model, fold_num):
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
        
        np.save('ESA_no_shift_trans_fold' + str(fold_num) + '.npy', np.array(new_matrix_case1.data)[:4,:4])
        np.save('ESA_shift_trans_fold' + str(fold_num) + '.npy', np.array(new_matrix_case2.data)[:4,:4])
        
        '''
        fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[1,1.25]))
        #fig.set_size_inches(12, 3) 
        global_max = -100
        global_min = 100
        for matrix in [np.array(new_matrix_case1.data)[:4,:4], np.array(new_matrix_case2.data)[:4,:4]]:
            local_min = matrix.min()
            local_max = matrix.max()
            if global_max < local_max:
                global_max = local_max
            if global_min > local_min:
                global_min = local_min
        sns.set(font_scale=1.0)
        sns.heatmap(np.array(new_matrix_case1.data)[:4,:4], annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=['ang', 'hap', 'neu', 'sad'], ax=axs[0], cbar=False, vmin=global_min, vmax=global_max, square=True, cmap="gray_r", annot_kws={"size": 14})
        sns.heatmap(np.array(new_matrix_case2.data)[:4,:4], annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=False, ax=axs[1], cbar=True, vmin=global_min, vmax=global_max, square=True, cbar_kws={"shrink": 1}, cmap="gray_r", annot_kws={"size": 14})
        #plt.show()
        plt.savefig('matrix_heatmap.png')
        '''
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
    parser.add_argument('-v', "--pretrain_version", type=str, help="which version of pretrain model you want to use?", default='original_output')
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    parser.add_argument("-e", "--emo_shift", type=str, help="which emo_shift prob. to use?", default = 'model')
    args = parser.parse_args()
    
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5
    '''
    output_fold1 = joblib.load('../data/original_output/utt_logits_outputs_fold1.pkl')
    output_fold2 = joblib.load('../data/original_output/utt_logits_outputs_fold2.pkl')
    output_fold3 = joblib.load('../data/original_output/utt_logits_outputs_fold3.pkl')
    output_fold4 = joblib.load('../data/original_output/utt_logits_outputs_fold4.pkl')
    output_fold5 = joblib.load('../data/original_output/utt_logits_outputs_fold5.pkl')
    
    out_dict = {}
    for utt in output_fold1:
        if utt[4] == '1':
            out_dict[utt] = output_fold1[utt]
        elif utt[4] == '2':
            out_dict[utt] = output_fold2[utt]
        elif utt[4] == '3':
            out_dict[utt] = output_fold3[utt]
        elif utt[4] == '4':
            out_dict[utt] = output_fold4[utt]
        elif utt[4] == '5':
            out_dict[utt] = output_fold5[utt]
    '''
    out_dict = joblib.load('../data/speech_only/outputs_4_all_fold_text.pkl')
    #dialogs = joblib.load('../data/dialog_iemocap.pkl')
    #dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    dialogs = joblib.load('../data/speech_only/dialogs_speech_only.pkl')
    dialogs_edit = joblib.load('../data/speech_only/dialogs_edit_speech_only.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/speech_only/emo_all_speech_only.pkl')
        dias = dialogs_edit
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/'+ args.pretrain_version + '/U2U_4emo_all.pkl')
        dias = dialogs
    
    bias_dict_label = joblib.load('../data/speech_only/4emo_shift_all_speech_only.pkl')
    if args.emo_shift == 'constant':
        spk_dialogs = utils.split_dialog(dias)
        bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
        for utt in bias_dict:
            bias_dict[utt] = 1.0
    else:
        bias_dict = joblib.load('../data/speech_only/rf_emo_shift_output_text.pkl')
        #bias_dict = bias_dict_label
        '''
        emo_prob_fold1 = joblib.load('../data/'+ args.pretrain_version + '/MLPPytorch_emo_shift_output_fold1.pkl')
        emo_prob_fold2 = joblib.load('../data/'+ args.pretrain_version + '/MLPPytorch_emo_shift_output_fold2.pkl')
        emo_prob_fold3 = joblib.load('../data/'+ args.pretrain_version + '/MLPPytorch_emo_shift_output_fold3.pkl')
        emo_prob_fold4 = joblib.load('../data/'+ args.pretrain_version + '/MLPPytorch_emo_shift_output_fold4.pkl')
        emo_prob_fold5 = joblib.load('../data/'+ args.pretrain_version + '/MLPPytorch_emo_shift_output_fold5.pkl')
        bias_dict = {}
        for utt in emo_prob_fold1:
            if utt[4] == '1':
                bias_dict[utt] = emo_prob_fold1[utt]
            elif utt[4] == '2':
                bias_dict[utt] = emo_prob_fold2[utt]
            elif utt[4] == '3':
                bias_dict[utt] = emo_prob_fold3[utt]
            elif utt[4] == '4':
                bias_dict[utt] = emo_prob_fold4[utt]
            elif utt[4] == '5':
                bias_dict[utt] = emo_prob_fold5[utt]
        '''
        '''
        for k in bias_dict:
            if bias_dict[k] > 0.5:
                bias_dict[k] = 1.0
            else:
                bias_dict[k] = 0.0
        '''
    p, g = [], []
    emo_shift_list, emo_no_shift_list, all_list = [], [], []
    for utt in bias_dict_label:
        if 'Ses0' in utt:
            all_list.append(bias_dict[utt])
            if bias_dict[utt] > 0.5:
                p.append(1)
                #emo_shift_list.append(bias_dict[utt])
            else:
                p.append(0)
                #emo_no_shift_list.append(bias_dict[utt])
            g.append(int(bias_dict_label[utt]))
            if int(bias_dict_label[utt]) == 1:
                emo_shift_list.append(bias_dict[utt])
            else:
                emo_no_shift_list.append(bias_dict[utt])
    print('AVG of emo_shift prob.:', round(np.mean(np.array(emo_shift_list)), 2), '+-', round(np.std(np.array(emo_shift_list), ddof=0), 2))
    print('AVG of emo_no_shift prob.:', round(np.mean(np.array(emo_no_shift_list)), 3), '+-', round(np.std(np.array(emo_no_shift_list), ddof=0), 3))
    print('AVG of all prob.:', round(np.mean(np.array(all_list)), 3), '+-', round(np.std(np.array(all_list), ddof=0), 3))
    print('## EMO_SHIFT MODEL PERFORMANCE ##')
    print(len(p), len(g))
    print('F1:', round(f1_score(g, p, average='weighted')*100, 2), '%')
    print('UAR:', round(recall_score(g, p, average='macro')*100, 2), '%')
    print('ACC:', round(accuracy_score(g, p)*100, 2), '%')
    print('UAR 2 type:', recall_score(g, p, average=None))
    print('precision 2 type:', precision_score(g, p, average=None))
    print(confusion_matrix(g, p))
    print('##########')
    
    test_data_Ses01 = []
    test_data_Ses02 = []
    test_data_Ses03 = []
    test_data_Ses04 = []
    test_data_Ses05 = []
    for dialog in dias:
        if dialog[4] == '1':
            test_data_Ses01.append(([],[]))
            test_data_Ses01.append(([],[]))
            for utt in dias[dialog]:
                if utt.split('_')[-2] == 'A':
                    test_data_Ses01[-2][0].append(utt)
                    test_data_Ses01[-2][1].append(emo_dict[utt])
                elif utt.split('_')[-2] == 'B':
                    test_data_Ses01[-1][0].append(utt)
                    test_data_Ses01[-1][1].append(emo_dict[utt])
            if len(test_data_Ses01[-2][0]) == 0:
                del test_data_Ses01[-2]
            if len(test_data_Ses01[-1][0]) == 0:
                del test_data_Ses01[-1]
        elif dialog[4] == '2':
            test_data_Ses02.append(([],[]))
            test_data_Ses02.append(([],[]))
            for utt in dias[dialog]:
                if utt.split('_')[-2] == 'A':
                    test_data_Ses02[-2][0].append(utt)
                    test_data_Ses02[-2][1].append(emo_dict[utt])
                elif utt.split('_')[-2] == 'B':
                    test_data_Ses02[-1][0].append(utt)
                    test_data_Ses02[-1][1].append(emo_dict[utt])
            if len(test_data_Ses02[-2][0]) == 0:
                del test_data_Ses02[-2]
            if len(test_data_Ses02[-1][0]) == 0:
                del test_data_Ses02[-1]
        elif dialog[4] == '3':
            test_data_Ses03.append(([],[]))
            test_data_Ses03.append(([],[]))
            for utt in dias[dialog]:
                if utt.split('_')[-2] == 'A':
                    test_data_Ses03[-2][0].append(utt)
                    test_data_Ses03[-2][1].append(emo_dict[utt])
                elif utt.split('_')[-2] == 'B':
                    test_data_Ses03[-1][0].append(utt)
                    test_data_Ses03[-1][1].append(emo_dict[utt])
            if len(test_data_Ses03[-2][0]) == 0:
                del test_data_Ses03[-2]
            if len(test_data_Ses03[-1][0]) == 0:
                del test_data_Ses03[-1]
        elif dialog[4] == '4':
            test_data_Ses04.append(([],[]))
            test_data_Ses04.append(([],[]))
            for utt in dias[dialog]:
                if utt.split('_')[-2] == 'A':
                    test_data_Ses04[-2][0].append(utt)
                    test_data_Ses04[-2][1].append(emo_dict[utt])
                elif utt.split('_')[-2] == 'B':
                    test_data_Ses04[-1][0].append(utt)
                    test_data_Ses04[-1][1].append(emo_dict[utt])
            if len(test_data_Ses04[-2][0]) == 0:
                del test_data_Ses04[-2]
            if len(test_data_Ses04[-1][0]) == 0:
                del test_data_Ses04[-1]
        elif dialog[4] == '5':
            test_data_Ses05.append(([],[]))
            test_data_Ses05.append(([],[]))
            for utt in dias[dialog]:
                if utt.split('_')[-2] == 'A':
                    test_data_Ses05[-2][0].append(utt)
                    test_data_Ses05[-2][1].append(emo_dict[utt])
                elif utt.split('_')[-2] == 'B':
                    test_data_Ses05[-1][0].append(utt)
                    test_data_Ses05[-1][1].append(emo_dict[utt])
            if len(test_data_Ses05[-2][0]) == 0:
                del test_data_Ses05[-2]
            if len(test_data_Ses05[-1][0]) == 0:
                del test_data_Ses05[-1]

    utt_to_ix = {}
    for dialog, emos in test_data_Ses01:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses02:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses03:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses04:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses05:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
                
    ix_to_utt = {}
    for key in utt_to_ix:
        val = utt_to_ix[key]
        ix_to_utt[val] = key

    emo_to_ix = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3, START_TAG: 4, STOP_TAG: 5}

    # Load model
    model_1 = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model_1.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses01.pth')
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_1.eval()

    model_2 = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model_2.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses02.pth')
    model_2.load_state_dict(checkpoint['model_state_dict'])
    model_2.eval()

    model_3 = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model_3.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses03.pth')
    model_3.load_state_dict(checkpoint['model_state_dict'])
    model_3.eval()

    model_4 = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model_4.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses04.pth')
    model_4.load_state_dict(checkpoint['model_state_dict'])
    model_4.eval()

    model_5 = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model_5.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses05.pth')
    model_5.load_state_dict(checkpoint['model_state_dict'])
    model_5.eval()
    
    # inference
    predict = []
    pred_dict = {}
    with torch.no_grad():
        for i in range(0, len(test_data_Ses01), 1):
            precheck_dia = prepare_dialog(test_data_Ses01[i][0], utt_to_ix)
            tmp = model_1(precheck_dia, test_data_Ses01[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses01[i][0]):
                pred_dict[utt] = tmp[j]
        
        for i in range(0, len(test_data_Ses02), 1):
            precheck_dia = prepare_dialog(test_data_Ses02[i][0], utt_to_ix)
            tmp = model_2(precheck_dia, test_data_Ses02[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses02[i][0]):
                pred_dict[utt] = tmp[j]
            
        for i in range(0, len(test_data_Ses03), 1):
            precheck_dia = prepare_dialog(test_data_Ses03[i][0], utt_to_ix)
            tmp = model_3(precheck_dia, test_data_Ses03[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses03[i][0]):
                pred_dict[utt] = tmp[j]
            
        for i in range(0, len(test_data_Ses04), 1):
            precheck_dia = prepare_dialog(test_data_Ses04[i][0], utt_to_ix)
            tmp = model_4(precheck_dia, test_data_Ses04[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses04[i][0]):
                pred_dict[utt] = tmp[j]
        
        for i in range(0, len(test_data_Ses05), 1):
            precheck_dia = prepare_dialog(test_data_Ses05[i][0], utt_to_ix)
            tmp = model_5(precheck_dia, test_data_Ses05[i][0])[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses05[i][0]):
                pred_dict[utt] = tmp[j]

    ori_emo_dict = joblib.load('../data/speech_only/emo_all_speech_only.pkl')
    label = []
    for dia_emos_tuple in test_data_Ses01:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses02:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses03:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses04:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses05:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])

    for i in range(0, len(label), 1):
        if label[i] == 'Anger':
            label[i] = 0
        elif label[i] == 'Happiness':
            label[i] = 1
        elif label[i] == 'Neutral':
            label[i] = 2
        elif label[i] == 'Sadness':
            label[i] = 3
        else:
            label[i] = -1
    
    uar, acc, f1, conf = utils.evaluate(predict, label, final_test=1)
    print('UAR:', uar)
    print('ACC:', acc)
    print('F1:', f1)
    print(conf)

    path = 'uar.txt'
    f = open(path, 'a')
    f.write(str(uar)+'\n')
    f.close()
    
    path = 'acc.txt'
    f = open(path, 'a')
    f.write(str(acc)+'\n')
    f.close()

    joblib.dump(pred_dict, './model/' + args.pretrain_version + '/' + args.dataset + '/preds_4.pkl')
    
    # ensure pretrained model performance
    labels = []
    predicts = []
    dialogs_edit = joblib.load('../data/speech_only/dialogs_edit_speech_only.pkl')
    emo_dict = joblib.load('../data/speech_only/emo_all_speech_only.pkl')
    emo2num = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3}
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            labels.append(emo2num[emo_dict[utt]])
            predicts.append(out_dict[utt].argmax())
            
    print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')
    print('pretrained F1:', round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    '''
    view_new_matrix(model_1, fold_num=1)
    view_new_matrix(model_2, fold_num=2)
    view_new_matrix(model_3, fold_num=3)
    view_new_matrix(model_4, fold_num=4)
    view_new_matrix(model_5, fold_num=5)
    '''