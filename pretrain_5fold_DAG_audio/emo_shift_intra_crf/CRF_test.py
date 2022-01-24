import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from CRF_train import CRF

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
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', "--pretrain_version", type=str, help="which version of pretrain model you want to use?", default='dialog_rearrange_output')
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    parser.add_argument("-e", "--emo_shift", type=str, help="which emo_shift prob. to use?", default = 'model')
    args = parser.parse_args()
    
    print(args)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    out_dict = joblib.load('../data/'+ args.pretrain_version + '/dag_outputs_4_all_fold_single_rearrange.pkl')
    #dialogs = joblib.load('../data/dialog_iemocap.pkl')
    #dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    dialogs = joblib.load('../data/dialog_rearrange.pkl')
    dialogs_edit = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
        dias = dialogs_edit
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/'+ args.pretrain_version + '/U2U_4emo_all_iemocap.pkl')
        dias = dialogs
        
    if args.emo_shift == 'constant':
        spk_dialogs = utils.split_dialog(dias)
        bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
    else:
        bias_dict = joblib.load('../data/'+ args.pretrain_version + '/RandomForest_emo_shift_output.pkl')
    
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
                if utt[-4] == 'M':
                    test_data_Ses01[-2][0].append(utt)
                    test_data_Ses01[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
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
                if utt[-4] == 'M':
                    test_data_Ses02[-2][0].append(utt)
                    test_data_Ses02[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
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
                if utt[-4] == 'M':
                    test_data_Ses03[-2][0].append(utt)
                    test_data_Ses03[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
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
                if utt[-4] == 'M':
                    test_data_Ses04[-2][0].append(utt)
                    test_data_Ses04[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
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
                if utt[-4] == 'M':
                    test_data_Ses05[-2][0].append(utt)
                    test_data_Ses05[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
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

    emo_to_ix = {"ang": 0, "hap": 1, "neu": 2, "sad": 3, START_TAG: 4, STOP_TAG: 5}

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

    ori_emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
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
        if label[i] == 'ang':
            label[i] = 0
        elif label[i] == 'hap':
            label[i] = 1
        elif label[i] == 'neu':
            label[i] = 2
        elif label[i] == 'sad':
            label[i] = 3
        else:
            label[i] = -1
    
    uar, acc, conf = utils.evaluate(predict, label, final_test=1)
    print('UAR:', uar)
    print('ACC:', acc)
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
    dialogs_edit = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
    emo2num = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            labels.append(emo2num[emo_dict[utt]])
            predicts.append(out_dict[utt].argmax())
          
    print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')