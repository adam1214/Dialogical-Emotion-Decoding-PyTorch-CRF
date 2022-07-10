import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pdb
import numpy as np 
import pandas as pd

def generate_interaction_sample(index_words, seq_dict, emo_dict, case_num):
    global negative_sec_cnt, closest_self_opp_cnt, total_closest, case_1, case_2, case_3
    """ 
    Generate interaction training pairs,
    total 4 class, total 5531 emo samples."""
    emo = ['ang', 'hap', 'neu', 'sad']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    self_emo_shift = []
    self_time_dur = []
    closest_time_dur = []
    for index, center in enumerate(index_words):
        if emo_dict[center] in emo:
            time_self_self = 10000.
            time_self_opp = 10000.
            #if True:
            center_.append(center)
            center_label.append(emo_dict[center])
            pt = []
            pp = []
            for word in index_words[max(0, index - 8): index]:
                if word[-4] == center[-4]:
                    pt.append(word)
                else:
                    pp.append(word)
            
            if len(pt) != 0:
                target_.append(pt[-1])
                target_label.append(emo_dict[pt[-1]])
                target_dist.append(index - index_words.index(pt[-1]))
                if emo_dict[pt[-1]] == emo_dict[center]:
                    self_emo_shift.append(0)
                else:
                    self_emo_shift.append(1)
                time_self_self = 0 - 0
                self_time_dur.append(time_self_self)
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)
                self_time_dur.append('None')

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
                time_self_opp = 0 - 0
                if time_self_opp < 0:
                    negative_sec_cnt += 1
                    print('negative sample', negative_sec_cnt, center, pp[-1])
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')
            
            if min(time_self_opp, time_self_self) != 10000.:
                total_closest += 1
                if min(time_self_opp, time_self_self) == time_self_opp:
                    closest_self_opp_cnt += 1
                closest_time_dur.append(min(time_self_opp, time_self_self))
            else:
                closest_time_dur.append('None')
                
            if case_num == 1 and center_label[-1] != target_label[-1] and target_label[-1] != 'pad':
                del center_[-1]
                del target_[-1]
                del opposite_[-1]
                del center_label[-1]
                del target_label[-1]
                del opposite_label[-1]
                del target_dist[-1]
                del opposite_dist[-1]
                del self_emo_shift[-1]
                del self_time_dur[-1]
                del closest_time_dur[-1]
            elif case_num == 2 and (center_label[-1] == target_label[-1] or target_label[-1] == 'pad'):
                del center_[-1]
                del target_[-1]
                del opposite_[-1]
                del center_label[-1]
                del target_label[-1]
                del opposite_label[-1]
                del target_dist[-1]
                del opposite_dist[-1]
                del self_emo_shift[-1]
                del self_time_dur[-1]
                del closest_time_dur[-1]

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift, self_time_dur, closest_time_dur

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, case_num, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train, self_time_dur_train, closest_time_dur_train = [], [], [], [], [], [], [], [], [], [], []
    
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses, std, ctd = generator(dialog_order, seq_dict, emo_dict, case_num=case_num)
        center_train += c
        target_train += t
        opposite_train += o
        center_label_train += cl
        target_label_train += tl
        opposite_label_train += ol
        target_dist_train += td
        opposite_dist_train += od
        self_emo_shift_train += ses
        self_time_dur_train += std
        closest_time_dur_train += ctd

    # save dialog pairs to train.csv and test.csv
    #data_filename = './all_data.csv'
    data_filename = './case' + str(case_num) + '_data.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift', 'self_time_dur', 'closest_time_dur']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train, 'self_time_dur': self_time_dur_train, 'closest_time_dur': closest_time_dur_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(data_filename, sep=',', index = False)

def case_acc(case_dict, outputs, emo2num):
    labels = []
    predicts = []
    for utts_dict in case_dict.values():
        labels.append(emo2num[utts_dict['U_c_emo']])
        predicts.append(outputs[utts_dict['U_c']])
    return labels, predicts

def analyze_case1_2(emo_dict, outputs, dialog, emo2num, case1_dict, case2_dict):
    '''
    U_c: 當前語者的當前utt
    U_p: 當前語者的前一次utt
    U_r: 另一語者的前一次utt
    '''
    
    total_case = len(case1_dict) + len(case2_dict)
    print('Data points for case_1:', round(len(case1_dict)*100/total_case, 2), '%')
    print('Data points for case_2:', round(len(case2_dict)*100/total_case, 2), '%')
    
    labels, predicts = case_acc(case1_dict, outputs, emo2num)
    print("case 1 UAR:", round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print("case 1 ACC:", round(accuracy_score(labels, predicts)*100, 2), '%')
    print("case 1 F1:", round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    labels, predicts = case_acc(case2_dict, outputs, emo2num)
    print("case 2 UAR:", round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print("case 2 ACC:", round(accuracy_score(labels, predicts)*100, 2), '%')
    print("case 2 F1:", round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    
if __name__ == '__main__':
    total_closest = 0
    closest_self_opp_cnt = 0
    original_utt_text = joblib.load('../../../../data/original_utt_text.pkl')
    dialog = joblib.load('../../../../data/dialog_rearrange.pkl')
    dialog_edit = joblib.load('../../../../data/dialog_rearrange_4emo_iemocap.pkl')
    emo_dict = joblib.load('../../../../data/emo_all_iemocap.pkl')
    emo_shift_all = joblib.load('../../../../data/dialog_rearrange_output/4emo_shift_all_rearrange.pkl')
    outputs = joblib.load('./preds_4.pkl')
    pretrained_outputs = joblib.load('../../../../data/dialog_rearrange_output/DAG_outputs_4_all_fold_text_audio.pkl')
    bias_dict = joblib.load('../../../../data/dialog_rearrange_output/SVM_emo_shift_output_text_audio.pkl')
    emo2num = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    num2emo = {0:'ang', 1:'hap', 2:'neu', 3:'sad'}
    
    emo_set = set()
    for k in emo_dict:
        emo_set.add(emo_dict[k])
    print(emo_set)
    
    # ensure total acc
    labels = []
    predicts = []
    for dialog_name in dialog_edit:
        for utt in dialog_edit[dialog_name]:
            labels.append(emo2num[emo_dict[utt]])
            predicts.append(outputs[utt])
            
    print('Total UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('Total ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')
    print('Total F1:', round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    
    case1_dict = {}
    case2_dict = {}
    # generate data
    for case_num in range(1, 3, 1):
        generate_interaction_data(dialog, {}, emo_dict, case_num)
        
        data_filename = './case' + str(case_num) + '_data.csv'
        all_data = pd.read_csv(data_filename)
        # build case12_dict
        for index, row in all_data.iterrows():
            center_utt = row[0]
            target_utt = row[1]
            opp_utt = row[2]
            center_label = row[3]
            target_label = row[4]
            opp_label = row[5]
            if case_num == 1:
                case1_dict[len(case1_dict)] = {}
                case1_dict[len(case1_dict)-1]['U_c'] = center_utt
                case1_dict[len(case1_dict)-1]['U_p'] = target_utt
                case1_dict[len(case1_dict)-1]['U_r'] = opp_utt
                case1_dict[len(case1_dict)-1]['U_c_emo'] = center_label
                case1_dict[len(case1_dict)-1]['U_p_emo'] = target_label
                case1_dict[len(case1_dict)-1]['U_r_emo'] = opp_label
            elif case_num == 2:
                case2_dict[len(case2_dict)] = {}
                case2_dict[len(case2_dict)-1]['U_c'] = center_utt
                case2_dict[len(case2_dict)-1]['U_p'] = target_utt
                case2_dict[len(case2_dict)-1]['U_r'] = opp_utt
                case2_dict[len(case2_dict)-1]['U_c_emo'] = center_label
                case2_dict[len(case2_dict)-1]['U_p_emo'] = target_label
                case2_dict[len(case2_dict)-1]['U_r_emo'] = opp_label
        
    analyze_case1_2(emo_dict, outputs, dialog, emo2num, case1_dict, case2_dict)
    '''
    utt_name_first_9_with_many_emo_shift_case_dict = {}
    for dia in dialog_edit:
        for i, utt in enumerate(dialog_edit[dia]):
            emo_shift_cnt = 0
            for j in range(i, i+9, 1):
                if j == len(dialog_edit[dia]):
                    break
                elif emo_shift_all[dialog_edit[dia][j]] == 1.0:
                    emo_shift_cnt += 1
            if emo_shift_cnt >= 5:
                utt_name_first_9_with_many_emo_shift_case_dict[utt] = emo_shift_cnt
                
    utt_name_first_9_with_many_correct_emo_shift_case_dict = {}
    for utt in utt_name_first_9_with_many_emo_shift_case_dict:
        dia_name = utt[:-5]
        utt_index = dialog_edit[dia_name].index(utt)
        #print('###############')
        emo_shift_correct_cnt = 0
        for j in range(utt_index, utt_index+9, 1):
            if j == len(dialog_edit[dia_name]):
                break
            #print(dialog_edit[dia_name][j], dialog_edit[dia_name][j][-4], emo_dict[dialog_edit[dia_name][j]], num2emo[pretrained_outputs[dialog_edit[dia_name][j]].argmax()], num2emo[outputs[dialog_edit[dia_name][j]]])
            if emo_dict[dialog_edit[dia_name][j]] == num2emo[outputs[dialog_edit[dia_name][j]]] and emo_dict[dialog_edit[dia_name][j]] != num2emo[pretrained_outputs[dialog_edit[dia_name][j]].argmax()] and emo_shift_all[dialog_edit[dia_name][j]] == 1.0 and bias_dict[dialog_edit[dia_name][j]] > 0.5:
                emo_shift_correct_cnt += 1
        if emo_shift_correct_cnt >= 1:
            utt_name_first_9_with_many_correct_emo_shift_case_dict[dialog_edit[dia_name][utt_index]] = emo_shift_correct_cnt
            for j in range(utt_index, utt_index+9, 1):
                if j == len(dialog_edit[dia_name]):
                    break
                correct_shift_flag = (emo_dict[dialog_edit[dia_name][j]] == num2emo[outputs[dialog_edit[dia_name][j]]] and emo_dict[dialog_edit[dia_name][j]] != num2emo[pretrained_outputs[dialog_edit[dia_name][j]].argmax()] and emo_shift_all[dialog_edit[dia_name][j]] == 1.0 and bias_dict[dialog_edit[dia_name][j]] > 0.5)
                print(dialog_edit[dia_name][j], dialog_edit[dia_name][j][-4], emo_dict[dialog_edit[dia_name][j]], num2emo[pretrained_outputs[dialog_edit[dia_name][j]].argmax()], num2emo[outputs[dialog_edit[dia_name][j]]], correct_shift_flag, emo_shift_all[dialog_edit[dia_name][j]], round(bias_dict[dialog_edit[dia_name][j]],2), original_utt_text[dialog_edit[dia_name][j]][:-1])
            print('###############')
    '''