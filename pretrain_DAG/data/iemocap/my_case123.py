import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pdb
import numpy as np 

def gen_case1_2_3_dict(case1_dict, case2_dict, case3_dict, spkID, emo_dict, dialog):
    #emo = ['ang', 'hap', 'neu', 'sad']
    for dialog_name in dialog:
        if dialog_name[4] != '5':
            continue
        U_p_emo = ""
        U_p = ""
        for t in range(0, len(dialog[dialog_name]), 1):
            if dialog[dialog_name][t][-4] == spkID:
                if U_p_emo != "" and dialog[dialog_name][t][-4] != dialog[dialog_name][t-1][-4]:
                    U_c_emo = emo_dict[dialog[dialog_name][t]]
                    U_r_emo = emo_dict[dialog[dialog_name][t-1]]
                    U_c = dialog[dialog_name][t]
                    U_r = dialog[dialog_name][t-1]
                    if U_c_emo == U_p_emo and U_c_emo == U_r_emo:
                        #case1_dict[len(case1_dict)] = {'U_c': U_c, 'U_p', U_p, 'U_r', U_r, 'U_c_emo': U_c_emo, 'U_p_emo': U_p_emo, 'U_r_emo': U_r_emo}
                        case1_dict[len(case1_dict)] = {}
                        case1_dict[len(case1_dict)-1]['U_c'] = U_c
                        case1_dict[len(case1_dict)-1]['U_p'] = U_p
                        case1_dict[len(case1_dict)-1]['U_r'] = U_r
                        case1_dict[len(case1_dict)-1]['U_c_emo'] = U_c_emo
                        case1_dict[len(case1_dict)-1]['U_p_emo'] = U_p_emo
                        case1_dict[len(case1_dict)-1]['U_r_emo'] = U_r_emo
                    elif U_c_emo == U_p_emo or U_c_emo == U_r_emo:
                        case2_dict[len(case2_dict)] = {}
                        case2_dict[len(case2_dict)-1]['U_c'] = U_c
                        case2_dict[len(case2_dict)-1]['U_p'] = U_p
                        case2_dict[len(case2_dict)-1]['U_r'] = U_r
                        case2_dict[len(case2_dict)-1]['U_c_emo'] = U_c_emo
                        case2_dict[len(case2_dict)-1]['U_p_emo'] = U_p_emo
                        case2_dict[len(case2_dict)-1]['U_r_emo'] = U_r_emo
                    elif U_c_emo != U_p_emo and U_c_emo != U_r_emo:
                        case3_dict[len(case3_dict)] = {}
                        case3_dict[len(case3_dict)-1]['U_c'] = U_c
                        case3_dict[len(case3_dict)-1]['U_p'] = U_p
                        case3_dict[len(case3_dict)-1]['U_r'] = U_r
                        case3_dict[len(case3_dict)-1]['U_c_emo'] = U_c_emo
                        case3_dict[len(case3_dict)-1]['U_p_emo'] = U_p_emo
                        case3_dict[len(case3_dict)-1]['U_r_emo'] = U_r_emo
                U_p_emo = emo_dict[dialog[dialog_name][t]]
                U_p = dialog[dialog_name][t]

def case_acc(case_dict, outputs, emo2num):
    labels = []
    predicts = []
    for utts_dict in case_dict.values():
        labels.append(emo2num[utts_dict['U_c_emo']])
        predicts.append(outputs[utts_dict['U_c']].index(max(outputs[utts_dict['U_c']])))
    return labels, predicts

def analyze_case1_2_3(emo_dict, outputs, dialog, emo2num):
    '''
    U_c: 當前語者的當前utt
    U_p: 當前語者的前一次utt
    U_r: 另一語者的前一次utt
    '''
    case1_dict = {}
    case2_dict = {}
    case3_dict = {}
    gen_case1_2_3_dict(case1_dict, case2_dict, case3_dict, 'M', emo_dict, dialog)
    gen_case1_2_3_dict(case1_dict, case2_dict, case3_dict, 'F', emo_dict, dialog)
    total_case = len(case1_dict) + len(case2_dict) + len(case3_dict)
    print('Data points for case_1:', round(len(case1_dict)*100/total_case, 2), '%')
    print('Data points for case_2:', round(len(case2_dict)*100/total_case, 2), '%')
    print('Data points for case_3:', round(len(case3_dict)*100/total_case, 2), '%')
    
    labels, predicts = case_acc(case1_dict, outputs, emo2num)
    print("case 1:", round(accuracy_score(labels, predicts)*100, 2), '%')
    labels, predicts = case_acc(case2_dict, outputs, emo2num)
    print("case 2:", round(accuracy_score(labels, predicts)*100, 2), '%')
    labels, predicts = case_acc(case3_dict, outputs, emo2num)
    print("case 3:", round(accuracy_score(labels, predicts)*100, 2), '%')
    
if __name__ == '__main__':
    #dialog = joblib.load('../../../../data/iemocap/dialog_iemocap.pkl')
    dialog_edit = joblib.load('./dialog_6emo_iemocap.pkl')
    emo_dict = joblib.load('./emo_all_iemocap_6.pkl')
    outputs = joblib.load('./outputs_6.pkl')
    emo2num = {'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5}
    
    emo_set = set()
    for k in emo_dict:
        emo_set.add(emo_dict[k])
    print(emo_set)
    
    # ensure total acc
    labels = []
    predicts = []
    for dialog_name in dialog_edit:
        for utt in dialog_edit[dialog_name]:
            if utt[4] == '5':
                labels.append(emo2num[emo_dict[utt]])
                predicts.append(outputs[utt].index(max(outputs[utt])))
            
    print('Total weighted f1:', round(f1_score(labels, predicts, average='weighted')*100, 2), '%')
    
    analyze_case1_2_3(emo_dict, outputs, dialog_edit, emo2num)
