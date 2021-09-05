import joblib
from sklearn.metrics import f1_score, accuracy_score, recall_score
import copy
if __name__ == '__main__':
    emo_dict = joblib.load('./emo_all_iemocap_4.pkl')
    out_dict = joblib.load('./outputs_4.pkl')
    '''
    emo_num_dict = {'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5}
    predicts = []
    labels = []
    for utt_name in emo_dict:
        if emo_dict[utt_name] == '---' or utt_name[4] != '5':
            continue
        else:
            labels.append(emo_num_dict[emo_dict[utt_name]])
            predicts.append(out_dict[utt_name].index(max(out_dict[utt_name])))
    avg_fscore = round(f1_score(labels, predicts, average='weighted') * 100, 2)
    uar = recall_score(labels, predicts, average='macro')
    acc = accuracy_score(labels, predicts)
    '''
    '''
    emo_label_list = ['exc', 'neu', 'fru', 'sad', 'hap', 'ang']
    emo_dict_U2U = emo_dict.copy()
    for utt in emo_dict_U2U:
        if emo_dict_U2U[utt] == '---':
            max_logits_val = max(out_dict[utt])
            max_index = out_dict[utt].index(max_logits_val)
            emo_dict_U2U[utt] = emo_label_list[max_index]
    joblib.dump(emo_dict_U2U, './U2U_6emo_all_iemocap.pkl')
    '''
    utt_cnt = 0
    dialog_6 = joblib.load('./dialog_6emo_iemocap.pkl')
    dialog = joblib.load('./dialog_iemocap.pkl')
    dialog_copy = copy.deepcopy(dialog)
    for dia_name in dialog:
        for utt in dialog[dia_name]:
            if emo_dict[utt] == '---':
                dialog_copy[dia_name].remove(utt)
    for dia_name in dialog_copy:
        utt_cnt += len(dialog_copy[dia_name])
    print(utt_cnt)
    joblib.dump(dialog_copy, './dialog_4emo_iemocap.pkl')