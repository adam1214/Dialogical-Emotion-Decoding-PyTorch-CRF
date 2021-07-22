import joblib
from sklearn.metrics import f1_score, accuracy_score, recall_score
if __name__ == '__main__':
    emo_dict = joblib.load('./emo_all_iemocap.pkl')
    out_dict = joblib.load('./outputs.pkl')
    
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
    emo_label_list = ['exc', 'neu', 'fru', 'sad', 'hap', 'ang']
    emo_dict_U2U = emo_dict.copy()
    for utt in emo_dict_U2U:
        if emo_dict_U2U[utt] == '---':
            max_logits_val = max(out_dict[utt])
            max_index = out_dict[utt].index(max_logits_val)
            emo_dict_U2U[utt] = emo_label_list[max_index]
    joblib.dump(emo_dict_U2U, './U2U_6emo_all_iemocap.pkl')
    '''