import joblib
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
if __name__ == '__main__':
    emo_dict = joblib.load('./emo_all_iemocap.pkl')
    out_dict = joblib.load('./outputs.pkl')
    utts_concat_representation = joblib.load('./utts_concat_representation.pkl')
    '''
    emo_num_dict = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
    predicts = []
    labels = []
    for utt_name in emo_dict:
        if emo_dict[utt_name] != 'ang' and emo_dict[utt_name] != 'hap' and emo_dict[utt_name] != 'neu' and emo_dict[utt_name] != 'sad':
            continue
        else:
            labels.append(emo_num_dict[emo_dict[utt_name]])
            predicts.append(out_dict[utt_name].tolist().index(max(out_dict[utt_name])))
    #avg_fscore = round(f1_score(labels, predicts, average='weighted') * 100, 2)
    uar = recall_score(labels, predicts, average='macro')
    acc = accuracy_score(labels, predicts)
    print('uar:', uar)
    print('acc:', acc)
    print(confusion_matrix(labels, predicts))
    '''
    '''
    emo_label_list = ['ang', 'hap', 'neu', 'sad']
    emo_dict_U2U = emo_dict.copy()
    for utt in emo_dict_U2U:
        if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad':
            max_logits_val = max(out_dict[utt].tolist())
            max_index = out_dict[utt].tolist().index(max_logits_val)
            emo_dict_U2U[utt] = emo_label_list[max_index]
    joblib.dump(emo_dict_U2U, './U2U_4emo_all_iemocap.pkl')
    '''
