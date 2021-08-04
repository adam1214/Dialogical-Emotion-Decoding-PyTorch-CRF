import joblib
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
if __name__ == '__main__':
    dialogs = joblib.load('./dialog_meld.pkl')
    emo_dict = joblib.load('./emo_all_meld.pkl')
    out_dict = joblib.load('./outputs.pkl')
    
    emo_num_dict = {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6}
    predicts = []
    labels = []
    for dia_name in dialogs:
        if dia_name.split('_')[0] == 'test':
            for utt_name in dialogs[dia_name]:
                labels.append(emo_num_dict[emo_dict[utt_name]])
                predicts.append(out_dict[utt_name].index(max(out_dict[utt_name])))
    
    avg_fscore = round(f1_score(labels, predicts, average='weighted') * 100, 2)
    uar = recall_score(labels, predicts, average='macro')
    acc = accuracy_score(labels, predicts)
    print('UAR', uar)
    print('ACC', acc)
    print('f1', avg_fscore)
    print(confusion_matrix(labels, predicts))