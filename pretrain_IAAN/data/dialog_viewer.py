import joblib
dialog = joblib.load('./dialog_iemocap.pkl')
emo_dict = joblib.load('./emo_all_iemocap.pkl')

emo_tran = 0
emo_spk_tran = 0
total = 0

for dialog_name in dialog:
    pre_emo = ''
    pre_spk = ''
    for utt in dialog[dialog_name]:
        if pre_emo != '' and emo_dict[utt] != pre_emo:
            emo_tran += 1
        if pre_emo != '' and emo_dict[utt] != pre_emo and pre_spk != utt[-4]:
            emo_spk_tran += 1
        
        if pre_emo != '':
            total += 1
        
        pre_emo = emo_dict[utt]
        pre_spk = utt[-4]