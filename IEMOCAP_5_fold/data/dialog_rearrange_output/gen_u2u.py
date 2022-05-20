import joblib
import numpy as np
import copy

out_dict = joblib.load('outputs.pkl')
emo_dict = joblib.load('emo_all_iemocap.pkl')
emo_list = ['ang', 'hap', 'neu', 'sad']
emo_dict_edit = copy.deepcopy(emo_dict)

for utt in emo_dict:
    if emo_dict[utt] not in emo_list:
        emo_dict_edit[utt] = emo_list[np.argmax(out_dict[utt])]
joblib.dump(emo_dict_edit, 'U2U_4emo_all_iemocap.pkl') #utterance to utterance mapping according to pretrained classifier