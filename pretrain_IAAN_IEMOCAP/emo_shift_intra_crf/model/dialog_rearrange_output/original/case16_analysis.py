import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pdb
import numpy as np 
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def split_dialog(dialogs):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialogs: dict, for example, utterances of two speakers in dialog_01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialogs = {}
  for dialog_id in dialogs.keys():
    spk_dialogs[dialog_id+'_M'] = []
    spk_dialogs[dialog_id+'_F'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id[-4] == 'M':
        spk_dialogs[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        spk_dialogs[dialog_id+'_F'].append(utt_id)

  return spk_dialogs

def get_16_case_performance(case_list_dict, model_outputs):
    for key_trans in case_dict_label:
        if 'utt_list' in key_trans:
            if case_list_dict.get(key_trans.replace('utt', 'emo')) == None:
                case_list_dict[key_trans.replace('utt', 'emo')] = []
            for utt in case_dict_label[key_trans]:
                try:
                    case_list_dict[key_trans.replace('utt', 'emo')].append(model_outputs[utt].argmax())
                except:
                    case_list_dict[key_trans.replace('utt', 'emo')].append(model_outputs[utt])
    
    case_performance_dict = {}
    for key_trans in case_list_dict:
        #case_performance_dict[key_trans[0:3]+'_UAR'] = recall_score(case_dict_label[key_trans], case_list_dict[key_trans], average='macro')
        case_performance_dict[key_trans[0:3]+'_RECALL'] = accuracy_score(case_dict_label[key_trans], case_list_dict[key_trans])
        
    return case_performance_dict

def check_model_performance(out_dict):
    pred, gt = [], []
    for utt in intra_crf_outputs:
        if emo_dict[utt] in ['ang', 'hap', 'neu', 'sad']:
            try:
                pred.append(out_dict[utt].argmax())
            except:
                pred.append(out_dict[utt])
            gt.append(emo_num_dict[emo_dict[utt]])
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    print('ACC:', round(accuracy_score(gt, pred)*100, 2), '%')

if __name__ == '__main__':
    intra_crf_outputs = joblib.load('./intra_crf_preds_4.pkl')
    iaan_output_fold1 = joblib.load('../../../../data/dialog_rearrange_output/utt_logits_outputs_fold1.pkl')
    iaan_output_fold2 = joblib.load('../../../../data/dialog_rearrange_output/utt_logits_outputs_fold2.pkl')
    iaan_output_fold3 = joblib.load('../../../../data/dialog_rearrange_output/utt_logits_outputs_fold3.pkl')
    iaan_output_fold4 = joblib.load('../../../../data/dialog_rearrange_output/utt_logits_outputs_fold4.pkl')
    iaan_output_fold5 = joblib.load('../../../../data/dialog_rearrange_output/utt_logits_outputs_fold5.pkl')
    rescoring_outputs = joblib.load('./preds_4.pkl')
    
    dialog_edit = joblib.load('../../../../data/dialog_rearrange_4emo_iemocap.pkl')
    spk_dialog_edit = split_dialog(dialog_edit)
    emo_dict = joblib.load('../../../../data/emo_all_iemocap.pkl')
    emo_num_dict = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
    '''
    iaan_out_dict = {}
    for utt in iaan_output_fold1:
        if utt[4] == '1':
            iaan_out_dict[utt] = iaan_output_fold1[utt]
        elif utt[4] == '2':
            iaan_out_dict[utt] = iaan_output_fold2[utt]
        elif utt[4] == '3':
            iaan_out_dict[utt] = iaan_output_fold3[utt]
        elif utt[4] == '4':
            iaan_out_dict[utt] = iaan_output_fold4[utt]
        elif utt[4] == '5':
            iaan_out_dict[utt] = iaan_output_fold5[utt]
    '''
    print('CHECK intra_crf PERFORMANCE:')
    check_model_performance(intra_crf_outputs)
    print('##########')
    
    print('CHECK RESCORING ALGO. PERFORMANCE:')
    check_model_performance(rescoring_outputs)
    print('##########')
    
    # 切分case1~16
    case_dict_label = {}
    for dia_name in spk_dialog_edit:
        for i, utt in enumerate(spk_dialog_edit[dia_name]):
            if i != 0:
                cur_emo = emo_dict[utt][0]
                pre_emo = emo_dict[spk_dialog_edit[dia_name][i-1]][0]
                key_trans = pre_emo + '2' + cur_emo + '_utt_list'
                if case_dict_label.get(key_trans) == None:
                    case_dict_label[key_trans] = []
                case_dict_label[key_trans].append(utt)
                
                key_trans = pre_emo + '2' + cur_emo + '_emo_list'
                if case_dict_label.get(key_trans) == None:
                    case_dict_label[key_trans] = []
                case_dict_label[key_trans].append(emo_num_dict[emo_dict[utt]])
    
    # check total utt in 16 case
    case_total_utt_cnt = 0
    for key_trans in case_dict_label:
        case_total_utt_cnt += len(case_dict_label[key_trans])
    print('check total utt in 16 case:', case_total_utt_cnt/2)
    '''
    # iaan case1~16 performance
    iaan_case_list_dict = {}
    iaan_case_performance_dict = get_16_case_performance(iaan_case_list_dict, iaan_out_dict)
    '''
    intra_crf_case_list_dict = {}
    intra_crf_case_performance_dict = get_16_case_performance(intra_crf_case_list_dict, intra_crf_outputs)
    # rescoring lago. case1~16 performance
    rescoring_case_list_dict = {}
    rescoring_case_performance_dict = get_16_case_performance(rescoring_case_list_dict, rescoring_outputs)
    
    # imporve percent
    result_performance_cnt_dict = {}
    result_performance_recall_dict = {}
    for key_trans in rescoring_case_performance_dict:
        result_performance_cnt_dict[key_trans] = (rescoring_case_performance_dict[key_trans] - intra_crf_case_performance_dict[key_trans])*len(case_dict_label[key_trans[0:3]+'_emo_list'])
        result_performance_recall_dict[key_trans] = 100*(rescoring_case_performance_dict[key_trans] - intra_crf_case_performance_dict[key_trans])
    #key_sorting_list = sorted(result_performance_dict, key=result_performance_dict.get, reverse=True) # [Large, ... , Small]
    '''
    for i in range(0, 5, 1):
        print(key_sorting_list[i][0:3], result_performance_dict[key_sorting_list[i]], '%')
    '''
    '''
    display_count = 0
    for i in range(0, len(key_sorting_list), 1):
        if key_sorting_list[i][0] != key_sorting_list[i][2]:
            #print(key_sorting_list[i][0:3], result_performance_dict[key_sorting_list[i]], '%', '/', len(rescoring_case_list_dict[key_sorting_list[i][0:3]+'_emo_list']))
            print(key_sorting_list[i][0:3], result_performance_dict[key_sorting_list[i]])
            display_count += 1
        if display_count == 5:
            break
    '''
    for result_dict in [result_performance_recall_dict, result_performance_cnt_dict]:
        result_list = [[],[],[],[]]
        result_list[0].append(result_dict['a2a_RECALL'])
        result_list[0].append(result_dict['h2a_RECALL'])
        result_list[0].append(result_dict['n2a_RECALL'])
        result_list[0].append(result_dict['s2a_RECALL'])
        
        result_list[1].append(result_dict['a2h_RECALL'])
        result_list[1].append(result_dict['h2h_RECALL'])
        result_list[1].append(result_dict['n2h_RECALL'])
        result_list[1].append(result_dict['s2h_RECALL'])
        
        result_list[2].append(result_dict['a2n_RECALL'])
        result_list[2].append(result_dict['h2n_RECALL'])
        result_list[2].append(result_dict['n2n_RECALL'])
        result_list[2].append(result_dict['s2n_RECALL'])
        
        result_list[3].append(result_dict['a2s_RECALL'])
        result_list[3].append(result_dict['h2s_RECALL'])
        result_list[3].append(result_dict['n2s_RECALL'])
        result_list[3].append(result_dict['s2s_RECALL'])
        plt.figure()
        sns.set(font_scale=2)
        sns.heatmap(np.array(result_list), annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=['ang', 'hap', 'neu', 'sad'], cbar=True, square=True, cmap="OrRd", annot_kws={"size": 14})
        break