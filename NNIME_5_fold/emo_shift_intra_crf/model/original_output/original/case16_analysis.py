import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pdb
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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
    spk_dialogs[dialog_id+'_A'] = []
    spk_dialogs[dialog_id+'_B'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id.split('_')[-2] == 'A':
        spk_dialogs[dialog_id+'_A'].append(utt_id)
      elif utt_id.split('_')[-2] == 'B':
        spk_dialogs[dialog_id+'_B'].append(utt_id)

  return spk_dialogs

def get_16_case_performance(case_list_dict, model_outputs, fold_num):
    for key_trans in case_dict_label:
        if 'utt_list' in key_trans:
            if case_list_dict.get(key_trans.replace('utt', 'emo')) == None:
                case_list_dict[key_trans.replace('utt', 'emo')] = []
            for utt in case_dict_label[key_trans]:
                if utt[4] == fold_num:
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
    for utt in emo_dict:
        if emo_dict[utt] in ['Anger', 'Happiness', 'Neutral', 'Sadness']:
            try:
                pred.append(out_dict[utt].argmax())
            except:
                pred.append(out_dict[utt])
            gt.append(emo_num_dict[emo_dict[utt]])
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    print('ACC:', round(accuracy_score(gt, pred)*100, 2), '%')
    print('F1:', round(f1_score(gt, pred, average='weighted')*100, 2), '%')
    

if __name__ == '__main__':
    FOLD_NUM = '5'
    intra_tran_matrix = np.load('../../../../intra_crf/intra_trans_fold' + FOLD_NUM + '.npy')[:4,:4]
    ESA_no_shift_tran_matrix = np.load('../../../../emo_shift_intra_crf/ESA_no_shift_trans_fold' + FOLD_NUM + '.npy')[:4,:4]
    ESA_shift_tran_matrix = np.load('../../../../emo_shift_intra_crf/ESA_shift_trans_fold' + FOLD_NUM + '.npy')[:4,:4]
    intra_crf_outputs = joblib.load('./intra_crf_preds_4.pkl')

    #dag_output = joblib.load('../../../../data/speech_only/DAG_outputs_4_all_audio.pkl')
    rescoring_outputs = joblib.load('./preds_4.pkl')
    
    dialog_edit = joblib.load('../../../../data/speech_only/dialogs_edit_speech_only.pkl')
    spk_dialog_edit = split_dialog(dialog_edit)
    emo_dict = joblib.load('../../../../data/speech_only/emo_all_speech_only.pkl')
    emo_num_dict = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3}
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
    
    '''
    print('CHECK DAG PERFORMANCE:')
    check_model_performance(dag_output)
    print('##########')
    '''
    print('CHECK RESCORING ALGO. PERFORMANCE:')
    check_model_performance(rescoring_outputs)
    print('##########')
    
    # 切分case1~16
    case_dict_label = {}
    for dia_name in spk_dialog_edit:
        for i, utt in enumerate(spk_dialog_edit[dia_name]):
            if i != 0 and utt[4] == FOLD_NUM:
                #if i != 0:
                cur_emo = emo_dict[utt][0].lower()
                pre_emo = emo_dict[spk_dialog_edit[dia_name][i-1]][0].lower()
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
    intra_crf_case_performance_dict = get_16_case_performance(intra_crf_case_list_dict, intra_crf_outputs, fold_num=FOLD_NUM)
    
    '''
    dag_case_list_dict = {}
    dag_case_performance_dict = get_16_case_performance(dag_case_list_dict, dag_output, fold_num=FOLD_NUM)
    '''
    # rescoring lago. case1~16 performance
    rescoring_case_list_dict = {}
    rescoring_case_performance_dict = get_16_case_performance(rescoring_case_list_dict, rescoring_outputs, fold_num=FOLD_NUM)
    
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
    for i, result_dict in enumerate([result_performance_recall_dict, rescoring_case_performance_dict, intra_crf_case_performance_dict]):
        result_list = [[],[],[],[]]
        result_list[0].append(result_dict.get('a2a_RECALL',0))
        result_list[0].append(result_dict.get('h2a_RECALL',0))
        result_list[0].append(result_dict.get('n2a_RECALL',0))
        result_list[0].append(result_dict.get('s2a_RECALL',0))
        
        result_list[1].append(result_dict.get('a2h_RECALL',0))
        result_list[1].append(result_dict.get('h2h_RECALL',0))
        result_list[1].append(result_dict.get('n2h_RECALL',0))
        result_list[1].append(result_dict.get('s2h_RECALL',0))
        
        result_list[2].append(result_dict.get('a2n_RECALL',0))
        result_list[2].append(result_dict.get('h2n_RECALL',0))
        result_list[2].append(result_dict.get('n2n_RECALL',0))
        result_list[2].append(result_dict.get('s2n_RECALL',0))
        
        result_list[3].append(result_dict.get('a2s_RECALL',0))
        result_list[3].append(result_dict.get('h2s_RECALL',0))
        result_list[3].append(result_dict.get('n2s_RECALL',0))
        result_list[3].append(result_dict.get('s2s_RECALL',0))
        if i == 0:
            plt.figure()
            sns.set(font_scale=1.5)
            sns.heatmap(np.array(result_list), annot=True, fmt='.2f', xticklabels=['ang', 'hap', 'neu', 'sad'], yticklabels=['ang', 'hap', 'neu', 'sad'], cbar=True, square=True, cmap="OrRd", annot_kws={"size": 14})
            plt.savefig('case16_heatmap.png')
        elif i == 1:
            esa_crf_recall_arr = np.array(result_list)
        elif i == 2:
            intra_crf_recall_arr = np.array(result_list)
    
    intra_crf_tran_mtx_case1, intra_crf_tran_mtx_case2, esa_crf_tran_mtx_case1, esa_crf_tran_mtx_case2 = [],[],[],[]
    intra_crf_recall_case1, intra_crf_recall_case2, esa_crf_recall_case1, esa_crf_recall_case2 = [],[],[],[]
    for i in range(0,4,1):
        for j in range(0,4,1):
            if i == j: #case1
                intra_crf_tran_mtx_case1.append(intra_tran_matrix[i][j])
                esa_crf_tran_mtx_case1.append(ESA_no_shift_tran_matrix[i][j])
                intra_crf_recall_case1.append(intra_crf_recall_arr[i][j])
                esa_crf_recall_case1.append(esa_crf_recall_arr[i][j])
            else: #case2
                intra_crf_tran_mtx_case2.append(intra_tran_matrix[i][j])
                esa_crf_tran_mtx_case2.append(ESA_shift_tran_matrix[i][j])
                intra_crf_recall_case2.append(intra_crf_recall_arr[i][j])
                esa_crf_recall_case2.append(esa_crf_recall_arr[i][j])
    
    print('pearsonr ##########')
    corr, _ = pearsonr(np.array(intra_crf_tran_mtx_case1), np.array(intra_crf_recall_case1))
    print('CRF (intra) (case1): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = pearsonr(np.array(intra_crf_tran_mtx_case2), np.array(intra_crf_recall_case2))
    print('CRF (intra) (case2): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = pearsonr(np.array(esa_crf_tran_mtx_case1), np.array(esa_crf_recall_case1))
    print('ESA CRF (case1): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = pearsonr(np.array(esa_crf_tran_mtx_case2), np.array(esa_crf_recall_case2))
    print('ESA CRF (case2): corr between tran mtx & recall', round(corr,4))
    ##########
    print('spearmanr ##########')
    corr, _ = spearmanr(np.array(intra_crf_tran_mtx_case1), np.array(intra_crf_recall_case1))
    print('CRF (intra) (case1): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = spearmanr(np.array(intra_crf_tran_mtx_case2), np.array(intra_crf_recall_case2))
    print('CRF (intra) (case2): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = spearmanr(np.array(esa_crf_tran_mtx_case1), np.array(esa_crf_recall_case1))
    print('ESA CRF (case1): corr between tran mtx & recall', round(corr,4))
    
    corr, _ = spearmanr(np.array(esa_crf_tran_mtx_case2), np.array(esa_crf_recall_case2))
    print('ESA CRF (case2): corr between tran mtx & recall', round(corr,4))
        