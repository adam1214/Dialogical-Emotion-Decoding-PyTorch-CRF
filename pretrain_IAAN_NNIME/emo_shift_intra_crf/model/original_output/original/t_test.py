import scipy
from scipy import stats
import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def check_model_performance(out_dict):
    pred, gt = [], []
    for utt in ded_outputs:
        if emo_dict[utt] in ['Anger', 'Happiness', 'Neutral', 'Sadness']:
            try:
                pred.append(out_dict[utt].argmax())
            except:
                pred.append(out_dict[utt])
            gt.append(emo_num_dict[emo_dict[utt]])
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    print('ACC:', round(accuracy_score(gt, pred)*100, 2), '%')
    print('==========')
    '''
    a_pred, h_pred, n_pred, s_pred = [], [], [], []
    a_gt, h_gt, n_gt, s_gt = [], [], [], []
    for utt in ded_outputs:
        if emo_dict[utt] == 'Anger':
            try:
                a_pred.append(out_dict[utt].argmax())
            except:
                a_pred.append(out_dict[utt])
            a_gt.append(emo_num_dict[emo_dict[utt]])
        elif emo_dict[utt] == 'Happiness':
            try:
                h_pred.append(out_dict[utt].argmax())
            except:
                h_pred.append(out_dict[utt])
            h_gt.append(emo_num_dict[emo_dict[utt]])
        elif emo_dict[utt] == 'Neutral':
            try:
                n_pred.append(out_dict[utt].argmax())
            except:
                n_pred.append(out_dict[utt])
            n_gt.append(emo_num_dict[emo_dict[utt]])
        elif emo_dict[utt] == 'Sadness':
            try:
                s_pred.append(out_dict[utt].argmax())
            except:
                s_pred.append(out_dict[utt])
            s_gt.append(emo_num_dict[emo_dict[utt]])
    print('ANG ACC:', round(accuracy_score(a_gt, a_pred)*100, 2), '%')
    print('HAP ACC:', round(accuracy_score(h_gt, h_pred)*100, 2), '%')
    print('NEU ACC:', round(accuracy_score(n_gt, n_pred)*100, 2), '%')
    print('SAD ACC:', round(accuracy_score(s_gt, s_pred)*100, 2), '%')
    '''
def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1)
    std2 = np.std(group2)
    nobs1 = len(group1)
    nobs2 = len(group2)

    modified_std1 = np.sqrt(np.float32(nobs1)/
                    np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/
                    np.float32(nobs2-1)) * std2
    (statistic, pvalue) = stats.ttest_ind_from_stats( 
               mean1=mean1, std1=modified_std1, nobs1=nobs1,
               mean2=mean2, std2=modified_std2, nobs2=nobs2 )
    return statistic, pvalue

if __name__ == '__main__':
    ded_outputs = joblib.load('./DED_preds_4.pkl')
    esa_crf_outputs = joblib.load('./preds_4.pkl')
    iaan_outputs = {}
    iaan_output_fold1 = joblib.load('./iaan_utt_logits_outputs_fold1.pkl')
    iaan_output_fold2 = joblib.load('./iaan_utt_logits_outputs_fold2.pkl')
    iaan_output_fold3 = joblib.load('./iaan_utt_logits_outputs_fold3.pkl')
    iaan_output_fold4 = joblib.load('./iaan_utt_logits_outputs_fold4.pkl')
    iaan_output_fold5 = joblib.load('./iaan_utt_logits_outputs_fold5.pkl')
    for utt in iaan_output_fold1:
        if utt[4] == '1':
            iaan_outputs[utt] = iaan_output_fold1[utt]
        elif utt[4] == '2':
            iaan_outputs[utt] = iaan_output_fold2[utt]
        elif utt[4] == '3':
            iaan_outputs[utt] = iaan_output_fold3[utt]
        elif utt[4] == '4':
            iaan_outputs[utt] = iaan_output_fold4[utt]
        elif utt[4] == '5':
            iaan_outputs[utt] = iaan_output_fold5[utt]
    emo_dict = joblib.load('../../../../data/emo_all.pkl')
    emo_num_dict = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3}
    
    print('CHECK IAAN PERFORMANCE:')
    check_model_performance(iaan_outputs)
    print('##########')
    
    print('CHECK DED PERFORMANCE:')
    check_model_performance(ded_outputs)
    print('##########')
    
    print('CHECK ESA_CRF PERFORMANCE:')
    check_model_performance(esa_crf_outputs)
    print('##########')
    '''
    iaan_uar_list, iaan_acc_list, iaan_pred_list = [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]
    ded_uar_list, ded_acc_list, ded_pred_list = [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]
    esa_crf_uar_list, esa_crf_acc_list, esa_crf_pred_list = [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]
    gt_list = [[],[],[],[],[]]
    
    for utt in esa_crf_outputs:
        ses_index = int(utt[4])-1
        iaan_pred_list[ses_index].append(iaan_outputs[utt].argmax())
        ded_pred_list[ses_index].append(ded_outputs[utt])
        esa_crf_pred_list[ses_index].append(esa_crf_outputs[utt])
        gt_list[ses_index].append(emo_num_dict[emo_dict[utt]])
        
    for i in range(0, 5, 1):
        iaan_uar_list[i].append(recall_score(gt_list[i], iaan_pred_list[i], average='macro'))
        iaan_acc_list[i].append(accuracy_score(gt_list[i], iaan_pred_list[i]))
        
        ded_uar_list[i].append(recall_score(gt_list[i], ded_pred_list[i], average='macro'))
        ded_acc_list[i].append(accuracy_score(gt_list[i], ded_pred_list[i]))
        
        esa_crf_uar_list[i].append(recall_score(gt_list[i], esa_crf_pred_list[i], average='macro'))
        esa_crf_acc_list[i].append(accuracy_score(gt_list[i], esa_crf_pred_list[i]))
    
    print('UAR #####################')
    #t_statistic, p_value = scipy.stats.ttest_ind(np.array(esa_crf_uar_list)*100, np.array(ded_uar_list)*100, equal_var=False)
    t_statistic, p_value = t_test(np.array(esa_crf_uar_list)-np.array(iaan_uar_list), np.array(ded_uar_list)-np.array(iaan_uar_list))
    print('t_statistic', round(t_statistic, 2))
    print('p_value', round(p_value, 2))
    
    print('ACC #####################')
    #t_statistic, p_value = scipy.stats.ttest_ind(np.array(esa_crf_acc_list)*100, np.array(ded_acc_list)*100, equal_var=False)
    t_statistic, p_value = t_test(np.array(esa_crf_acc_list)-np.array(iaan_acc_list), np.array(ded_acc_list)-np.array(iaan_acc_list))
    print('t_statistic', round(t_statistic, 2))
    print('p_value', round(p_value, 2))
    '''
    
    table = np.zeros((2,2), dtype=np.int32)
    for utt in ded_outputs:
        if emo_dict[utt] in ['Anger', 'Happiness', 'Neutral', 'Sadness']:
            if ded_outputs[utt] == emo_num_dict[emo_dict[utt]] and esa_crf_outputs[utt] == emo_num_dict[emo_dict[utt]]:
                table[0][0] += 1
            elif ded_outputs[utt] == emo_num_dict[emo_dict[utt]] and esa_crf_outputs[utt] != emo_num_dict[emo_dict[utt]]:
                table[0][1] += 1
            elif ded_outputs[utt] != emo_num_dict[emo_dict[utt]] and esa_crf_outputs[utt] == emo_num_dict[emo_dict[utt]]:
                table[1][0] += 1
            elif ded_outputs[utt] != emo_num_dict[emo_dict[utt]] and esa_crf_outputs[utt] != emo_num_dict[emo_dict[utt]]:
                table[1][1] += 1
    print(table)
    result = mcnemar(table, exact=False, correction=True)
    # result = mcnemar(table, exact=True) # 有小於25的element用這條
    print('statistic=%.4f, p-value=%.8f' % (result.statistic, result.pvalue))
    