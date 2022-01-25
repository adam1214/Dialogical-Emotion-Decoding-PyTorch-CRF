import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

def generate_interaction_sample(index_words, seq_dict, emo_dict, four_type_utt_list, emo_shift_dict):
    """ 
    Generate interaction training pairs,
    total 4 class, total 5531 emo samples."""
    emo = ['anger', 'joy', 'neutral', 'sadness']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    self_emo_shift = []
    for index, center in enumerate(index_words):
        #if emo_dict[center] in emo:
        if True:
            if four_type_utt_list != None and emo_dict[center] in emo:
                four_type_utt_list.append(center)
                
            center_.append(center)
            center_label.append(emo_dict[center])
            pt = []
            pp = []
            for word in index_words[max(0, index - 8): index]:
                if word[-4] == center[-4]:
                    pt.append(word)
                else:
                    pp.append(word)

            if len(pt) != 0:
                target_.append(pt[-1])
                target_label.append(emo_dict[pt[-1]])
                target_dist.append(index - index_words.index(pt[-1]))
                if emo_dict[pt[-1]] == emo_dict[center]:
                    self_emo_shift.append(0.)
                    emo_shift_dict[center] = 0.
                else:
                    self_emo_shift.append(1.)
                    emo_shift_dict[center] = 1.
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0.)
                emo_shift_dict[center] = 0.

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift

def generate_interaction_data(train_dialog_dict, val_dialog_dict, test_dialog_dict, train_seq_dict, val_seq_dict, test_seq_dict, train_emo_all_dict, val_emo_all_dict, test_emo_all_dict, four_type_utt_list=None, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train = [], [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    center_test, target_test, opposite_test, center_label_test, target_label_test, opposite_label_test, target_dist_test, opposite_dist_test, self_emo_shift_test = [], [], [], [], [], [], [], [], []
    if mode=='context':
        generator = generate_interaction_sample
    # training set
    for k in train_dialog_dict.keys():
        dialog_order = train_dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, train_seq_dict, train_emo_all_dict, four_type_utt_list, train_emo_shift_dict)
        center_train += c
        target_train += t
        opposite_train += o
        center_label_train += cl
        target_label_train += tl
        opposite_label_train += ol
        target_dist_train += td
        opposite_dist_train += od
        self_emo_shift_train += ses
            
    # validation set
    for k in val_dialog_dict.keys():
        dialog_order = val_dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, val_seq_dict, val_emo_all_dict, four_type_utt_list, val_emo_shift_dict)
        center_val += c
        target_val += t
        opposite_val += o
        center_label_val += cl
        target_label_val += tl
        opposite_label_val += ol
        target_dist_val += td
        opposite_dist_val += od
        self_emo_shift_val += ses
        
    # test set
    for k in test_dialog_dict.keys():
        dialog_order = test_dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, test_seq_dict, test_emo_all_dict, four_type_utt_list, test_emo_shift_dict)
        center_test += c
        target_test += t
        opposite_test += o
        center_label_test += cl
        target_label_test += tl
        opposite_label_test += ol
        target_dist_test += td
        opposite_dist_test += od
        self_emo_shift_test += ses

    # save dialog pairs to train.csv and test.csv
    train_filename = 'emo_train.csv'
    val_filename = 'emo_val.csv'
    test_filename = 'emo_test.csv'
    
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)
    
    # validation
    d = {'center': center_val, 'target': target_val, 'opposite': opposite_val, 'center_label': center_label_val, 
         'target_label': target_label_val, 'opposite_label': opposite_label_val, 'target_dist': target_dist_val, 'opposite_dist': opposite_dist_val, 'self_emo_shift': self_emo_shift_val}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(val_filename, sep=',', index = False)
    
    # test
    d = {'center': center_test, 'target': target_test, 'opposite': opposite_test, 'center_label': center_label_test, 
         'target_label': target_label_test, 'opposite_label': opposite_label_test, 'target_dist': target_dist_test, 'opposite_dist': opposite_dist_test, 'self_emo_shift': self_emo_shift_test}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(test_filename, sep=',', index = False)
    
    return self_emo_shift_train

def gen_train_val_test(data_frame, X, Y, utt_name=None):
    for index, row in data_frame.iterrows():
        center_utt_name = row[0]
        target_utt_name = row[1]
        oppo_utt_name = row[2]
        
        center_utt_feat = np.array([1, 2, 3])
        target_utt_feat = np.array([1, 2, 3])
        oppo_utt_feat = np.array([1, 2, 3])
        
        #target_utt_emo = emo_num_dict[row[4]]
        #oppo_utt_emo = emo_num_dict[row[5]]
        self_emo_shift = row[-1]

        if utt_name != None: # test & val
            X.append([])
            X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
            Y.append(self_emo_shift)
            utt_name.append(center_utt_name)
            
        elif utt_name == None and center_utt_name in four_type_utt_list: # train (get four type utt only)
            X.append([])
            X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
            Y.append(self_emo_shift)

if __name__ == "__main__":
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    
    # label
    train_emo_all_dict = joblib.load('./train_emo_all.pkl')
    val_emo_all_dict = joblib.load('./val_emo_all.pkl')
    test_emo_all_dict = joblib.load('./test_emo_all.pkl')
    
    # dialog order
    train_dialog_dict = joblib.load('./train_dialog_4emo.pkl')
    val_dialog_dict = joblib.load('./val_dialog_4emo.pkl')
    test_dialog_dict = joblib.load('./test_dialog_4emo.pkl')

    train_emo_shift_dict = {}
    val_emo_shift_dict = {}
    test_emo_shift_dict = {}
    
    four_type_utt_list = [] # len:5531
    
    train_X, train_Y, test_X, test_Y = [], [], [], []

    # generate training data/val data
    generate_interaction_data(train_dialog_dict, val_dialog_dict, test_dialog_dict, {}, {}, {}, train_emo_all_dict, val_emo_all_dict, test_emo_all_dict, four_type_utt_list)
    
    cnt_1 = 0
    cnt_0 = 0
    for utt in train_emo_shift_dict:
        if train_emo_shift_dict[utt] == 1.:
            cnt_1 += 1
        else:
            cnt_0 += 1
    train_emo_shift_dict['cnt_1'] = cnt_1
    train_emo_shift_dict['cnt_0'] = cnt_0
    
    cnt_1 = 0
    cnt_0 = 0
    for utt in val_emo_shift_dict:
        if val_emo_shift_dict[utt] == 1.:
            cnt_1 += 1
        else:
            cnt_0 += 1
    val_emo_shift_dict['cnt_1'] = cnt_1
    val_emo_shift_dict['cnt_0'] = cnt_0
    
    cnt_1 = 0
    cnt_0 = 0
    for utt in test_emo_shift_dict:
        if test_emo_shift_dict[utt] == 1.:
            cnt_1 += 1
        else:
            cnt_0 += 1
    test_emo_shift_dict['cnt_1'] = cnt_1
    test_emo_shift_dict['cnt_0'] = cnt_0
        
    joblib.dump(train_emo_shift_dict, './train_4emo_shift_all.pkl')
    joblib.dump(val_emo_shift_dict, './val_4emo_shift_all.pkl')
    joblib.dump(test_emo_shift_dict, './test_4emo_shift_all.pkl')
    