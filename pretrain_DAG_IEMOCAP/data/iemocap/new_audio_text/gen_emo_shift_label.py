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

def generate_interaction_sample(index_words, seq_dict, emo_dict, only_four, val=False):
    """ 
    Generate interaction training pairs,
    total 4 class, total 5531 emo samples."""
    emo = ['ang', 'hap', 'neu', 'sad', 'fru', 'exc']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    self_emo_shift = []
    for index, center in enumerate(index_words):
        if (only_four and emo_dict[center] in emo) or only_four == False:
            if emo_dict[center] in emo:
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
                    self_emo_shift.append(0)
                    utt_emo_shift_dict[center] = 0.
                else:
                    self_emo_shift.append(1)
                    utt_emo_shift_dict[center] = 1.
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)
                utt_emo_shift_dict[center] = 0.

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, val_set, mode='context', only_four=False):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train = [], [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        # training set
        if val_set not in k:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict, only_four)
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
        else:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict, only_four, val=True)
            center_val += c
            target_val += t
            opposite_val += o
            center_label_val += cl
            target_label_val += tl
            opposite_label_val += ol
            target_dist_val += td
            opposite_dist_val += od
            self_emo_shift_val += ses

    # save dialog pairs to train.csv and test.csv
    train_filename= './emo_train.csv'
    val_filename= './emo_test.csv'
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

def gen_train_test_pair(data_frame, X, Y, test_utt_name=None):
    for index, row in data_frame.iterrows():
        X.append([])
        center_utt_name = row[0]
        target_utt_name = row[1]
        oppo_utt_name = row[2]
        
        center_utt_feat = np.zeros((2, 45))
        target_utt_feat = np.zeros((2, 45))
        oppo_utt_feat = np.zeros((2, 45))
        
        #target_utt_emo = emo_num_dict[row[4]]
        #oppo_utt_emo = emo_num_dict[row[5]]
        self_emo_shift = row[-1]
        
        X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
        if center_utt_name in four_type_utt_list:
            Y.append(self_emo_shift)

        if test_utt_name != None:
            test_utt_name.append(center_utt_name)
        
def upsampling(X, Y):
    #counter = Counter(Y)
    #print(counter)
    
    # transform the dataset
    #oversample = SMOTE(random_state=100, n_jobs=-1, sampling_strategy='auto', k_neighbors=5)
    oversample = RandomOverSampler(random_state=100)
    #oversample = ClusterCentroids(random_state=100, n_jobs=-1)
    #oversample = SMOTETomek(random_state=100, n_jobs=-1, sampling_strategy='auto')
    X_upsample, Y_upsample = oversample.fit_resample(np.array(X).squeeze(1), Y)
    
    #counter = Counter(Y_upsample)
    #print(counter)

    return X_upsample, Y_upsample

def gen_train_val_test(data_frame, X, Y, utt_name=None):
    for index, row in data_frame.iterrows():
        center_utt_name = row[0]
        target_utt_name = row[1]
        oppo_utt_name = row[2]
        
        center_utt_feat = np.zeros((2, 45))
        target_utt_feat = np.zeros((2, 45))
        oppo_utt_feat = np.zeros((2, 45))
        
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
    feat_pooled = {}
    
    # label
    emo_all_dict = joblib.load('./emo_all_6.pkl')
    
    # dialog order
    #dialog_dict = joblib.load('./data/dialog_rearrange.pkl')
    dialog_dict = joblib.load('./dialog_rearrange_6_emo.pkl')
    
    val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    utt_emo_shift_dict = {}
    for val_ in val:
        four_type_utt_list = [] # len:5531
        print("################{}################".format(val_))
        
        train_X, train_Y, test_X, test_Y = [], [], [], []
        test_utt_name = []

        # generate training data/val data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set=val_)
        emo_train = pd.read_csv('./emo_train.csv')
        emo_test = pd.read_csv('./emo_test.csv')
        break
    joblib.dump(utt_emo_shift_dict, './6_emo_shift_all_rearrange.pkl')

    for val_ in val:
        train_X, train_Y = [], []
        test_utt_name = []

        # generate training data/val data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set=val_, only_four=True)
        emo_train = pd.read_csv('./emo_train.csv')
        gen_train_val_test(emo_train, train_X, train_Y)
        counter = Counter(train_Y)
        utt_emo_shift_dict = joblib.load('./6_emo_shift_all_rearrange.pkl')
        utt_emo_shift_dict['fold'+val_[-1]+'_0'] = counter[0]
        utt_emo_shift_dict['fold'+val_[-1]+'_1'] = counter[1]
        joblib.dump(utt_emo_shift_dict, './6_emo_shift_all_rearrange.pkl')
        