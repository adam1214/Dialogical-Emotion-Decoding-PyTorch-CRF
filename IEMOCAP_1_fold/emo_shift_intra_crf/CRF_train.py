import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils
import matplotlib.pyplot as plt
import pdb
import math
import os
import random

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_dialog(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        
class CRF(nn.Module):

    def __init__(self, vocab_size, emo_to_ix, out_dict, bias_dict, ix_to_utt, device):
        super(CRF, self).__init__()
        self.vocab_size = vocab_size
        self.emo_to_ix = emo_to_ix
        self.tagset_size = len(emo_to_ix)
        self.out_dict = out_dict
        self.bias_dict = bias_dict
        self.ix_to_utt = ix_to_utt
        self.device = device
        
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)) #6*6
        #self.weight_for_emo_with_shift_in_activate = nn.Parameter(torch.randn(self.tagset_size-2)) #[4]
        self.weight_for_emo_shift_in_activate = nn.Parameter(torch.randn(self.tagset_size-2)) #[4]
        self.bias_in_activate_no_shift = nn.Parameter(torch.randn(1)) #[1]
        self.bias_in_activate_with_shift = nn.Parameter(torch.randn(1)) #[1]
        
        self.weight_for_emo_with_shift_out_activate = nn.Parameter(torch.randn(self.tagset_size-2)) #[4]
        self.weight_for_emo_no_shift_out_activate = nn.Parameter(torch.randn(self.tagset_size-2)) #[4]
        self.activate_fun = nn.Tanh()
        self.multiplier = nn.Parameter(torch.randn(self.tagset_size-2, self.tagset_size-2)) #4*4
        self.multiplier_softmax = nn.Softmax(dim=0)
        #self.diagonal_neg_infinity_matrix = torch.zeros(self.tagset_size-2, self.tagset_size-2) #4*4
        for i in range(0, self.tagset_size-2, 1):
            #self.diagonal_neg_infinity_matrix[i][i] = -math.inf
            self.multiplier.data[i][i] = -10000.
        
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, emo_to_ix[STOP_TAG]] = -10000
        '''
            0	            1	            2	            3	            4	            5        (start point)
        0	-0.015255959	-0.007502318	-0.006539809	-0.016094847	-0.0010016718	-10000.0
        1	-0.009797722	-0.016090963	-0.007121446	0.0030372199	-0.007773143	-10000.0
        2	-0.002222705	0.016871134	    0.0022842516	0.004676355	    -0.0069697243	-10000.0
        3	0.0069954237	0.0019908163	0.001990565	    0.00045702778	0.0015295692	-10000.0
        4	-10000.0	   -10000.0	        -10000.0	    -10000.0	    -10000.0	    -10000.0
        5	0.015747981	   -0.006298472	    0.02406978	    0.0027856624	0.0024675291	-10000.0
        (end point)
        '''

    def _forward_alg(self, feats, dialog):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.emo_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        
        multiplier_after_softmax = self.multiplier_softmax(self.multiplier)
        
        # Iterate through the sentence
        for i, feat in enumerate(feats):
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                if next_tag < (self.tagset_size-2): # 0 1 2 3
                    multiplier_row = multiplier_after_softmax[next_tag]
                    multiplier_row.data[next_tag] = -1
                    if self.bias_dict[dialog[i]] > 0.5:
                        trans_score = ( self.transitions[next_tag] + torch.cat([self.weight_for_emo_with_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_with_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)]) ).view(1, -1)
                    else:
                        trans_score = ( self.transitions[next_tag] + torch.cat([self.weight_for_emo_no_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_no_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)]) ).view(1, -1)
                else:
                    trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.emo_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_pretrain_model_features(self, dialog):
        output_vals = np.zeros((len(dialog), self.tagset_size))
        for i in range(0, len(dialog), 1):
            for j in range(0, self.tagset_size-2, 1):
                output_vals[i][j] = self.out_dict[self.ix_to_utt[dialog[i].item()]][j]
            if i == 0:
                output_vals[i][-2] = 3.0
            else:
                output_vals[i][-2] = -3.0
            if i == len(dialog)-1:
                output_vals[i][-1] = 3.0
            else:
                output_vals[i][-1] = -3.0
        pretrain_model_feats = torch.from_numpy(output_vals)
        return pretrain_model_feats.to(self.device) # tensor: (utt數量) * (情緒數量+2)

    def _score_sentence(self, feats, tags, dialog):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), tags])
        
        multiplier_after_softmax = self.multiplier_softmax(self.multiplier)
        
        for i, feat in enumerate(feats):
            if tags[i + 1] < (self.tagset_size-2) and tags[i] < (self.tagset_size-2): # both in 0, 1, 2, 3
                multiplier_row = multiplier_after_softmax[tags[i + 1]]
                multiplier_row.data[tags[i + 1]] = -1
                if self.bias_dict[dialog[i]] > 0.5:
                    trans_score = self.transitions[tags[i + 1], tags[i]] + (self.weight_for_emo_with_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_with_shift))[tags[i]]*multiplier_row[tags[i]]
                else:
                    trans_score = self.transitions[tags[i + 1], tags[i]] + (self.weight_for_emo_no_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_no_shift))[tags[i]]*multiplier_row[tags[i]]
            else:
                trans_score = self.transitions[tags[i + 1], tags[i]]
            score = score + trans_score + feat[tags[i + 1]]
        score = score + self.transitions[self.emo_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats, dialog):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.emo_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        
        multiplier_after_softmax = self.multiplier_softmax(self.multiplier)
        
        for i, feat in enumerate(feats):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                
                if next_tag < (self.tagset_size-2): # 0 1 2 3
                    multiplier_row = multiplier_after_softmax[next_tag]
                    multiplier_row.data[next_tag] = -1
                    if self.bias_dict[dialog[i]] > 0.5:
                        trans_score = self.transitions[next_tag] + torch.cat([self.weight_for_emo_with_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_with_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)])
                    else:
                        trans_score = self.transitions[next_tag] + torch.cat([self.weight_for_emo_no_shift_out_activate*self.activate_fun(self.bias_dict[dialog[i]]*self.weight_for_emo_shift_in_activate+self.bias_in_activate_no_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)])
                else:
                    trans_score = self.transitions[next_tag]
                
                next_tag_var = forward_var + trans_score
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.emo_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.emo_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, dialog):
        feats = self._get_pretrain_model_features(sentence)
        forward_score = self._forward_alg(feats, dialog)
        gold_score = self._score_sentence(feats, tags, dialog)
        #return torch.abs(forward_score - gold_score)
        return forward_score - gold_score

    def forward(self, sentence, dialog):  # dont confuse this with _forward_alg above, this for model predicting
        # Get the emission scores from the BiLSTM
        pretrain_model_feats = self._get_pretrain_model_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(pretrain_model_feats, dialog)
        return score, tag_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use?\niemocap_original_4\niemocap_original_6\niemocap_U2U_4\niemocap_U2U_6\nmeld", default = 'iemocap_original_4')
    parser.add_argument("-s", "--seed", type=int, help="select torch seed", default = 1)
    parser.add_argument("-e", "--emo_shift", type=str, help="which emo_shift prob. to use?", default = 'model')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5
    
    if args.dataset == 'iemocap_original_4':
        emo_dict = joblib.load('../data/iemocap/four_type/emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/four_type/dialog_rearrange_4emo_iemocap.pkl')
        #emo_dict = joblib.load('../data/iemocap/emo_all_iemocap_4.pkl')
        #dias = joblib.load('../data/iemocap/dialog_edit.pkl')
    elif args.dataset == 'iemocap_original_6':
        emo_dict = joblib.load('../data/iemocap/new_audio_text/emo_all_6.pkl')
        dias = joblib.load('../data/iemocap/new_audio_text/dialog_rearrange_6_emo.pkl')
        #emo_dict = joblib.load('../data/iemocap/old_text/emo_all_iemocap_6.pkl')
        #dias = joblib.load('../data/iemocap/old_text/dialog_edit.pkl')
    elif args.dataset == 'iemocap_U2U_4':
        emo_dict = joblib.load('../data/iemocap/U2U_4emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/dialog.pkl')
    elif args.dataset == 'iemocap_U2U_6':
        emo_dict = joblib.load('../data/iemocap/U2U_6emo_all_iemocap.pkl')
        dias = joblib.load('../data/iemocap/dialog.pkl')
    elif args.dataset == 'meld':
        emo_dict = joblib.load('../data/meld/emo_all_meld.pkl')
        dias = joblib.load('../data/meld/dialog_meld.pkl')

    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U_6':
        emo_to_ix = {'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5, START_TAG: 6, STOP_TAG: 7}
    elif args.dataset == 'iemocap_original_4' or args.dataset == 'iemocap_U2U_4':
        emo_to_ix = {'hap':1, 'neu':2, 'sad':3, 'ang':0, START_TAG: 4, STOP_TAG: 5}
    elif args.dataset == 'meld':
        emo_to_ix = {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6, START_TAG: 7, STOP_TAG: 8}

    if args.dataset == 'iemocap_original_4' or args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U_4' or args.dataset == 'iemocap_U2U_6':
        out_dict = {}
        for utt in emo_dict:
            if emo_dict[utt] in ['neu', 'sad', 'hap', 'ang']:
                #out_dict[utt] = np.zeros(6, dtype=np.float32)
                #out_dict[utt][emo_to_ix[emo_dict[utt]]] = 1.
                out_dict[utt] = np.ones(4, dtype=np.float32) * 0.
                out_dict[utt][emo_to_ix[emo_dict[utt]]] = 1.
        
    elif args.dataset == 'meld':
        out_dict = {}
    
    
    if args.emo_shift == 'constant':
        spk_dialogs = utils.split_dialog(dias)
        bias_dict = utils.get_val_bias(spk_dialogs, emo_dict)
    else:
        if 'iemocap' in args.dataset:
            bias_dict = joblib.load('../data/iemocap/four_type/4emo_shift_all_rearrange.pkl')
            #bias_dict = joblib.load('../data/iemocap/new_audio_text/6_emo_shift_all_rearrange.pkl')
            #bias_dict = joblib.load('../data/iemocap/old_text/6emo_shift_all.pkl') 
        else:
            bias_dict = joblib.load('../data/meld/7emo_shift_all.pkl')
        # replace with margin
        '''
        for utt in bias_dict:
            if 'Ses0' in utt:
                if bias_dict[utt] == 1.0:
                    bias_dict[utt] = 1.2
                else:
                    bias_dict[utt] = -0.2
        '''
        '''
        emo_prob_fold1 = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_output_fold1.pkl')
        emo_prob_fold2 = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_output_fold2.pkl')
        emo_prob_fold3 = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_output_fold3.pkl')
        emo_prob_fold4 = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_output_fold4.pkl')
        emo_prob_fold5 = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_output_fold5.pkl')
        if args.model_num == 1:
            bias_dict = emo_prob_fold1
        elif args.model_num == 2:
            bias_dict = emo_prob_fold2
        elif args.model_num == 3:
            bias_dict = emo_prob_fold3
        elif args.model_num == 4:
            bias_dict = emo_prob_fold4
        else:
            bias_dict = emo_prob_fold5
        '''
        '''
        if args.model_num == 1:
            bias_dict = joblib.load('../data/'+ args.pretrain_version + '/NB_emo_shift_output_fold1.pkl')
        elif args.model_num == 2:
            bias_dict = joblib.load('../data/'+ args.pretrain_version + '/NB_emo_shift_output_fold2.pkl')
        elif args.model_num == 3:
            bias_dict = joblib.load('../data/'+ args.pretrain_version + '/NB_emo_shift_output_fold3.pkl')
        elif args.model_num == 4:
            bias_dict = joblib.load('../data/'+ args.pretrain_version + '/NB_emo_shift_output_fold4.pkl')
        else:
            bias_dict = joblib.load('../data/'+ args.pretrain_version + '/NB_emo_shift_output_fold5.pkl')
        '''
        '''
        bias_dict = joblib.load('../data/'+ args.pretrain_version + '/iaan_emo_shift_variant_output.pkl')
        for k in bias_dict:
            if bias_dict[k] > 0.5:
                bias_dict[k] = 1.0
            else:
                bias_dict[k] = 0.0
        '''

    # Make up training data & testing data
    train_data = []
    val_data = []
    test_data = []
    if args.dataset == 'iemocap_original_6' or args.dataset == 'iemocap_U2U' or args.dataset == 'iemocap_original_4':
        val_dias_set = joblib.load('../data/iemocap/val_dias_set.pkl')
        for dialog in dias:
            if dialog[4] == '5':
                test_data.append(([],[]))
                test_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        test_data[-2][0].append(utt)
                        test_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        test_data[-1][0].append(utt)
                        test_data[-1][1].append(emo_dict[utt])
                if len(test_data[-2][0]) == 0:
                    del test_data[-2]
                if len(test_data[-1][0]) == 0:
                    del test_data[-1]
                    
            elif dialog in val_dias_set:
                val_data.append(([],[]))
                val_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        val_data[-2][0].append(utt)
                        val_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        val_data[-1][0].append(utt)
                        val_data[-1][1].append(emo_dict[utt])
                if len(val_data[-2][0]) == 0:
                    del val_data[-2]
                if len(val_data[-1][0]) == 0:
                    del val_data[-1]
                    
            else:
                train_data.append(([],[]))
                train_data.append(([],[]))
                for utt in dias[dialog]:
                    if utt[-4] == 'M':
                        train_data[-2][0].append(utt)
                        train_data[-2][1].append(emo_dict[utt])
                    elif utt[-4] == 'F':
                        train_data[-1][0].append(utt)
                        train_data[-1][1].append(emo_dict[utt])
                if len(train_data[-2][0]) == 0:
                    del train_data[-2]
                if len(train_data[-1][0]) == 0:
                    del train_data[-1]
                    
    elif args.dataset == 'meld':
        for dialog in dias:
            if dialog.split('_')[0] == 'train':
                train_data.append((dias[dialog],[]))
                for utt in train_data[-1][0]:
                    train_data[-1][1].append(emo_dict[utt])
            elif dialog.split('_')[0] == 'val':
                val_data.append((dias[dialog],[]))
                for utt in val_data[-1][0]:
                    val_data[-1][1].append(emo_dict[utt])
            elif dialog.split('_')[0] == 'test':
                test_data.append((dias[dialog],[]))
                for utt in test_data[-1][0]:
                    test_data[-1][1].append(emo_dict[utt])
    
    utt_to_ix = {}
    for dialog, emos in train_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in val_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)

    ix_to_utt = {}
    for key in utt_to_ix:
        val = utt_to_ix[key]
        ix_to_utt[val] = key
        
    label_val = []
    for dia_emos_tuple in val_data:
        label_val += dia_emos_tuple[1]
    for i in range(0, len(label_val), 1):
        if emo_to_ix.get(label_val[i]) != None:
            label_val[i] = emo_to_ix.get(label_val[i])
        else:
            label_val[i] = -1
    
    model = CRF(len(utt_to_ix), emo_to_ix, out_dict, bias_dict, ix_to_utt, device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.05, momentum=0.5)

    #max_uar_val = 0
    max_f1_val = 0
    best_epoch = -1
    val_loss_list = []
    train_loss_list = []
    for epoch in range(1, 70+1, 1):
        train_loss_sum = 0
        for dialog, emos in train_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            dialog_in = prepare_dialog(dialog, utt_to_ix)
            targets = torch.tensor([emo_to_ix[t] for t in emos], dtype=torch.long)
    
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(dialog_in, targets, dialog)
            train_loss_sum += loss.item()
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        train_loss_list.append(train_loss_sum/len(train_data))
        
        #check model performance on predefined validation set
        predict_val = []
        with torch.no_grad():
            val_loss_sum = 0
            for i in range(0, len(val_data), 1):
                precheck_dia = prepare_dialog(val_data[i][0], utt_to_ix)
                predict_val += model(precheck_dia, val_data[i][0])[1]
                
                targets = torch.tensor([emo_to_ix[t] for t in val_data[i][1]], dtype=torch.long)
                loss = model.neg_log_likelihood(precheck_dia, targets, val_data[i][0])
                val_loss_sum += loss.item()
                
        uar_val, acc_val, f1_val, conf_val = utils.evaluate(predict_val, label_val)
        val_loss_list.append(val_loss_sum/len(val_data))
        
        #Save the best model so far
        if f1_val > max_f1_val:
            best_epoch = epoch
            max_f1_val = f1_val
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
            if args.dataset == 'iemocap_original_4':
                torch.save(checkpoint, './model/iemocap/original/model' + str(args.seed) + '.pth')
            elif args.dataset == 'iemocap_U2U':
                torch.save(checkpoint, './model/iemocap/U2U/model' + str(args.seed) + '.pth')
            elif args.dataset == 'meld':
                torch.save(checkpoint, './model/meld/model' + str(args.seed) + '.pth')
        print('EPOCH:', epoch, ', train_loss:', round(train_loss_list[-1], 4), ', val_loss:', round(val_loss_list[-1], 4), ', val_uar:', round(100 * uar_val, 2), '%')
    print('The Best Epoch:', best_epoch)

    plt.plot(np.arange(len(train_loss_list))+1, train_loss_list, 's-', color = 'r', label="train_loss")
    plt.plot(np.arange(len(val_loss_list))+1, val_loss_list, 's-', color = 'b', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = "best")
    plt.savefig('./result.png')