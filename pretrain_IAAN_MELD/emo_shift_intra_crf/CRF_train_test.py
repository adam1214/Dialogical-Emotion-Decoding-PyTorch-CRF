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

    def __init__(self, vocab_size, emo_to_ix, device):
        super(CRF, self).__init__()
        self.vocab_size = vocab_size
        self.emo_to_ix = emo_to_ix
        self.tagset_size = len(emo_to_ix)
        self.device = device
        
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)*0.0001) #6*6
        self.weight_for_emo_shift = nn.Parameter(torch.randn(self.tagset_size-2)*0.0001) #[4]
        self.activate_fun = nn.Tanh()
        self.multiplier = nn.Parameter(torch.randn(self.tagset_size-2, self.tagset_size-2)*0.0001) #4*4
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
        
    def _forward_alg(self, feats, dialog, bias_dict):
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
                
                if next_tag < 4: # 0 1 2 3
                    multiplier_row = multiplier_after_softmax[next_tag]
                    multiplier_row.data[next_tag] = -1
                    trans_score = ( self.transitions[next_tag] + torch.cat([self.activate_fun((bias_dict[dialog[i]]-0.5)*self.weight_for_emo_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)]) ).view(1, -1)
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

    def _get_pretrain_model_features(self, dialog, out_dict):
        output_vals = np.zeros((len(dialog), 4+2))
        for i in range(0, len(dialog), 1):
            output_vals[i][0] = out_dict[ix_to_utt[dialog[i].item()]][0]
            output_vals[i][1] = out_dict[ix_to_utt[dialog[i].item()]][1]
            output_vals[i][2] = out_dict[ix_to_utt[dialog[i].item()]][2]
            output_vals[i][3] = out_dict[ix_to_utt[dialog[i].item()]][3]
            if i == 0:
                output_vals[i][4] = 3.0
            else:
                output_vals[i][4] = -3.0
            if i == len(dialog)-1:
                output_vals[i][5] = 3.0
            else:
                output_vals[i][5] = -3.0
            
        pretrain_model_feats = torch.from_numpy(output_vals)
        return pretrain_model_feats.to(self.device) # tensor: (utt數量) * (情緒數量+2)

    def _score_sentence(self, feats, tags, dialog, bias_dict):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), tags])
        
        multiplier_after_softmax = self.multiplier_softmax(self.multiplier)
        
        for i, feat in enumerate(feats):
            if tags[i + 1] < 4 and tags[i] < 4: # both in 0, 1, 2, 3
                multiplier_row = multiplier_after_softmax[tags[i + 1]]
                multiplier_row.data[tags[i + 1]] = -1
                trans_score = self.transitions[tags[i + 1], tags[i]] + (self.activate_fun((bias_dict[dialog[i]]-0.5)*self.weight_for_emo_shift))[tags[i]]*multiplier_row[tags[i]]
                
            else:
                trans_score = self.transitions[tags[i + 1], tags[i]]
            score = score + trans_score + feat[tags[i + 1]]
        score = score + self.transitions[self.emo_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats, dialog, bias_dict):
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
                
                if next_tag < 4: # 0 1 2 3
                    multiplier_row = multiplier_after_softmax[next_tag]
                    multiplier_row.data[next_tag] = -1
                    trans_score = self.transitions[next_tag] + torch.cat([self.activate_fun((bias_dict[dialog[i]]-0.5)*self.weight_for_emo_shift), torch.zeros(2).to(self.device)])*torch.cat([multiplier_row, torch.zeros(2).to(self.device)])
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

    def neg_log_likelihood(self, sentence, tags, dialog, out_dict, bias_dict):
        feats = self._get_pretrain_model_features(sentence, out_dict)
        forward_score = self._forward_alg(feats, dialog, bias_dict)
        gold_score = self._score_sentence(feats, tags, dialog, bias_dict)
        #return torch.abs(forward_score - gold_score)
        return forward_score - gold_score

    def forward(self, sentence, dialog, out_dict, bias_dict):  # dont confuse this with _forward_alg above, this for model predicting
        # Get the emission scores from the BiLSTM
        pretrain_model_feats = self._get_pretrain_model_features(sentence, out_dict)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(pretrain_model_feats, dialog, bias_dict)
        return score, tag_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', "--pretrain_version", type=str, help="which version of pretrain model you want to use?", default='original_output')
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    parser.add_argument("-s", "--seed", type=int, help="select torch seed", default = 1)
    parser.add_argument("-e", "--emo_shift", type=str, help="which emo_shift prob. to use?", default = 'model')
    args = parser.parse_args()
    print(args)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    torch.manual_seed(args.seed)
    
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    train_out_dict = joblib.load('../data/'+ args.pretrain_version + '/train_utts_logit.pkl')
    val_out_dict = joblib.load('../data/'+ args.pretrain_version + '/val_utts_logit.pkl')
    test_out_dict = joblib.load('../data/'+ args.pretrain_version + '/test_utts_logit.pkl')
    
    train_dialogs = joblib.load('../data/train_dialogs.pkl')
    train_dialogs_edit = joblib.load('../data/train_dialog_4emo.pkl')
    val_dialogs = joblib.load('../data/val_dialogs.pkl')
    val_dialogs_edit = joblib.load('../data/val_dialog_4emo.pkl')
    test_dialogs = joblib.load('../data/test_dialogs.pkl')
    test_dialogs_edit = joblib.load('../data/test_dialog_4emo.pkl')
    
    if args.dataset == 'original':
        train_emo_dict = joblib.load('../data/train_emo_all.pkl')
        train_dias = train_dialogs_edit
        val_emo_dict = joblib.load('../data/val_emo_all.pkl')
        val_dias = val_dialogs_edit
        test_emo_dict = joblib.load('../data/test_emo_all.pkl')
        test_dias = test_dialogs_edit
    '''
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/'+ args.pretrain_version + '/U2U_4emo.pkl')
        dias = dialogs
    '''  
    
    if args.emo_shift == 'model':
        train_bias_dict = joblib.load('../data/train_4emo_shift_all.pkl')
        val_bias_dict = joblib.load('../data/val_4emo_shift_all.pkl')
        test_bias_dict = joblib.load('../data/test_4emo_shift_all.pkl')
        
    # Make up training data & testing data
    train_spk_dialogs = utils.split_dialog(train_dias)
    val_spk_dialogs = utils.split_dialog(val_dias)
    test_spk_dialogs = utils.split_dialog(test_dias)
    
    train_data = []
    val_data = []
    test_data = []
    for dialog in train_spk_dialogs:
        train_data.append((train_spk_dialogs[dialog], []))
        for utt in train_data[-1][0]:
            train_data[-1][1].append(train_emo_dict[utt])
    for dialog in val_spk_dialogs:
        val_data.append((val_spk_dialogs[dialog], []))
        for utt in val_data[-1][0]:
            val_data[-1][1].append(val_emo_dict[utt])
    for dialog in test_spk_dialogs:
        test_data.append((test_spk_dialogs[dialog], []))
        for utt in test_data[-1][0]:
            test_data[-1][1].append(test_emo_dict[utt])

    utt_to_ix = {}
    for dialog, emos in train_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in val_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    
    ix_to_utt = {}
    for key in utt_to_ix:
        val = utt_to_ix[key]
        ix_to_utt[val] = key

    emo_to_ix = {'anger':0, 'joy':1, 'neutral':2, 'sadness':3, START_TAG: 4, STOP_TAG: 5}

    label_val = []
    for dia_emos_tuple in val_data:
        label_val += dia_emos_tuple[1]
    for i in range(0, len(label_val), 1):
        if label_val[i] == 'anger':
            label_val[i] = 0
        elif label_val[i] == 'joy':
            label_val[i] = 1
        elif label_val[i] == 'neutral':
            label_val[i] = 2
        elif label_val[i] == 'sadness':
            label_val[i] = 3
        else:
            label_val[i] = -1
    
    model = CRF(len(utt_to_ix), emo_to_ix, device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.00001, weight_decay=0.01)

    max_uar_val = 0
    best_epoch = -1
    loss_list = []
    val_loss_list = []
    train_loss_list = []
    for epoch in range(3):
        train_loss_sum = 0
        model.train()
        for dialog, emos in train_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            dialog_in = prepare_dialog(dialog, utt_to_ix)
            targets = torch.tensor([emo_to_ix[t] for t in emos], dtype=torch.long)
    
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(dialog_in, targets, dialog, train_out_dict, train_bias_dict)
            train_loss_sum += loss.item()
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        train_loss_list.append(train_loss_sum/len(train_data))
        
        #check model performance on predefined validation set
        predict_val = []
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            for i in range(0, len(val_data), 1):
                precheck_dia = prepare_dialog(val_data[i][0], utt_to_ix)
                predict_val += model(precheck_dia, val_data[i][0], val_out_dict, val_bias_dict)[1]
                
                targets = torch.tensor([emo_to_ix[t] for t in val_data[i][1]], dtype=torch.long)
                loss = model.neg_log_likelihood(precheck_dia, targets, val_data[i][0], val_out_dict, val_bias_dict)
                val_loss_sum += loss.item()
                
        uar_val, acc_val, conf_val = utils.evaluate(predict_val, label_val)
        val_loss_list.append(val_loss_sum/len(val_data))

        #Save the best model so far
        if uar_val > max_uar_val:
            val_loss_sum = 0
            best_epoch = epoch
            max_uar_val = uar_val
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
            torch.save(checkpoint, './model/' + args.pretrain_version + '/' + args.dataset + '/best_model.pth')
        print('EPOCH:', epoch, ', train_loss:', round(train_loss_list[-1], 4), ', val_loss:', round(val_loss_list[-1], 4), ', val_uar:', round(100 * uar_val, 2), '%')
            
    print('The Best Epoch:', best_epoch)
        
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, 's-', color = 'r', label="train_loss")
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, 's-', color = 'b', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = "best")
    plt.savefig('./loss_curve.png')
    
    # Load model for testing
    model = CRF(len(utt_to_ix), emo_to_ix, device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pred_dict = {}
    with torch.no_grad():
        for i in range(0, len(test_data), 1):
            precheck_dia = prepare_dialog(test_data[i][0], utt_to_ix)
            batch_predict_seq = model(precheck_dia, test_data[i][0], test_out_dict, test_bias_dict)[1]
            for j, utt in enumerate(test_data[i][0]):
                pred_dict[utt] = batch_predict_seq[j]
                
    joblib.dump(pred_dict, './model/' + args.pretrain_version + '/' + args.dataset + '/preds_4.pkl')
    ori_emo_dict = joblib.load('../data/test_emo_all.pkl')
    
    label = []
    predict = []
    emo2num = {'anger':0, 'joy':1, 'neutral':2, 'sadness':3}
    for dialog_name in test_dialogs_edit:
        for utt in test_dialogs_edit[dialog_name]:
            label.append(emo2num[test_emo_dict[utt]])
            predict.append(pred_dict[utt])
            
    uar, acc, conf = utils.evaluate(predict, label, final_test=1)
    print('UAR:', round(uar*100, 2))
    print('ACC:', round(acc*100, 2))
    print(conf)
    print('##########')
    
    label = []
    for _, dia in enumerate(test_dialogs):
        label += [utils.convert_to_index(ori_emo_dict[utt]) for utt in test_dialogs[dia]]
    
    pretrained_predict = []
    for _, dia in enumerate(test_dialogs):
        pretrained_predict += [test_out_dict[utt].argmax() for utt in test_dialogs[dia]]
    
    
    uar, acc, conf = utils.evaluate(pretrained_predict, label, final_test=1)
    print('pretrained UAR:', round(uar*100, 2))
    print('pretrained ACC:', round(acc*100, 2))
    print(conf)