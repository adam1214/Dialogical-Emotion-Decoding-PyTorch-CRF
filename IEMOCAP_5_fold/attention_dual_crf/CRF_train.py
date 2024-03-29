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
import random

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

    def __init__(self, vocab_size, emo_to_ix):
        super(CRF, self).__init__()
        self.vocab_size = vocab_size
        self.emo_to_ix = emo_to_ix
        self.tagset_size = len(emo_to_ix)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions_inter = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)) #6*6
        self.transitions_intra = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)) #6*6
        if args.attention == 'logit':
            self.weight_spk_unchange = nn.Parameter(torch.randn(1, (self.tagset_size-2)*2)) # 1*8
            self.weight_spk_change = nn.Parameter(torch.randn(1, (self.tagset_size-2)*2)) # 1*8
        elif args.attention == 'concat_representation':
            self.weight_spk_unchange = nn.Parameter(torch.randn(1, 2*len(utts_concat_representation['Ses01F_impro01_F000']))) # 1*(2*1536)
            self.weight_spk_change = nn.Parameter(torch.randn(1, 2*len(utts_concat_representation['Ses01F_impro01_F000']))) # 1*(2*1536)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions_inter.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions_inter.data[:, emo_to_ix[STOP_TAG]] = -10000

        self.transitions_intra.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions_intra.data[:, emo_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats, dialog):
        sigmoid_fun = nn.Sigmoid()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.emo_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the dialog
        i = 0
        for feat in feats:
            if i > 0:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[i]]).unsqueeze(0).T), 0) #(8,1)
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[i]]).unsqueeze(0).T), 0) # (1536*2,1)
            else:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T), 0) #(8, 1)
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T), 0) #(1536*2, 1)
            
            attention_alpha_spk_change = sigmoid_fun(torch.matmul(self.weight_spk_change, concat_utt_representation)) #1*1
            attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1

            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                if i == 0 or dialog[i-1][-4] == dialog[i][-4]:
                    trans_score = (attention_alpha_spk_unchange[0][0]*self.transitions_inter[next_tag] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[next_tag]).view(1, -1)
                else:
                    trans_score = (attention_alpha_spk_change[0][0]*self.transitions_inter[next_tag] + (1-attention_alpha_spk_change[0][0])*self.transitions_intra[next_tag]).view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            i += 1
        
        if args.attention == 'logit':
            concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T), 0) #8,1
        elif args.attention == 'concat_representation':
            concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T), 0) #1536*2,1
        
        attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1
        terminal_var = forward_var + (attention_alpha_spk_unchange[0][0]*self.transitions_inter[self.emo_to_ix[STOP_TAG]] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[self.emo_to_ix[STOP_TAG]])
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_pretrain_model_features(self, dialog):
        output_vals = np.zeros((len(dialog), self.tagset_size))
        for i in range(0, len(dialog), 1):
            for j in range(0, self.tagset_size-2, 1):
                output_vals[i][j] = out_dict[ix_to_utt[dialog[i].item()]][j]

            if i == 0:
                output_vals[i][-2] = 100.0
            else:
                output_vals[i][-2] = -100.0
            if i == len(dialog) - 1:
                output_vals[i][-1] = 100.0
            else:
                output_vals[i][-1] = -100.0
            
        pretrain_model_feats = torch.from_numpy(output_vals)
        return pretrain_model_feats # tensor: (utt數量) * (情緒數量+2)

    def _score_dialog(self, feats, emos, dialog):
        sigmoid_fun = nn.Sigmoid()
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        emos = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), emos])
        for i, feat in enumerate(feats):
            if i > 0:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[i]]).unsqueeze(0).T), 0) #4*2,1
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[i]]).unsqueeze(0).T), 0) #1536*2,1
            else:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T), 0) #4*2,1
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T), 0) #1536*2,1

            attention_alpha_spk_change = sigmoid_fun(torch.matmul(self.weight_spk_change, concat_utt_representation)) #1*1
            attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1

            if i == 0 or dialog[i-1][-4] == dialog[i][-4]:
                score = score + (attention_alpha_spk_unchange[0][0]*self.transitions_inter[emos[i + 1], emos[i]] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[emos[i + 1], emos[i]]) + feat[emos[i + 1]]
            else:
                score = score + (attention_alpha_spk_change[0][0]*self.transitions_inter[emos[i + 1], emos[i]] + (1-attention_alpha_spk_change[0][0])*self.transitions_intra[emos[i + 1], emos[i]]) + feat[emos[i + 1]]
        
        if args.attention == 'logit':
            concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T), 0) #4*2,1
        elif args.attention == 'concat_representation':
            concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T), 0) #1536*2,1
            
        attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1
        score = score + (attention_alpha_spk_unchange[0][0]*self.transitions_inter[self.emo_to_ix[STOP_TAG], emos[-1]] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[self.emo_to_ix[STOP_TAG], emos[-1]])
        return score

    def _viterbi_decode(self, feats, dialog):
        sigmoid_fun = nn.Sigmoid()
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.emo_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        i = 0
        for feat in feats:
            if i > 0:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[i]]).unsqueeze(0).T), 0) #4*2,1
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[i-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[i]]).unsqueeze(0).T), 0) #1536*2,1
            else:
                if args.attention == 'logit':
                    concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[0]]).unsqueeze(0).T), 0) #4*2,1
                elif args.attention == 'concat_representation':
                    concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[0]]).unsqueeze(0).T), 0) #1536*2,1

            attention_alpha_spk_change = sigmoid_fun(torch.matmul(self.weight_spk_change, concat_utt_representation)) #1*1
            attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1

            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                if i == 0 or dialog[i-1][-4] == dialog[i][-4]:
                    next_tag_var = forward_var + (attention_alpha_spk_unchange[0][0]*self.transitions_inter[next_tag] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[next_tag])
                else:
                    next_tag_var = forward_var + (attention_alpha_spk_change[0][0]*self.transitions_inter[next_tag] + (1-attention_alpha_spk_change[0][0])*self.transitions_intra[next_tag])
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            i += 1
            
        # Transition to STOP_TAG
        if args.attention == 'logit':
            concat_utt_representation = torch.cat((torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(out_dict[dialog[-1]]).unsqueeze(0).T), 0) #4*2,1
        elif args.attention == 'concat_representation':
            concat_utt_representation = torch.cat((torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T, torch.from_numpy(utts_concat_representation[dialog[-1]]).unsqueeze(0).T), 0) #1536*2,1
            
        attention_alpha_spk_unchange = sigmoid_fun(torch.matmul(self.weight_spk_unchange, concat_utt_representation)) #1*1
        terminal_var = forward_var + (attention_alpha_spk_unchange[0][0]*self.transitions_inter[self.emo_to_ix[STOP_TAG]] + (1-attention_alpha_spk_unchange[0][0])*self.transitions_intra[self.emo_to_ix[STOP_TAG]])
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

    def neg_log_likelihood(self, dialog_tensor, emos, dialog):
        feats = self._get_pretrain_model_features(dialog_tensor)
        forward_score = self._forward_alg(feats, dialog)
        gold_score = self._score_dialog(feats, emos, dialog)
        return forward_score - gold_score

    def forward(self, dialog_tensor, dialog):  # dont confuse this with _forward_alg above, this for model predicting
        # Get the emission scores from the pretrain model
        pretrain_model_feats = self._get_pretrain_model_features(dialog_tensor)

        # Find the best path, given the features.
        score, emo_seq = self._viterbi_decode(pretrain_model_feats, dialog)
        return score, emo_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-a', "--attention", type=str, help="which tensor you want for attention?", default='logit')
    parser.add_argument('-v', "--pretrain_version", type=str, help="which version of pretrain model you want to use?", default='original_output')
    parser.add_argument("-n", "--model_num", type=int, help="which model number you want to train?", default=1)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    parser.add_argument("-s", "--seed", type=int, help="select torch seed", default = 1)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    out_dict = joblib.load('../data/'+ args.pretrain_version + '/outputs.pkl')
    dialogs = joblib.load('../data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
        dias = dialogs_edit
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/'+ args.pretrain_version + '/U2U_4emo_all_iemocap.pkl')
        dias = dialogs
    
    if args.pretrain_version != 'original_output':
        utts_concat_representation = joblib.load('../data/'+ args.pretrain_version + '/utts_concat_representation.pkl')
    
    # Make up training data & testing data
    model_num_val_map = {1:'5', 2:'4', 3:'2', 4:'1', 5: '3'}
    train_data = []
    val_data = []
    for dialog in dias:
        if dialog[4] != str(args.model_num) and dialog[4] != model_num_val_map[args.model_num]: # assign to train set
            train_data.append((dias[dialog],[]))
            for utt in train_data[-1][0]:
                train_data[-1][1].append(emo_dict[utt])
        elif dialog[4] == model_num_val_map[args.model_num]: # assign to val set
            val_data.append((dias[dialog],[]))
            for utt in val_data[-1][0]:
                val_data[-1][1].append(emo_dict[utt])

    test_data_Ses01 = []
    test_data_Ses02 = []
    test_data_Ses03 = []
    test_data_Ses04 = []
    test_data_Ses05 = []
    for dialog in dias:
        if dialog[4] == '1':
            test_data_Ses01.append((dias[dialog],[]))
            for utt in test_data_Ses01[-1][0]:
                test_data_Ses01[-1][1].append(emo_dict[utt])
        elif dialog[4] == '2':
            test_data_Ses02.append((dias[dialog],[]))
            for utt in test_data_Ses02[-1][0]:
                test_data_Ses02[-1][1].append(emo_dict[utt])
        elif dialog[4] == '3':
            test_data_Ses03.append((dias[dialog],[]))
            for utt in test_data_Ses03[-1][0]:
                test_data_Ses03[-1][1].append(emo_dict[utt])
        elif dialog[4] == '4':
            test_data_Ses04.append((dias[dialog],[]))
            for utt in test_data_Ses04[-1][0]:
                test_data_Ses04[-1][1].append(emo_dict[utt])
        elif dialog[4] == '5':
            test_data_Ses05.append((dias[dialog],[]))
            for utt in test_data_Ses05[-1][0]:
                test_data_Ses05[-1][1].append(emo_dict[utt])

    utt_to_ix = {}
    for dialog, emos in test_data_Ses01:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses02:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses03:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses04:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in test_data_Ses05:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    
    ix_to_utt = {}
    for key in utt_to_ix:
        val = utt_to_ix[key]
        ix_to_utt[val] = key

    emo_to_ix = {"ang": 0, "hap": 1, "neu": 2, "sad": 3, START_TAG: 4, STOP_TAG: 5}

    label_val = []
    for dia_emos_tuple in val_data:
        label_val += dia_emos_tuple[1]
    for i in range(0, len(label_val), 1):
        if label_val[i] == 'ang':
            label_val[i] = 0
        elif label_val[i] == 'hap':
            label_val[i] = 1
        elif label_val[i] == 'neu':
            label_val[i] = 2
        elif label_val[i] == 'sad':
            label_val[i] = 3
        else:
            label_val[i] = -1
    
    model = CRF(len(utt_to_ix), emo_to_ix)
    params_dict = dict(model.named_parameters())
    params = []
    # UAR: 0.7099
    
    for key, value in params_dict.items():
        if key[:10] == 'weight_spk':
            if args.attention == 'logit':
                params += [{'params':value, 'weight_decay':5, 'lr':0.001}]
            
            elif args.attention == 'concat_representation':
                params += [{'params':value, 'weight_decay':1, 'lr':0.01}]
        else:
            params += [{'params':value, 'weight_decay':0.01, 'lr':0.01}]
    
    optimizer = optim.SGD(params, momentum=0.8)
    #optimizer = optim.Adam(params, amsgrad=False)
    #optimizer = optim.Adagrad(params)
    max_uar_val = 0
    best_epoch = -1
    sigmoid_fun = nn.Sigmoid()
    loss_list = []
    val_loss_list = []
    for epoch in range(10):
        if args.pretrain_version != 'original_output':
            random.shuffle(train_data)
        print('Epoch', epoch)
        for dialog, emos in train_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            dialog_tensor = prepare_dialog(dialog, utt_to_ix)
            targets = torch.tensor([emo_to_ix[t] for t in emos], dtype=torch.long)
    
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(dialog_tensor, targets, dialog)
            loss_list.append(loss.item())
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
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
        
        uar_val, acc_val, conf_val = utils.evaluate(predict_val, label_val)
        val_loss_list.append(val_loss_sum/len(val_data))

        #Save the best model so far
        if uar_val > max_uar_val:
            best_epoch = epoch
            max_uar_val = uar_val
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
            if args.pretrain_version == 'original_output':
                torch.save(checkpoint, './model/' + args.pretrain_version + '/' + args.dataset + '/Ses0' + str(args.model_num) + '.pth')
            else:
                torch.save(checkpoint, './model/' + args.pretrain_version + '/' + args.attention + '/' + args.dataset + '/Ses0' + str(args.model_num) + '.pth')
            
    print('The Best Epoch:', best_epoch)
    
    train_loss_list = []
    sum_loss = 0
    for i in range(0, len(loss_list), 1):
        if i != 0 and i % (len(train_data)) == 0:
            train_loss_list.append(sum_loss/len(train_data))
            sum_loss = 0
        sum_loss += loss_list[i]
    train_loss_list.append(sum_loss/len(train_data))
        
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, 's-', color = 'r', label="train_loss")
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, 's-', color = 'b', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = "best")
    plt.savefig('./Ses0' + str(args.model_num) + '.png')