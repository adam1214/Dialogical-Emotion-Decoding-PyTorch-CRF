import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils

torch.manual_seed(1)

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
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)) #6*6

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, emo_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.emo_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
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
        return pretrain_model_feats # tensor: (utt數量) * (情緒數量+2)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.emo_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.emo_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
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

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_pretrain_model_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above, this for model predicting
        # Get the emission scores from the BiLSTM
        pretrain_model_feats = self._get_pretrain_model_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(pretrain_model_feats)
        return score, tag_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-n", "--model_num", type=int, help="which model number you want to train?", default=1)
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'original')
    args = parser.parse_args()
    
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    out_dict = joblib.load('../data/outputs.pkl')
    dialogs = joblib.load('../data/dialog_iemocap.pkl')
    dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
        dias = dialogs_edit
    elif args.dataset == 'C2C':
        emo_dict = joblib.load('../data/C2C_4emo_all_iemocap.pkl')
        dias = dialogs
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/U2U_4emo_all_iemocap.pkl')
        dias = dialogs
        
    # Make up training data & testing data
    model_num_val_map = {1:'5', 2:'4', 3:'2', 4:'1', 5: '3'}
    train_data = []
    val_data = []
    for dialog in dias:
        if dialog[4] != str(args.model_num) and dialog[4] != model_num_val_map[args.model_num]: # assign to train set
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
        elif dialog[4] == model_num_val_map[args.model_num]: # assign to val set
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

    test_data_Ses01 = []
    test_data_Ses02 = []
    test_data_Ses03 = []
    test_data_Ses04 = []
    test_data_Ses05 = []
    for dialog in dias:
        if dialog[4] == '1':
            test_data_Ses01.append(([],[]))
            test_data_Ses01.append(([],[]))
            for utt in dias[dialog]:
                if utt[-4] == 'M':
                    test_data_Ses01[-2][0].append(utt)
                    test_data_Ses01[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
                    test_data_Ses01[-1][0].append(utt)
                    test_data_Ses01[-1][1].append(emo_dict[utt])
            if len(test_data_Ses01[-2][0]) == 0:
                del test_data_Ses01[-2]
            if len(test_data_Ses01[-1][0]) == 0:
                del test_data_Ses01[-1]
        elif dialog[4] == '2':
            test_data_Ses02.append(([],[]))
            test_data_Ses02.append(([],[]))
            for utt in dias[dialog]:
                if utt[-4] == 'M':
                    test_data_Ses02[-2][0].append(utt)
                    test_data_Ses02[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
                    test_data_Ses02[-1][0].append(utt)
                    test_data_Ses02[-1][1].append(emo_dict[utt])
            if len(test_data_Ses02[-2][0]) == 0:
                del test_data_Ses02[-2]
            if len(test_data_Ses02[-1][0]) == 0:
                del test_data_Ses02[-1]
        elif dialog[4] == '3':
            test_data_Ses03.append(([],[]))
            test_data_Ses03.append(([],[]))
            for utt in dias[dialog]:
                if utt[-4] == 'M':
                    test_data_Ses03[-2][0].append(utt)
                    test_data_Ses03[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
                    test_data_Ses03[-1][0].append(utt)
                    test_data_Ses03[-1][1].append(emo_dict[utt])
            if len(test_data_Ses03[-2][0]) == 0:
                del test_data_Ses03[-2]
            if len(test_data_Ses03[-1][0]) == 0:
                del test_data_Ses03[-1]
        elif dialog[4] == '4':
            test_data_Ses04.append(([],[]))
            test_data_Ses04.append(([],[]))
            for utt in dias[dialog]:
                if utt[-4] == 'M':
                    test_data_Ses04[-2][0].append(utt)
                    test_data_Ses04[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
                    test_data_Ses04[-1][0].append(utt)
                    test_data_Ses04[-1][1].append(emo_dict[utt])
            if len(test_data_Ses04[-2][0]) == 0:
                del test_data_Ses04[-2]
            if len(test_data_Ses04[-1][0]) == 0:
                del test_data_Ses04[-1]
        elif dialog[4] == '5':
            test_data_Ses05.append(([],[]))
            test_data_Ses05.append(([],[]))
            for utt in dias[dialog]:
                if utt[-4] == 'M':
                    test_data_Ses05[-2][0].append(utt)
                    test_data_Ses05[-2][1].append(emo_dict[utt])
                elif utt[-4] == 'F':
                    test_data_Ses05[-1][0].append(utt)
                    test_data_Ses05[-1][1].append(emo_dict[utt])
            if len(test_data_Ses05[-2][0]) == 0:
                del test_data_Ses05[-2]
            if len(test_data_Ses05[-1][0]) == 0:
                del test_data_Ses05[-1]

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
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2, momentum=0.5)

    max_uar_val = 0
    best_epoch = -1
    for epoch in range(10):
        print('Epoch', epoch)
        for dialog, emos in train_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            dialog_in = prepare_dialog(dialog, utt_to_ix)
            targets = torch.tensor([emo_to_ix[t] for t in emos], dtype=torch.long)
    
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(dialog_in, targets)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

        #check model performance on predefined validation set
        predict_val = []
        with torch.no_grad():
            for i in range(0, len(val_data), 1):
                precheck_dia = prepare_dialog(val_data[i][0], utt_to_ix)
                predict_val += model(precheck_dia)[1]
        uar_val, acc_val, conf_val = utils.evaluate(predict_val, label_val)

        #Save the best model so far
        if uar_val > max_uar_val:
            best_epoch = epoch
            max_uar_val = uar_val
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
            torch.save(checkpoint, './model/' + args.dataset + '/Ses0' + str(args.model_num) + '.pth')
            
    print('The Best Epoch:', best_epoch)
    # Save
    #torch.save(model.state_dict(), './model/' + args.dataset + '/Ses0' + str(args.model_num) + '.pth')