import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils

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

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions_inter.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions_inter.data[:, emo_to_ix[STOP_TAG]] = -10000

        self.transitions_intra.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions_intra.data[:, emo_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats, dialog):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.emo_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the dialog
        i = 0
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
                    pre_spk = dialog[i-1][-4]
                    cur_spk = dialog[i][-4]
                elif args.dataset == 'meld':
                    pre_spk = dialog[i-1][:-4]
                    cur_spk = dialog[i][:-4]
                
                if i > 0 and pre_spk != cur_spk:
                    trans_score = self.transitions_inter[next_tag].view(1, -1)
                else:
                    trans_score = self.transitions_intra[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            i += 1
        terminal_var = forward_var + self.transitions_intra[self.emo_to_ix[STOP_TAG]]
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
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        emos = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), emos])
        for i, feat in enumerate(feats):
            if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
                pre_spk = dialog[i-1][-4]
                cur_spk = dialog[i][-4]
            elif args.dataset == 'meld':
                pre_spk = dialog[i-1][:-4]
                cur_spk = dialog[i][:-4]
            if i > 0 and pre_spk != cur_spk:
                score = score + self.transitions_inter[emos[i + 1], emos[i]] + feat[emos[i + 1]]
            else:
                score = score + self.transitions_intra[emos[i + 1], emos[i]] + feat[emos[i + 1]]
        score = score + self.transitions_intra[self.emo_to_ix[STOP_TAG], emos[-1]]
        return score

    def _viterbi_decode(self, feats, dialog):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.emo_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        i = 0
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
                    pre_spk = dialog[i-1][-4]
                    cur_spk = dialog[i][-4]
                elif args.dataset == 'meld':
                    pre_spk = dialog[i-1][:-4]
                    cur_spk = dialog[i][:-4]
                    
                if i > 0 and pre_spk != cur_spk:
                    next_tag_var = forward_var + self.transitions_inter[next_tag]
                else:
                    next_tag_var = forward_var + self.transitions_intra[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            i += 1
            
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions_intra[self.emo_to_ix[STOP_TAG]]
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
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use?\niemocap_original\niemocap_U2U\nmeld", default = 'meld')
    parser.add_argument("-s", "--seed", type=int, help="select torch seed", default = 1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
        out_dict = joblib.load('../data/iemocap/outputs.pkl')
        dialogs = joblib.load('../data/iemocap/dialog_iemocap.pkl')
        dialogs_edit = joblib.load('../data/iemocap/dialog_6emo_iemocap.pkl')
    elif args.dataset == 'meld':
        out_dict = joblib.load('../data/meld/outputs.pkl')
        dialogs = joblib.load('../data/meld/dialog_meld.pkl')
    
    if args.dataset == 'iemocap_original':
        emo_dict = joblib.load('../data/iemocap/emo_all_iemocap.pkl')
        dias = dialogs_edit
    elif args.dataset == 'iemocap_U2U':
        emo_dict = joblib.load('../data/iemocap/U2U_6emo_all_iemocap.pkl')
        dias = dialogs
    elif args.dataset == 'meld':
        emo_dict = joblib.load('../data/meld/emo_all_meld.pkl')
        dias = dialogs
        
    # Make up training data & testing data
    train_data = []
    val_data = []
    test_data = []
    if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
        for dialog in dias:
            if dialog[4] == '5':
                test_data.append((dias[dialog],[]))
                for utt in test_data[-1][0]:
                    test_data[-1][1].append(emo_dict[utt])
            else:
                train_data.append((dias[dialog],[]))
                for utt in train_data[-1][0]:
                    train_data[-1][1].append(emo_dict[utt])
        val_data = train_data[100:]
        del train_data[100:]
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
    for dialog, emos in test_data:
        for utt in dialog:
            if utt not in utt_to_ix:
                utt_to_ix[utt] = len(utt_to_ix)
    for dialog, emos in train_data:
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

    if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
        emo_to_ix = {'exc':0, 'neu':1, 'fru':2, 'sad':3, 'hap':4, 'ang':5, START_TAG: 6, STOP_TAG: 7}
    elif args.dataset == 'meld':
        emo_to_ix = {'neutral':0, 'surprise':1, 'fear':2, 'sadness':3, 'joy':4, 'disgust':5, 'anger':6, START_TAG: 7, STOP_TAG: 8}

    # Load model
    model = CRF(len(utt_to_ix), emo_to_ix)
    if args.dataset == 'iemocap_original':
        checkpoint = torch.load('./model/iemocap/original/model' + str(args.seed) + '.pth')
    elif args.dataset == 'iemocap_U2U':
        checkpoint = torch.load('./model/iemocap/U2U/model' + str(args.seed) + '.pth')
    elif args.dataset == 'meld':
        checkpoint = torch.load('./model/meld/model' + str(args.seed) + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # inference
    predict = []
    with torch.no_grad():
        for i in range(0, len(test_data), 1):
            precheck_dia = prepare_dialog(test_data[i][0], utt_to_ix)
            predict += model(precheck_dia, test_data[i][0])[1]

    if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
        ori_emo_dict = joblib.load('../data/iemocap/emo_all_iemocap.pkl')
    elif args.dataset == 'meld':
        ori_emo_dict = joblib.load('../data/meld/emo_all_meld.pkl')

    label = []
    if args.dataset == 'iemocap_original' or args.dataset == 'iemocap_U2U':
        for dia_key in dias:
            if dia_key[4] == '5':
                for utt in dias[dia_key]:
                    label_emo = ori_emo_dict[utt]
                    if label_emo == '---':
                        label_emo = -1
                    else:
                        label_emo = emo_to_ix[label_emo]
                    label.append(label_emo)
    elif args.dataset == 'meld':
        for dia_key in dias:
            if dia_key.split('_')[0] == 'test':
                for utt in dias[dia_key]:
                    label_emo = ori_emo_dict[utt]
                    label_emo = emo_to_ix[label_emo]
                    label.append(label_emo)
    
    uar, acc, f1, conf = utils.evaluate(predict, label, final_test=1)
    print('UAR:', uar)
    print('ACC:', acc)
    print('Weighted F1', f1)
    print(conf)
    
    path = 'uar.txt'
    f = open(path, 'a')
    f.write(str(uar)+'\n')
    f.close()
    
    path = 'acc.txt'
    f = open(path, 'a')
    f.write(str(acc)+'\n')
    f.close()
    
    path = 'f1.txt'
    f = open(path, 'a')
    f.write(str(f1)+'\n')
    f.close()
    