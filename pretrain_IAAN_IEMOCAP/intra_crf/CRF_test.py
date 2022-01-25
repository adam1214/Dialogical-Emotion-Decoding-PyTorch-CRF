import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import utils
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

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
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size)*0.01) #6*6

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[emo_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, emo_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
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
        return pretrain_model_feats.to(device) # tensor: (utt數量) * (情緒數量+2)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.emo_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.emo_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
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
    parser.add_argument('-v', "--pretrain_version", type=str, help="which version of pretrain model you want to use?", default='dialog_rearrange_output')
    parser.add_argument("-d", "--dataset", type=str, help="which dataset to use? original or C2C or U2U", default = 'U2U')
    args = parser.parse_args()
    
    print(args)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    #EMBEDDING_DIM = 5

    out_dict = joblib.load('../data/'+ args.pretrain_version + '/outputs.pkl')
    #dialogs = joblib.load('../data/dialog_iemocap.pkl')
    #dialogs_edit = joblib.load('../data/dialog_4emo_iemocap.pkl')
    dialogs = joblib.load('../data/dialog_rearrange.pkl')
    dialogs_edit = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    
    if args.dataset == 'original':
        emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
        dias = dialogs_edit
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('../data/'+ args.pretrain_version + '/U2U_4emo_all_iemocap.pkl')
        dias = dialogs
    
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

    # Load model
    model_1 = CRF(len(utt_to_ix), emo_to_ix)
    model_1.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses01.pth')
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_1.eval()

    model_2 = CRF(len(utt_to_ix), emo_to_ix)
    model_2.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses02.pth')
    model_2.load_state_dict(checkpoint['model_state_dict'])
    model_2.eval()

    model_3 = CRF(len(utt_to_ix), emo_to_ix)
    model_3.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses03.pth')
    model_3.load_state_dict(checkpoint['model_state_dict'])
    model_3.eval()

    model_4 = CRF(len(utt_to_ix), emo_to_ix)
    model_4.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses04.pth')
    model_4.load_state_dict(checkpoint['model_state_dict'])
    model_4.eval()

    model_5 = CRF(len(utt_to_ix), emo_to_ix)
    model_5.to(device)
    checkpoint = torch.load('./model/' + args.pretrain_version + '/' + args.dataset + '/Ses05.pth')
    model_5.load_state_dict(checkpoint['model_state_dict'])
    model_5.eval()
    
    # inference
    predict = []
    pred_dict = {}
    with torch.no_grad():
        for i in range(0, len(test_data_Ses01), 1):
            precheck_dia = prepare_dialog(test_data_Ses01[i][0], utt_to_ix)
            tmp = model_1(precheck_dia)[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses01[i][0]):
                pred_dict[utt] = tmp[j]
        
        for i in range(0, len(test_data_Ses02), 1):
            precheck_dia = prepare_dialog(test_data_Ses02[i][0], utt_to_ix)
            tmp = model_2(precheck_dia)[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses02[i][0]):
                pred_dict[utt] = tmp[j]
            
        for i in range(0, len(test_data_Ses03), 1):
            precheck_dia = prepare_dialog(test_data_Ses03[i][0], utt_to_ix)
            tmp = model_3(precheck_dia)[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses03[i][0]):
                pred_dict[utt] = tmp[j]
            
        for i in range(0, len(test_data_Ses04), 1):
            precheck_dia = prepare_dialog(test_data_Ses04[i][0], utt_to_ix)
            tmp = model_4(precheck_dia)[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses04[i][0]):
                pred_dict[utt] = tmp[j]
        
        for i in range(0, len(test_data_Ses05), 1):
            precheck_dia = prepare_dialog(test_data_Ses05[i][0], utt_to_ix)
            tmp = model_5(precheck_dia)[1]
            predict += tmp
            for j, utt in enumerate(test_data_Ses05[i][0]):
                pred_dict[utt] = tmp[j]

    ori_emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
    label = []
    for dia_emos_tuple in test_data_Ses01:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses02:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses03:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses04:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])
    for dia_emos_tuple in test_data_Ses05:
        for utt in dia_emos_tuple[0]:
            label.append(ori_emo_dict[utt])

    for i in range(0, len(label), 1):
        if label[i] == 'ang':
            label[i] = 0
        elif label[i] == 'hap':
            label[i] = 1
        elif label[i] == 'neu':
            label[i] = 2
        elif label[i] == 'sad':
            label[i] = 3
        else:
            label[i] = -1
    
    uar, acc, conf = utils.evaluate(predict, label, final_test=1)
    print('UAR:', uar)
    print('ACC:', acc)
    print(conf)

    path = 'uar.txt'
    f = open(path, 'a')
    f.write(str(uar)+'\n')
    f.close()
    
    path = 'acc.txt'
    f = open(path, 'a')
    f.write(str(acc)+'\n')
    f.close()

    joblib.dump(pred_dict, './model/' + args.pretrain_version + '/' + args.dataset + '/preds_4.pkl')
    
    # ensure pretrained model performance
    labels = []
    predicts = []
    dialogs_edit = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    emo_dict = joblib.load('../data/emo_all_iemocap.pkl')
    emo2num = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            labels.append(emo2num[emo_dict[utt]])
            predicts.append(out_dict[utt].argmax())
            
    print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')