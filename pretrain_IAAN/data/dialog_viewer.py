import joblib
dialog = joblib.load('./dialog_4emo_iemocap.pkl')
emo_dict = joblib.load('./emo_all_iemocap.pkl')

'''
# 分析自身情緒轉換 (Ex:hap有多少機率轉換到自己hap)
a2a = 0
a2h = 0
a2n = 0
a2s = 0

h2a = 0
h2h = 0
h2n = 0
h2s = 0

n2a = 0
n2h = 0
n2n = 0
n2s = 0

s2a = 0
s2h = 0
s2n = 0
s2s = 0

# for spk M
for dialog_name in dialog:
    pre_emo = ''
    for utt in dialog[dialog_name]:
        if utt[-4] == 'M':
            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                a2a += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'hap':
                a2h += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'neu':
                a2n += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'sad':
                a2s += 1
                
            elif pre_emo == 'hap' and emo_dict[utt] == 'ang':
                h2a += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'hap':
                h2h += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'neu':
                h2n += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'sad':
                h2s += 1
            
            elif pre_emo == 'neu' and emo_dict[utt] == 'ang':
                n2a += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'hap':
                n2h += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'neu':
                n2n += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'sad':
                n2s += 1
            
            elif pre_emo == 'sad' and emo_dict[utt] == 'ang':
                s2a += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'hap':
                s2h += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'neu':
                s2n += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'sad':
                s2s += 1
            
            pre_emo = emo_dict[utt]

# for spk F
for dialog_name in dialog:
    pre_emo = ''
    for utt in dialog[dialog_name]:
        if utt[-4] == 'F':
            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                a2a += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'hap':
                a2h += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'neu':
                a2n += 1
            elif pre_emo == 'ang' and emo_dict[utt] == 'sad':
                a2s += 1
                
            elif pre_emo == 'hap' and emo_dict[utt] == 'ang':
                h2a += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'hap':
                h2h += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'neu':
                h2n += 1
            elif pre_emo == 'hap' and emo_dict[utt] == 'sad':
                h2s += 1
            
            elif pre_emo == 'neu' and emo_dict[utt] == 'ang':
                n2a += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'hap':
                n2h += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'neu':
                n2n += 1
            elif pre_emo == 'neu' and emo_dict[utt] == 'sad':
                n2s += 1
            
            elif pre_emo == 'sad' and emo_dict[utt] == 'ang':
                s2a += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'hap':
                s2h += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'neu':
                s2n += 1
            elif pre_emo == 'sad' and emo_dict[utt] == 'sad':
                s2s += 1
            
            pre_emo = emo_dict[utt]

from_a = a2a + a2h + a2n + a2s
from_h = h2a + h2h + h2n + h2s
from_n = n2a + n2h + n2n + n2s
from_s = s2a + s2h + s2n + s2s

print('a2a:', round(a2a*100/from_a, 2), '%')
print('a2h:', round(a2h*100/from_a, 2), '%')
print('a2n:', round(a2n*100/from_a, 2), '%')
print('a2s:', round(a2s*100/from_a, 2), '%')

print('h2a:', round(h2a*100/from_h, 2), '%')
print('h2h:', round(h2h*100/from_h, 2), '%')
print('h2n:', round(h2n*100/from_h, 2), '%')
print('h2s:', round(h2s*100/from_h, 2), '%')

print('n2a:', round(n2a*100/from_n, 2), '%')
print('n2h:', round(n2h*100/from_n, 2), '%')
print('n2n:', round(n2n*100/from_n, 2), '%')
print('n2s:', round(n2s*100/from_n, 2), '%')

print('s2a:', round(s2a*100/from_s, 2), '%')
print('s2h:', round(s2h*100/from_s, 2), '%')
print('s2n:', round(s2n*100/from_s, 2), '%')
print('s2s:', round(s2s*100/from_s, 2), '%')
'''

# 受別人影響的自身情緒轉換 
# Ex:hap有多少機率轉換到自己ang的情況下，是受到previous utt(對方)也是ang的影響, hap(spk1) -> … -> ang(spk2) -> ang(spk1)

event_dict = {}
def event_counter(spk):
    for dialog_name in dialog:
        pre_emo_self = ''
        for i in range(0, len(dialog[dialog_name]), 1):
            if dialog[dialog_name][i][-4] == spk:
                if i > 0 and dialog[dialog_name][i-1][-4] != dialog[dialog_name][i][-4]:
                    pre_emo_opp = emo_dict[dialog[dialog_name][i-1]][0]
                    cur_emo_self = emo_dict[dialog[dialog_name][i]][0]
                    KEY = pre_emo_self+'2'+cur_emo_self+'_'+pre_emo_opp
                    if KEY[0] != '2':
                        if  KEY not in event_dict:
                            event_dict[KEY] = 1
                        else:
                            event_dict[KEY] += 1
                pre_emo_self = emo_dict[dialog[dialog_name][i]][0]
event_counter('M')
event_counter('F')
event_prob_dict = {}
emo_list = ['a', 'h', 'n', 's']

for e_pre_self in emo_list:
    for e_cur_self in emo_list:
        sum_4_emo = 0
        for e_pre_opp_1 in emo_list:
            KEY = str(e_pre_self+'2'+e_cur_self+'_'+e_pre_opp_1)
            if KEY not in event_dict:
                event_dict[KEY] = 0
            sum_4_emo += event_dict[KEY]
        for e_pre_opp_2 in emo_list:
            KEY = str(e_pre_self+'2'+e_cur_self+'_'+e_pre_opp_2)
            event_prob_dict[KEY] = event_dict[KEY]*100/sum_4_emo

for k in event_prob_dict:
    print(k, ':', round(event_prob_dict[k], 2), '%')
