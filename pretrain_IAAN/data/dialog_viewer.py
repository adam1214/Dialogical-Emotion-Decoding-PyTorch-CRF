import joblib
dialog = joblib.load('./dialog_iemocap.pkl')

s1_cnt=0
s2_cnt=0
s3_cnt=0
s4_cnt=0
s5_cnt=0

for dialog_name in dialog:
    if dialog_name[4] == '1':
        s1_cnt += 1
    elif dialog_name[4] == '2':
        s2_cnt += 1
    elif dialog_name[4] == '3':
        s3_cnt += 1
    elif dialog_name[4] == '4':
        s4_cnt += 1
    elif dialog_name[4] == '5':
        s5_cnt += 1