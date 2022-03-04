import joblib

def split_dialog(dialogs):
  """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
     See eq (5) in the paper.
  Arg:
    dialogs: dict, for example, utterances of two speakers in dialog_01: 
            {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
  Return:
    spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
            {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
             dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
  """

  spk_dialogs = {}
  for dialog_id in dialogs.keys():
    spk_dialogs[dialog_id+'_M'] = []
    spk_dialogs[dialog_id+'_F'] = []
    for utt_id in dialogs[dialog_id]:
      if utt_id[-4] == 'M':
        spk_dialogs[dialog_id+'_M'].append(utt_id)
      elif utt_id[-4] == 'F':
        spk_dialogs[dialog_id+'_F'].append(utt_id)

  return spk_dialogs

if __name__ == '__main__':
    dialog_edit = joblib.load('./dialog_rearrange_4emo_iemocap.pkl')
    emo_dict = joblib.load('./emo_all_iemocap.pkl')
    spk_dialogs = split_dialog(dialog_edit)
    
    transit_num = 0
    total_transit = 0
    
    a_2_other = 0
    h_2_other = 0
    n_2_other = 0
    s_2_other = 0
    
    other_2_a = 0
    other_2_h = 0
    other_2_n = 0
    other_2_s = 0
    for dialog_id in spk_dialogs.values():
      if len(dialog_id) == 0:
        continue
      for entry in range(len(dialog_id) - 1):
        if emo_dict[dialog_id[entry]] != emo_dict[dialog_id[entry + 1]]:
          transit_num += 1
          if emo_dict[dialog_id[entry]] == 'ang':
              a_2_other += 1
          elif emo_dict[dialog_id[entry]] == 'hap':
              h_2_other += 1
          elif emo_dict[dialog_id[entry]] == 'neu':
              n_2_other += 1
          elif emo_dict[dialog_id[entry]] == 'sad':
              s_2_other += 1
              
          if emo_dict[dialog_id[entry+1]] == 'ang':
              other_2_a += 1
          elif emo_dict[dialog_id[entry+1]] == 'hap':
              other_2_h += 1
          elif emo_dict[dialog_id[entry+1]] == 'neu':
              other_2_n += 1
          elif emo_dict[dialog_id[entry+1]] == 'sad':
              other_2_s += 1
      #total_transit += (len(dialog_id) - 1)

    #bias = (transit_num + 1) / total_transit
    
    print('a to the other emotion:', round(a_2_other/(a_2_other+h_2_other+n_2_other+s_2_other)*100, 2), '%')
    print('h to the other emotion:', round(h_2_other/(a_2_other+h_2_other+n_2_other+s_2_other)*100, 2), '%')
    print('n to the other emotion:', round(n_2_other/(a_2_other+h_2_other+n_2_other+s_2_other)*100, 2), '%')
    print('s to the other emotion:', round(s_2_other/(a_2_other+h_2_other+n_2_other+s_2_other)*100, 2), '%')
    print('==========')
    print('the other emotion to a:', round(other_2_a/(other_2_a+other_2_h+other_2_n+other_2_s)*100, 2), '%')
    print('the other emotion to h:', round(other_2_h/(other_2_a+other_2_h+other_2_n+other_2_s)*100, 2), '%')
    print('the other emotion to n:', round(other_2_n/(other_2_a+other_2_h+other_2_n+other_2_s)*100, 2), '%')
    print('the other emotion to s:', round(other_2_s/(other_2_a+other_2_h+other_2_n+other_2_s)*100, 2), '%')
    
    a_shift = 0
    a_no_shift = 0
    h_shift = 0
    h_no_shift = 0
    n_shift = 0
    n_no_shift = 0
    s_shift = 0
    s_no_shift = 0
    
    emo_shift_cnt_dict = {}
    
    for dialog_id in spk_dialogs.values():
      if len(dialog_id) == 0:
          continue
      for entry in range(len(dialog_id) - 1):
          if emo_dict[dialog_id[entry]] == 'ang' and emo_dict[dialog_id[entry+1]] == 'ang':
              a_no_shift += 1
          elif emo_dict[dialog_id[entry]] == 'ang' and emo_dict[dialog_id[entry+1]] != 'ang':
              a_shift += 1
              if emo_shift_cnt_dict.get(emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]) == None:
                  emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] = 0
              emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] += 1
              
          if emo_dict[dialog_id[entry]] == 'hap' and emo_dict[dialog_id[entry+1]] == 'hap':
              h_no_shift += 1
          elif emo_dict[dialog_id[entry]] == 'hap' and emo_dict[dialog_id[entry+1]] != 'hap':
              h_shift += 1
              if emo_shift_cnt_dict.get(emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]) == None:
                  emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] = 0
              emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] += 1
              
          if emo_dict[dialog_id[entry]] == 'neu' and emo_dict[dialog_id[entry+1]] == 'neu':
              n_no_shift += 1
          elif emo_dict[dialog_id[entry]] == 'neu' and emo_dict[dialog_id[entry+1]] != 'neu':
              n_shift += 1
              if emo_shift_cnt_dict.get(emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]) == None:
                  emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] = 0
              emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] += 1
              
          if emo_dict[dialog_id[entry]] == 'sad' and emo_dict[dialog_id[entry+1]] == 'sad':
              s_no_shift += 1
          elif emo_dict[dialog_id[entry]] == 'sad' and emo_dict[dialog_id[entry+1]] != 'sad':
              s_shift += 1
              if emo_shift_cnt_dict.get(emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]) == None:
                  emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] = 0
              emo_shift_cnt_dict[emo_dict[dialog_id[entry]][0] + '2' + emo_dict[dialog_id[entry+1]][0]] += 1
    print('#####')
    print('shift from ang:', round(a_shift/(a_no_shift+a_shift)*100, 2), '%')
    print('shift from hap:', round(h_shift/(h_no_shift+h_shift)*100, 2), '%')
    print('shift from neu:', round(n_shift/(n_no_shift+n_shift)*100, 2), '%')
    print('shift from sad:', round(s_shift/(s_no_shift+s_shift)*100, 2), '%')
    print('#####')
    
    for key_trans_1 in emo_shift_cnt_dict:
        first_emo = key_trans_1[0]
        second_emo = key_trans_1[1]
        first_2_total_cnt = 0
        for key_trans_2 in emo_shift_cnt_dict:
            if key_trans_2[0] == first_emo:
                first_2_total_cnt  += emo_shift_cnt_dict[key_trans_2]
        print(key_trans_1, round(emo_shift_cnt_dict[key_trans_1]/first_2_total_cnt, 2))