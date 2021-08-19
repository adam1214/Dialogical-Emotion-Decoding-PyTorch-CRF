## Result
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |67.1|65.3|||
| sequential_utt        |71.05|69.28|68.46|66.57|
| spk_info              |**73.21**|**71.67**|70.49|68.83|
| dual_crf              |71.15|69.70|68.36|66.46|
| weighted_dual_crf     |71.24|69.63|68.48|66.61|
| attention_dual_crf (logit)    |71.28|69.52|68.75|66.91|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    6 emotions (ang, hap, neu, sad, exc, fru)
        *    Training + Validation set:Session01, Session02, Session03, Session04(20 dialogs at the end as validation set)
        *    Testing set:Session05 
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|Utt to Utt Training Data weighted F1|
| -- | -- | -- | -- | -- | -- | -- |
| Pretrained Classifier|66.22|67.76|67.87||||
| sequential_utt|66.44+-.36|67.79+-.13|67.95+-.13|66.34+-.25|67.71+-.20|67.87+-.19|
| spk_info|66.44+-.67|67.86+-.57|68.04+-.59|**66.73+-.53**|**68.29+-.39**|**68.47+-.40**|
| dual_crf|66.54|67.76|67.92|66.31|68.00|68.07|
| weighted_dual_crf|||||||

--------------------------------------------------
*    Dataset:MELD
        *    7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |
| -- | -- | -- | -- |
| Pretrained Classifier|48.65|63.83|63.43|
| sequential_utt|48.57+-.24|63.72+-.09|63.35+-.11|
| spk_info|48.66+-.27|63.81+-.09|63.42+-.10|
| dual_crf|48.87|64.02|63.64|
| weighted_dual_crf||||