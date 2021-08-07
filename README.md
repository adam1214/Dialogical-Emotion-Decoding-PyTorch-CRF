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
| dual_crf              |71.44|69.99|68.49|66.61|
| weighted_dual_crf     |71.49|70.24|68.54|66.77|

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
| dual_crf|65.94+-.45|67.40+-.39|67.58+-.40|66.25+-.24|67.79+-.27|67.94+-.26|
| weighted_dual_crf|66.11+-.61|67.60+-.46|67.74+-.44|66.10+-.60|67.77+-.35|67.90+-.35|

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
| dual_crf|48.69+-.46|63.80+-.19|63.41+-.18|
| weighted_dual_crf|48.60+-.31|63.74+-.17|63.35+-.15|