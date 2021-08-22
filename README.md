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
| attention_dual_crf (logit)    |71.14|69.63|68.56|66.73|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN (reproduce experimental results to get concat_representation)

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |65.02|64.17|||
| sequential_utt        |68.22|67.02|67.20|65.97|
| spk_info              |69.27|**68.40**|68.52|67.69|
| dual_crf              |68.01|67.02|66.97|65.72|
| weighted_dual_crf     |68.67|67.58|67.73|66.44|
| attention_dual_crf (logit)    |**69.64**|68.25|67.10|65.68|
| attention_dual_crf (concat_representation)    |68.95|67.56|66.40|64.80|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN variants (BiGRU+ATT)
        *    A BiGRU network with the classical attention (ATT) trained using current utterances only.

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |59.80|58.45|||
| sequential_utt        |65.82|64.62|62.91|61.45|
| spk_info              |**68.69**|**67.69**|64.54|63.26|
| dual_crf              |65.50|64.60|63.81|62.52|
| weighted_dual_crf     |65.80|64.49|63.64|62.36|
| attention_dual_crf (logit)    |64.02|62.74|63.52|62.29|
| attention_dual_crf (concat_representation)    |64.61|63.26|62.86|61.51|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:BERT_IAAN
        *    word embeding by BERT base (uncased)
                *    summation over the last 4 layers
                *    dimension of each word: (1, 768)
                *    dimension of each utterance: (n, 768)
                        *    n: number of words in the utterance

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |76.01|75.21|||
| sequential_utt        |78.36|77.80|77.22|76.50|
| spk_info              |**81.62**|**81.07**|79.76|78.94|
| dual_crf              |78.12|77.51|76.83|76.17|
| weighted_dual_crf     |78.16|77.73|76.87|76.19|
| attention_dual_crf (logit)    |78.10|77.65|77.09|76.51|
| attention_dual_crf (concat_representation)    |77.48|77.00|76.78|76.15|

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
| weighted_dual_crf|66.20+-.76|67.50+-.64|67.66+-.60|66.11+-.59|67.84+-.34|67.95+-.33|

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
| weighted_dual_crf|48.23|63.83|63.45|