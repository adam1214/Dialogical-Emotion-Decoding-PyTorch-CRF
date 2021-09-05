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
| attention_dual_crf (logit)    |71.15|69.59|68.53|66.68|

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
| spk_info              |69.11+-.28|**68.24+-.25**|68.52|67.69|
| dual_crf              |68.01|67.02|66.97|65.72|
| weighted_dual_crf     |68.67|67.58|67.73|66.44|
| attention_dual_crf (logit)    |68.94+-.92|67.62+-.87|66.79|65.38|
| attention_dual_crf (concat_representation)    |**69.53+-.70**|68.21+-.69|66.62|62.14|

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
| spk_info              |**69.07+-.21**|**68.07+-.21**|64.54|63.26|
| dual_crf              |65.50|64.60|63.81|62.52|
| weighted_dual_crf     |65.80|64.49|63.64|62.36|
| attention_dual_crf (logit)    |66.97+-1.23|65.79+-1.16|64.66|63.21|
| attention_dual_crf (concat_representation)    |67.34+-.70|66.30+-.77|64.95|62.59|

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
| attention_dual_crf (logit)    |78.42|77.51|76.69|75.88|
| attention_dual_crf (concat_representation)    |78.84|78.05|77.45|76.48|

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
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    Training + Validation set:Session01, Session02, Session03, Session04(20 dialogs at the end as validation set)
        *    Testing set:Session05 
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|Utt to Utt Training Data weighted F1|
| -- | -- | -- | -- | -- | -- | -- |
| Pretrained Classifier|83.11|82.27|82.39||||
| sequential_utt|82.98|82.19|82.28||||

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    6 emotions (ang, hap, neu, sad, exc, fru)
        *    5 fold, 5 model
                * fold 5: the last 20 dialogs in session 4 are val set
                * fold 4: the last 20 dialogs in session 5 are val set
                * fold 3: the last 20 dialogs in session 1 are val set
                * fold 2: the last 20 dialogs in session 3 are val set
                * fold 1: the last 20 dialogs in session 2 are val set
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|Utt to Utt Training Data weighted F1|
| -- | -- | -- | -- | -- | -- | -- |
| Pretrained Classifier|77.23|77.25|77.24||||
| sequential_utt|76.75|76.80|76.71||||

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