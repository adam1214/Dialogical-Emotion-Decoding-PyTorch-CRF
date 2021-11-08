## Result
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |67.1|65.3|||
| sequential_crf       |71.05|69.28|68.46|66.57|
| intra_crf              |**73.21**|**71.67**|70.49|68.83|
| dual_crf              |71.58|70.11|69.09|67.18|
| weighted_dual_crf     |71.69|70.22|68.77|66.97|
| attention_dual_crf (logit)    |71.23|69.90|68.64|66.79|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN (reproduce experimental results to get concat_representation)

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |65.02|64.17|||
| sequential_crf        |68.22|67.02|67.20|65.97|
| intra_crf              |69.11+-.28|68.24+-.25|68.52|67.69|
| dual_crf              |69.38|68.72|67.17|66.01|
| weighted_dual_crf     |68.75|68.02|68.01|66.77|
| attention_dual_crf (logit)    |69.12|68.32|66.10|64.69|
| attention_dual_crf (concat_representation)    |**70.56**|**69.01**|66.49|64.89|

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
| sequential_crf        |65.82|64.62|62.91|61.45|
| intra_crf              |**69.07+-.21**|**68.07+-.21**|64.54|63.26|
| dual_crf              |65.97|65.16|64.45|63.23|
| weighted_dual_crf     |66.62|65.56|63.93|62.65|
| attention_dual_crf (logit)    |67.14|67.15|64.49|62.57|
| attention_dual_crf (concat_representation)   |67.10|66.26|64.08|62.30|

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
| sequential_crf        |78.36|77.80|77.22|76.50|
| intra_crf              |**81.62**|**81.07**|79.76|78.94|
| dual_crf              |78.50|77.87|76.88|76.28|
| weighted_dual_crf     |78.83|78.23|76.84|76.21|
| attention_dual_crf (logit)    |78.67|77.98|76.61|75.79|
| attention_dual_crf (concat_representation)    |78.55|77.67|77.11|76.15|

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
| sequential_crf|66.44+-.36|67.79+-.13|67.95+-.13|66.34+-.25|67.71+-.20|67.87+-.19|
| intra_crf|66.44+-.67|67.86+-.57|68.04+-.59|66.73+-.53|68.29+-.39|68.47+-.40|
| dual_crf|65.72|67.39|67.50|66.14|68.06|68.13|
| weighted_dual_crf|66.83|67.88|68.03|64.88|67.02|67.09|

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
| sequential_crf|82.98|82.19|82.28||||

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
| sequential_crf|76.75|76.80|76.71||||

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    4 emotions (ang, hap, neu, sad)
        *    5 fold, 5 model
                * fold 5: the last 20 dialogs in session 4 are val set
                * fold 4: the last 20 dialogs in session 5 are val set
                * fold 3: the last 20 dialogs in session 1 are val set
                * fold 2: the last 20 dialogs in session 3 are val set
                * fold 1: the last 20 dialogs in session 2 are val set
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| -- | -- | -- | -- | -- |
| Pretrained Classifier|68.46|66.82|||||
| intra_crf|67.61|66.19|||||

--------------------------------------------------
*    Dataset:MELD
        *    7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |
| -- | -- | -- | -- |
| Pretrained Classifier|48.65|63.83|63.43|
| sequential_crf|48.57+-.24|63.72+-.09|63.35+-.11|
| intra_crf|48.66+-.27|63.81+-.09|63.42+-.10|
| dual_crf|48.67|64.10|63.70|
| weighted_dual_crf|48.57|64.02|63.64|