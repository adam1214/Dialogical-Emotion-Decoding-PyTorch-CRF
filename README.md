## Result
*    Dataset:IEMOCAP 
        *    5 fold, 5 model
        *    Every model with one session as validation set
*    Pre-trained classifier:IAAN

|| Original Training Data UAR | Original Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------------- | --- | --- |
| Pretrained Classifier |67.1|65.3|||
| sequential_utt        |71.05|69.28|68.46|66.57|
| spk_info              |**73.21**|**71.67**|70.49|68.83|
| dual_crf              |71.44|69.99|68.49|66.61|
| weighted_dual_crf     |71.54|70.10|68.78|66.91|

--------------------------------------------------
*    Dataset:IEMOCAP 
        *    Training + Validation set:Session01, Session02, Session03, Session04(20 dialogs at the end as validation set)
        *    Testing set:Session05 
        *    10 run: avg+-sample_std
*    Pre-trained classifier:DAG-ERC

|| Original Training Data UAR | Original Training Data ACC | Original Training Data weighted F1 |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|Utt to Utt Training Data weighted F1|
| -- | -- | -- | -- | -- | -- | -- |
| Pretrained Classifier|66.22|67.76|67.87||||
| sequential_utt|66.66+-.44|67.76+-.36|67.91+-.37|66.63+-.25|67.73+-.23|67.89+-.24|
| spk_info|66.21+-.57|67.76+-.58|67.88+-.54|66.56+-.25|68.16+-.16|68.34+-.15|
| dual_crf|66.00+-.45|67.19+-.39|67.36+-.38|66.31+-.30|67.63+-.30|67.78+-.30|
| weighted_dual_crf|66.05+-.45|67.53+-.40|67.69+-.40|66.01+-.22|67.47+-.18|67.64+-.18|