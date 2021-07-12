### Tuning model (SGD algo.)
|                       | Original Training Data UAR | Original Training Data ACC | Class to Class Training Data UAR | Class to Class Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------- | -------------------------------- | -------------------------------- | --- | --- |
| sequential_utt        |0.7105|0.6928|0.5758|0.5966|0.6846|0.6657|
| spk_info              |**0.7321**|**0.7167**|0.6029|0.6266|0.7049|0.6883|
| dual_crf              |0.7144|0.6999|0.5986|0.6172|0.6849|0.6661|