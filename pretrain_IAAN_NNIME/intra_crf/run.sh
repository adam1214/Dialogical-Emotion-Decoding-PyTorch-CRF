for model_num in 1 2 3 4 5; 
do
    python CRF_train.py -v dialog_rearrange_output -d original -n $model_num
done
echo "original"
python CRF_test.py -v dialog_rearrange_output -d original

for model_num in 1 2 3 4 5; 
do
    python CRF_train.py -v dialog_rearrange_output -d U2U -n $model_num
done
echo "U2U"
python CRF_test.py -v dialog_rearrange_output -d U2U