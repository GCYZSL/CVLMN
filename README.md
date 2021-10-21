# CVLNM/ pytorch 0.4.0/ testing
I provide the anaconda environment for running my code in https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing. You should download the file ''environment_yx1.yml'' from this link and set up the environment as follows.
1.Download the anaconda from the website https://www.anaconda.com/ and install it.
2.Go to website https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment to learn how to 'creating an environment from an environment.yml file'.
```
conda env create -f environment_yx1.yml
```
3.After installing anaconda and setting up the environment, run the following code to get into the environment.
```
source activate yx1
```
If you want to exit from this environment, you can run the following code to exit.
```
source deactivate
```
# Download Bottom-up features.
Download pre-extracted feature from https://github.com/peteanderson80/bottom-up-attention. You can either download adaptive one or fixed one. We use the ''10 to 100 features per image (adaptive)'' in our experiments.
For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```
Then :
```
python script/make_bu_data.py --output_dir data/cocobu
```
This will create data/cocobu_fc, data/cocobu_att and data/cocobu_box. 
# Training the model
1.After downloading the codes and meta data, you can train the model by using the following code:
```
python train.py --id c1  --checkpoint_path c1 --caption_model mcap_rs3_mem_new  --mtopdown_num 1 --mtopdown_res 1 --topdown_res 1 --input_json data/cocobu.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_attr_dir data/cocobu_att --input_rela_dir data/cocobu_att --input_label_h5 data/cocobu_label.h5 --batch_size 50 --accumulate_number 2 --learning_rate_decay_start 0 --learning_rate 5e-4 --learning_rate_decay_every 5 --scheduled_sampling_start 37 --save_checkpoint_every 5000 --val_images_use 50 --max_epochs 100 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 37 --train_split train --gpu 0 --combine_att concat --cont_ver 1 --relu_mod leaky_relu --memory_cell_path kg/kg.npz
```
Note that due to the limited GPU memory, we accumulate a few batch to approximate a bigger batch size, e.g., if --accumulate_number is 2 and --batch_size is 50, then the used batch size is 50 \* 2=100. However, the performance of such approximation is weaker than bigger batch size, e.g., --accumulate_number is 1 and --batch_size is 100.

# Evaluating the model
1.After training the model or downloading the well-trained model, you can evaluate them by using the following code:
```
python eval_rs.py --dump_images 0 --num_images 5000 --model c1/modelc10001.pth --infos_path c1/infos_c10001.pkl --language_eval 1 --beam_size 5 --split test --index_eval 1 --gpu 1 --batch_size 100 --memory_cell_path c1/memory_cellrc10001.npz
```
what you need to do is to switch the model id with your id, like c10001 to c10023.

