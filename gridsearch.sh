
for Lamda in 0.01 0.05
do
#for TRY in {1..100}!!
#do
for DROPOUT in 0.1 0.2
do
      for TT in 2 3
      do
          for Unmask in 0.5 0.3
          do
              for B in 64 128 256
              do
                  for Layer in 4  6
                  do
                     for Head in 2 4 6 8
                     do
                        python train.py \
                        -data_pkl ./data/pre_data.pkl \
                        -n_layers $Layer \
                        -n_head $Head \
                        -b $B \
                        -unmask $Unmask \
                        -T $TT \
                        -dropout $DROPOUT \
                        -epoch 60 \
                        -use_tb \
                        -d_k 38 \
                        -d_v 38 \
                        -warmup 128000 \
                        -lr_mul 200 \
                        -seed 10 \
                        -lambda_con $Lamda\
                        -isRandMask \
                        -isContrastLoss \
                        -save_mode best \
                        -output_dir output
                      done
                  done
              done
         done
      done
done
done
