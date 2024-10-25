export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:'pwd'
dim_list=(5)
T_list=(100)
m_list=(10000)
for dim in ${dim_list[@]};do
for T in ${T_list[@]};do
for m in ${m_list[@]};do
python main_train_ar.py \
    --dim $dim \
    --T $T \
    --m $m \ # number of samples
    --model lsa \
    --a 0.1 \
    --b 0.1 \
    --isotropic 0 \ # 0:full-one, 1:gaussian, 2:sparse vector
    --scale 1 \
    --lr # step size
    --save
done
done
done