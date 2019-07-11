# cox 1-8
#set -v
#for i in {1..8}
#do
#    echo "#####################block_num=$i begin.#####################"
#    python3 train.py -d survival -s RESULTS/survdl${i}_cv10_drop05 -cv 10 -lr 0.00001 -smi 1 --block_num $i --no_atten --dropout_rate 0.5
#done

# svm 1-8 rank_ratio=1 
#set -v
#for i in {1..8}
#do
#    echo "#####################block_num=$i begin.#####################"
#    python3 train.py -d survival -s RESULTS/survdl${i}_cv10_drop05_svm -cv 10 -lr 0.00001 -smi 1 --block_num $i --no_atten --dropout_rate 0.5 --loss_type svm
#done

# svmloss rank_ratio=0.9-0.0
set -v
for f in {0..9}
do
    rr=`echo "scale=3;$f / 10" |bc`
    echo "#####################rank_ratio=${rr} begin.#####################"
    python3 train.py -s RESULTS/zoom20_rr$f -e 40 --zoom 20.0 --rank_ratio $rr
done


# svm 1-8 rank_ratio=0.
#set -v
#for i in {1..8}
#do
#    echo "#####################block_num=$i begin.#####################"
#    python3 train.py -d survival -s RESULTS/survdl${i}_cv10_drop05_svm_rr0 -cv 10 -lr 0.001 -smi 1 --block_num $i --no_atten --dropout_rate 0.5 --loss_type svm --svm_rankratio 0.
#done

# cox 1-8 attention
#set -v
#for i in {1..8}
#do
#    echo "#####################block_num=$i begin.#####################"
#    python3 train.py -d survival -s RESULTS/survdl${i}_atten_cv10_drop05 -cv 10 -lr 0.0001 -smi 1 --block_num $i --dropout_rate 0.5
#done



# cox 1-8 dropout_rate:0.5-0.9 lr:1e-4
#set -v
#for f in {5..9}
#do
#    rr=`echo "scale=3;$f / 10" |bc`
#    for i in {1..8}
#    do
#        echo "#####################dropout_ratio=${rr}, block_num=$i begin.#####################"
#        python3 train.py -d survival -s RESULTS/survdl${i}_cv10_drop0${f}_cox -cv 10 -lr 0.0001 -smi 1 --block_num $i --no_atten --dropout_rate $rr
#    done
#done



# cox 1-8 dropout_rate:0.5-0.9 lr:1e-4 explr:0.95
#set -v
#for f in {5..9}
#do
#    rr=`echo "scale=3;$f / 10" |bc`
#    for i in {1..8}
#    do
#        echo "#####################dropout_ratio=${rr}, block_num=$i begin.#####################"
#        python3 train.py -d survival -s RESULTS/survdl${i}_cv10_drop0${f}_cox_explr -cv 10 -lr 0.0001 -smi 1 --block_num $i --no_atten --dropout_rate $rr -lrs Exp
#    done
#done



# brca 1-8 dropout_rate:0.5-0.9 
#set -v
#for f in {5..9}
#do
#    rr=`echo "scale=3;$f / 10" |bc`
#    for i in {1..8}
#    do
#        echo "#####################dropout_ratio=${rr}, block_num=$i begin.#####################"
#        python3 train.py -d brca -s RESULTS/brcadl${i}_cv10_drop0${f} -cv 10 -smi 2 --block_num $i --no_atten --dropout_rate $rr
#    done
#done
