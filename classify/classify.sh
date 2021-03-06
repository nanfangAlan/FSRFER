#classify with Covpool-FER
##bicubic
#for i in $(seq 2 8)
#do
#python3 classify_FER.py CLASSIFY /data/RAFDB_bicubic/jpg/test/x${i}.0 "/fer_model/RAFDB&SFEW/20170815-144407" "/fer_model/RAFDB&SFEW/svc_144407.pkl" --batch_size=128 --image_size=100
#done

##metaSR
#for i in $(seq 2 8)
#do
#python3 classify_FER.py CLASSIFY /data/RAFDB_metaSR/jpg/test/x${i}.0 "/fer_model/RAFDB&SFEW/20170815-144407" "/fer_model/RAFDB&SFEW/svc_144407.pkl" --batch_size=128 --image_size=100
#done

##RCAN
#for i in $(seq 2 4)
#do
#python3 classify_FER.py CLASSIFY /data/RAFDB_RCAN/jpg/test/x${i}.0 "/fer_model/RAFDB&SFEW/20170815-144407" "/fer_model/RAFDB&SFEW/svc_144407.pkl" --batch_size=128 --image_size=100
#done


#classify with FSR-FER
##FSRFER
#train
#python3 classify_FSRFER.py TRAIN /opt/FSRFER/data/RAFDB/RAFDB_100/train_100 "/opt/FSRFER/checkpoint/WGAN_div/2021-04-26T22-11-20-load-pre-classimg144k-img" "/opt/FSRFER/checkpoint/WGAN_div/2021-04-26T22-11-20-load-pre-classimg144k-img/svc_test.pkl" --batch_size=32 --image_size=100


#classify
#python3 classify_FSRFER.py CLASSIFY /opt/FSRFER/data/RAFDB/RAFDB_100/test_100 "/opt/FSRFER/checkpoint/WGAN_div/2021-04-26T22-11-20-load-pre-classimg144k-img" "/opt/FSRFER/checkpoint/WGAN_div/2021-04-26T22-11-20-load-pre-classimg144k-img/svc.pkl" --batch_size=32 --image_size=100
#python3 classify_FSRFER.py TRAIN /opt/FSRFER/data/RAFDB/RAFDB_100_bicubicup/RAFDB_100_lowx1/train "/opt/FSRFER/checkpoint/WGAN_div/2021-10-07T22-39-38" "/opt/FSRFER/checkpoint/WGAN_div/2021-10-07T22-39-38/svc.pkl" --batch_size=32 --image_size=100
#python3 classify_FSRFER.py TRAIN /opt/FSRFER/data/RAFDB/RAFDB_100_multiscale_upto100_full/train_100 "/opt/FSRFER/checkpoint/WGAN_div/2021-10-07T22-39-38" "/opt/FSRFER/checkpoint/WGAN_div/2021-10-07T22-39-38/svc.pkl" --batch_size=32 --image_size=100

#python3 classify_FSRFER.py TRAIN /opt/FSRFER/data/RAFDB/RAFDB_100_multiscale_upto100_full/train_100 "/opt/FSRFER/checkpoint/WGAN_div/2021-1127T0912" "/opt/FSRFER/checkpoint/WGAN_div/2021-1127T0912/svc.pkl" --batch_size=32 --image_size=100
python3 classify_FSRFER.py CLASSIFY /opt/FSRFER/data/RAFDB/RAFDB_100_bicubicup/RAFDB_100_lowx1/test "/opt/FSRFER/checkpoint/WGAN_div/2021-1127T0912" "/opt/FSRFER/checkpoint/WGAN_div/2021-1127T0912/svc.pkl" --batch_size=32 --image_size=100


