wget -d CMP_facade_DB_base.zip http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
#wget -d CMP_facade_DB_extended.zip  http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip 
mkdir ../datasets/facadesHR
mkdir ../datasets/facadesHR/train_img
mkdir ../datasets/facadesHR/train_label
mkdir ../datasets/facadesHR/val_img
mkdir ../datasets/facadesHR/val_label
unzip -d ../datasets/facadesHR CMP_facade_DB_base.zip 
#unzip -d ../datasets/facadesHR CMP_facade_DB_extended.zip 
cp `cat facades_test_label.txt` ../datasets/facadesHR/val_label/
cp `cat facades_test_img.txt` ../datasets/facadesHR/val_img/
cp `cat facades_train_label.txt` ../datasets/facadesHR/train_label/
cp `cat facades_train_img.txt` ../datasets/facadesHR/train_img/
#rsync -ah --progress ../datasets/facadesHR/extended/*.jpg ../datasets/facadesHR/train_img
#rsync -ah --progress ../datasets/facadesHR/extended/*.png ../datasets/facadesHR/train_label
#rm -rf ../datasets/facadesHR/extended
rm -rf ../datasets/facadesHR/base
rm -rf CMP_facade_DB_base.zip
#rm -rf CMP_facade_DB_extended.zip