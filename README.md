# mask_r_cnn

1.用jsontodata.bat將label完的圖片 產生包含以下檔案的資料夾
-img.png
-info.yaml
-label.png
-label_names.txt
-label_viz.png

*jsontodata.bat用來批次使用 labelme_json_to_dataset檔案處理label的圖片

2.將label.png 複製並命名為 (原圖名稱).png 放置 samples\trainmy\myinfo\cv2_mask 裡面

3.samples\trainmy\myinfo\json 裡面放label完的json檔案

4.samples\trainmy\myinfo\labelme_json 裡面放jsontodata.bat完產生的資料夾

5.samples\trainmy\myinfo\pic 裡面放用來label的原圖

6.cd 至 mask_r_cnn 資料夾

7.python train.py
