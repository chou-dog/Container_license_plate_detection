# 貨櫃號碼的影像辨識  
利用YOLO（You Only Look Once）和OCR（Optical Character Recognition）技術結合的方法，用於貨櫃號碼的辨識。首先通過使用YOLO模型檢測貨櫃，然後對檢測到的貨櫃區域進行OCR辨識。    
下圖為YOLO偵測到貨櫃車牌的位置  
![image](https://github.com/chou-dog/Container_license_plate_detection/blob/main/YOLO.jpg)
下圖為使用tesseractOCR來對貨櫃號碼進行辨識  
![image](https://github.com/chou-dog/Container_license_plate_detection/blob/main/container_dection.png)  
下圖為針對影片將每秒拆成10偵，對每偵進行貨櫃號碼辨識，辨識結果採用多數決的方式來決定最終辨識到的貨櫃號碼準確率。  
![image](https://github.com/chou-dog/Container_license_plate_detection/blob/main/result1.png)
![image](https://github.com/chou-dog/Container_license_plate_detection/blob/main/result2.png)
