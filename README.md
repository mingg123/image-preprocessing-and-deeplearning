# 영상처리와 딥러닝을 이용한 유방암 판별 알고리즘

팀원


인하대학교 정보통신공학과 김민지

인하대학교 정보통신공학과 전수연

데모영상 : https://www.youtube.com/watch?v=2foBg-5BIPA&t=21s

데이터셋 : ftp://figment.csee.usf.edu/pub/DDSM/cases/

Python, Matlab을 이용하여 암 예상 영역을 추출한 뒤 머신러닝, 딥러닝에 돌려 정확도를 확인했습니다.



<div>
  <img width="1000" src="https://user-images.githubusercontent.com/52990629/71306002-852a9280-241e-11ea-9dfa-8c890a52e3f4.JPG">
  
  </div>
  
  LGPEG 파일을 PNG로 변환했습니다. 이후 라벨을 제거하고 근육을 제거했습니다.
  
  
  <div>
  <img height="500" width="800" src="https://user-images.githubusercontent.com/52990629/71306005-89ef4680-241e-11ea-8298-be468bbf237d.JPG">
  
  </div>
  
  근육 제거 과정입니다.
  
  <div>
    <img width="1000" src="https://user-images.githubusercontent.com/52990629/71306004-88258300-241e-11ea-9334-ebd63f764772.JPG">
  
  
  </div>
  
  
  
  alarm segment(암예상)영역을 추출한 이후, ehd(edge histogram discript) 20차원의 벡터를 생성했습니다.
  
  
  overlay는 실제 암영역 입니다. 
  
  
  <div>
    <img width="1000" src="https://user-images.githubusercontent.com/52990629/71306007-8bb90a00-241e-11ea-8511-54bcd852df3d.png">
    
    
   </div>
    
    
  딥러닝에 학습 시켰습니다.
  
  
  
   <div>
    <img height="500" width="600" src="https://user-images.githubusercontent.com/52990629/71306008-8cea3700-241e-11ea-876d-6e10ebd83d2b.png">
    
    
   </div>
    
    
  drop out층을 캡쳐한 모습입니다.
    
    
    
    
  머신러닝의 정확도는 대략 94.2% 이며, 딥러닝의 정확도는 98%가 나옴.

    
    
  
  
  
