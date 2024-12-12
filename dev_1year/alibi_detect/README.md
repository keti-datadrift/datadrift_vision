

# alibi-detect
- 데이터 드리프트를 https://github.com/SeldonIO/alibi-detect.git 의 alibi-detect를 활용하여 측정할 수 있음


## Environment

| Language            | Version | 
|---------------------|---------|
| python              | 3.8     |
   
   
   
```
  conda create -n <env_name> python=3.8
  
  # cuda 사양에 맞는 torch 설치 필요
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  
  pip install -r requirements.txt
```
## Dataset Rule

### 1. 데이터 셋은 다음과 같이 구성되어야 함 


| original_file                                                                                                                       | compare_file                                         | 
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| - 모델이 학습한 데이터로 데이터 드리프트가 발생하기 이전의 상황을 가정한 이미지의 폴더                                                                                   | - 모델이 학습하지 않은 데이터로 데이터 드리프트가 발생한 이후의 상황을 가정한 이미지의 폴더 |
| - Train:Test 절반으로 나뉘어 적용됨<br>- gaussian_noise, motion_blur, brightness, pixelate가 각각 적용되어 compare image와 더불어 기본 train 이미지와의 비교가 진행됨 
                                                                                                                                                                                      


### 2. <orignal_file_path>와 <compare_file_path> argument를 채워서 다음 명령어를 터미널에서 실행
```
    python main.py -o <original_file_path> -c <compare_file_path>
```
### 3. 출력을 확인
```
# 출력 예시
    No corruption
    Drift? No!
    p-value: 0.630
    Time (s) 4.840
    
    Corruption type : train_images_gaussian_noise
    Drift? Yes!
    p-value 0.000
    Time (s) 4.387
    
    Corruption type : train_images_motion_blur
    Drift? Yes!
    p-value : 0.000
    Time (s) 4.427
    
    Corruption type : train_images_brightness
    Drift? Yes!
    p-value : 0.000
    Time (s) 4.430
    
    Corruption type : train_images_pixelate
    Drift : Yes!
    p-value : 0.000
    Time (s) 4.423
    ================END================
    Elapsed time : 0:00:01.426061

```

## Description

### [-](link)    
  - -

## Issues to Address

- [ ] 다른 detector 옵션 추가
