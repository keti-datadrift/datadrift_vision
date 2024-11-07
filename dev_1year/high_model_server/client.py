import requests
import json
from time import time,sleep
import os

cwd = os.getcwd()
print(cwd)
# os.chdir(cwd+'/dev_1year/high_model_server/')
cwd = os.getcwd()
print(cwd)

url = "http://172.16.15.77:8080/api/drift/v1/high_model/"

i = 0
while 1:
    # Image file to upload
    image0 = {"image": open("./images/a0.jpg", "rb")}
    image1 = {"image": open("./images/a1.jpg", "rb")}
    image2 = {"image": open("./images/a2.jpg", "rb")}
    image3 = {"image": open("./images/a3.jpg", "rb")}
    image4 = {"image": open("./images/a4.jpg", "rb")}
    image5 = {"image": open("./images/a5.jpg", "rb")}
    image6 = {"image": open("./images/a6.jpg", "rb")}
    image7 = {"image": open("./images/a7.jpg", "rb")}
    image8 = {"image": open("./images/a8.jpg", "rb")}
    image9 = {"image": open("./images/a9.jpg", "rb")}
    image10 = {"image": open("./images/a10.jpg", "rb")}
    image11 = {"image": open("./images/a11.jpg", "rb")}
    text = '1. 사고 유형: 화재\n2. 이미지 설명: 영상에는 야외 산업 지역이 보이며, 지면 중앙에 작은 불이 타고 있다. 한 사람이 근처에 서서 화재를 관찰하고 있다.\n3. 우선순위: 긴급\n4. AI가 추천하는 대응 조치: 즉시 소방서와 현장 인력에 화재 진압을 알리세요. 필요 시 지역을 대피시키고, 전문 도움이 도착 할 때까지 사용 가능한 소화기를 이용해 화재를 진압하세요.\n5. 카메라 상태: 정상'
    # Text description
    aiout0 = {"ai_memo": f"{text}0. 어린이들이 기차놀이 하고 있습니다. 옆에는 우거진 나무들이 있고, 많은 사람들이 보고 있습니다.이 사진에는 나무가 우거진 야외 공간에서 여러 명의 아이들이 함께 춤을 추고 있는 장면이 담겨 있습니다. 아이들은 모두 캐주얼한 청바지와 흰색 상의를 입고 있으며, 서로 팔을 벌려 손을 맞잡은 상태로 원형을 그리며 춤을 추는 모습입니다. 사진의 배경에는 많은 사람들이 앉아 있거나 서서 이 장면을 관람하고 있는 모습이 보입니다. 나무가 무성하게 자란 배경과 아이들의 활동적인 춤 동작이 조화를 이루며, 자연 속에서 열린 즐거운 행사임을 보여줍니다. 관람객들은 대부분 편안한 옷차림을 하고 있으며, 웃으며 춤을 지켜보고 있는 듯합니다.", "score": 0.91,"event": "기차놀이","camera_uid":"카메라 방배1"}
    aiout1 = {"ai_memo": f"{text}1. 이 사진은 도시의 횡단보도를 건너고 있는 한 여성을 중심으로 촬영된 장면입니다. 여성은 밝은 하늘색 상의와 청바지를 입고 있으며, 어깨에는 가죽 가방을 메고 있습니다. 그녀는 여유로운 걸음으로 횡단보도를 건너고 있으며, 도로에는 여러 대의 차량이 대기하고 있는 모습이 보입니다.\
                            차량 중 가장 눈에 띄는 것은 노란색 택시로, 여성을 향해 직진할 수 있는 위치에 있습니다. 그 뒤로는 여러 색상의 일반 차량들이 줄지어 서 있으며, 모두 정지해 있는 상태입니다. 도로 양쪽에는 공원처럼 보이는 푸른 잔디와 나무가 심어진 녹지 공간이 있어 도심 속에서도 자연이 조화를 이루고 있는 느낌을 줍니다.\
                            사진은 차량과 보행자가 공존하는 일상적인 도심 풍경을 담고 있으며, 여성은 안전하게 도로를 건너는 모습을 보여줍니다.", "score": 0.99,"event": "횡단보도를 건너고 있는 한 여성","camera_uid":"카메라 방배1"}
    aiout2 = {"ai_memo": f"{text}2. 이 사진은 밝고 맑은 날씨 아래에 있는 대형 광장의 모습을 담고 있습니다. 사진 속 광장은 사람들이 많이 모여 있는 활기찬 분위기로, 여러 명의 사람들이 광장의 벤치에 앉아 휴식을 취하거나 대화를 나누고 있습니다. \
                            광장의 왼쪽에는 고풍스러운 붉은 벽돌과 하얀 외벽이 조화를 이루는 건물이 보이며, 시계탑이 있는 중앙의 타워는 역사적인 느낌을 줍니다. 이 건물은 웅장한 건축 양식을 보여주며, 많은 사람들이 이 주변을 오가고 있습니다.\
                            광장 중앙에는 분수가 자리하고 있고, 그 주변에는 녹색 식물이 심어져 있어 휴식을 취하기에 좋은 공간을 제공합니다. 분수 주변에는 가족 단위의 사람들이 유모차와 함께 앉아 쉬고 있는 모습도 보입니다.\
                            오른쪽에는 큰 광고판이 설치되어 있으며, 'TIO PEPE'라는 큰 간판과 넷플릭스의 광고가 눈에 띕니다. 광고판 옆에는 기마 동상이 자리하고 있어 광장의 랜드마크 역할을 하고 있습니다.\
                            이 사진은 현대적이고 활기찬 도심 속에서 역사적인 건축물과 현대적인 요소가 조화를 이루는 모습이며, 사람들이 자유롭게 휴식하고 여가를 즐기는 광경을 보여줍니다.", "score": 0.95,"event": "대형 광장의 모습, 사람들이 휴식","camera_uid":"카메라 방배1"}
    aiout3 = {"ai_memo": f"{text}3. 하얀 건물과 나무.", "score": 0.91,"event": "하얀 건물과 나무","camera_uid":"카메라 방배1"}
    aiout4 = {"ai_memo": f"{text}4. 군인이 백파이프 불고 있음.", "score": 0.99,"event": "군인이 백파이프 불고 있음","camera_uid":"카메라 방배1"}
    aiout5 = {"ai_memo": f"{text}5. 파라솔 아래에 앉아 있는 커플.", "score": 0.99,"event": "파라솔 아래에 앉아 있는 커플, 노랑 아파트 옆에서","camera_uid":"카메라 방배1"}    
    aiout6 = {"ai_memo": f"{text}6. 어린이들이 줄 아래를 지남.", "score": 0.95,"event": "어린이들이 줄 아래를 지남","camera_uid":"카메라 방배1"}
    aiout7 = {"ai_memo": f"{text}7. 빨간 옷 입은 여성.", "score": 0.91,"event": "빨간 옷 입은 여성","camera_uid":"카메라 방배1"}
    aiout8 = {"ai_memo": f"{text}8. 많은 사람들이 노인들 무도회 인듯.", "score": 0.99,"event": "많은 사람들이 노인들 무도회 인듯","camera_uid":"카메라 방배1"}
    aiout9 = {"ai_memo": f"{text}9. 분홍색 어린이 들이 발레이.", "score": 0.95,"event": "분홍색 어린이 들이 발레이","camera_uid":"카메라 방배1"}
    aiout10 = {"ai_memo": f"{text}10. 페레이드.", "score": 0.91,"event": "페레이드","camera_uid":"카메라 방배1"}
    aiout11 = {"ai_memo": f"{text}11. 빨간 어린이들 광장.", "score": 0.99,"event": "빨간 어린이들 광장","camera_uid":"카메라 방배1"}
    aiout12 = {"ai_memo": "restart", "score": 0,"event": "restart","camera_uid":"restart"}

    # aiout12 = {"ai_memo": "11. .", "score": 0.95,"event": "Hounoring the spirit of children 시위 행진"}
    
    image_files =[image0,image1,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11]
    aiouts = [aiout0,aiout1,aiout2,aiout3,aiout4,aiout5,aiout6,aiout7,aiout8,aiout9,aiout10,aiout11]

    zp = list(zip(image_files,aiouts))
    length = len(zp) 

    image_file, aiout = zp[i]

    # image_file = image0
    # aiout = aiout12
    data = {"item": json.dumps(aiout)}

    response = requests.post(url, data=data, files=image_file)
    # Output the response
    print('-------------------------------',url)
    # print(response.json())
    print(response)

    i += 1
    i = i%length
    if 0==i:
        print('one cycle is finished!!!')

    sleep(3)    
    
