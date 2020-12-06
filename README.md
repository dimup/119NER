# **119NER**
<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101279253-5c083b00-3804-11eb-9558-63f6623a19c8.png" /></p>

언어모델 기반 개체명 인식 기술을 활용한 119 신고 접수 도움 서비스.  
신고자의 음성에서 (피해 장소, 피해 유형, 피해 인원) 등의 주요 개체들을 인식하여 빠르고 정확한 상황 요약을 제공합니다.


#### **파일 설명**
+ "KoBERT_NER_KMOU_for_119NER.ipynb"
  + KoBERT 기반의 개체명 인식 모델을 구현하는 과정 및 코드가 작성된 파일입니다.
  + 보다 고성능 환경에서 구현하기 위해 Google Colab에서 작성하였습니다.
+ "119ner.py"
  + main code가 작성된 파일입니다.
  + ETRI 음성 인식 API, 구현된 개체명 인식 모델, 규칙 기반 모델을 사용하여 실제 기능을 하도록 이루어져 있습니다.
+ "rule_based_model.py"
  + 임의의 문장이 구현된 개체명 인식 모델을 통과한 결과를 일정한 출력 형태로 맞추기 위한 규칙 기반 모델입니다.  

#### **사용법**
+ "KoBERT_NER_KMOU_for_119NER.ipynb"를 제외하고, 모두 같은 폴더에 설치한 뒤, "119ner.py"를 실행하면 됩니다. 
  + "requirements.txt"를 통해 필요한 패키지 및 라이브러리를 설치를 한 번에 진행하면 더욱 수월합니다.
+ 실행을 위해서는 ETRI API 개인키를 신청하여 발급받아야 합니다. (개발자의 개인 키를 공유해드릴 순 없으므로..)
  + 발급 후  "119NER.py" 파일의 287 line에서 accessKey 변수에 키 값을 대입하면 됩니다.
+ 구현한 개체명 인식 모델은 용량 문제로 구글 드라이브에서 공유했습니다. 해당 url을 통해 "119ner.py"와 같은 폴더에 내려받으면 됩니다.
  + https://drive.google.com/file/d/16Vjpc1WlhL7jov-RvtvF95zWD4CWSb41/view?usp=sharing


#### **참고**
+ ETRI 음성인식 open API : https://aiopen.etri.re.kr/guide_recognition.php
+ 한국해양대학교 개체명 코퍼스 : https://github.com/kmounlp/NER
+ Google BERT : https://github.com/google-research/bert
+ SK T-Brain KoBERT : https://github.com/SKTBrain/KoBERT


## 119NER 구현 과정

<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101279555-57448680-3806-11eb-9f6f-17de4d10402a.png" /></p>
<br>

### 음성 -> 텍스트


### 데이터셋
한국해양대학교 개체명 코퍼스에서 input data와 target data 각각 약 21000 문장을 파싱.
+ Training set : 약 17000 문장
+ Validation set : 약 4000 문장


추가적으로 119 신고 도메인에 맞추기 위해 '피해 유형'을 나타내는 EMR 태그 생성했고, 이에 대한 문장 데이터를 각각 1000여개씩 구축함
<br>
<br>
### 모델링
<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101274746-eee4ad80-37e3-11eb-9601-45ceac5140ea.png"/>
</p>  
사전학습 된 KoBERT 모델에 Token 분류 레이어를 추가하는 형태로 구현. 

### Training 및 Validation
Hyper parameters를 다음과 같이 설정하여 Training 진행

# 내용 추가 예정
