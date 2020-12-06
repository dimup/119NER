# **119NER**
<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101279253-5c083b00-3804-11eb-9558-63f6623a19c8.png" /></p>

**언어모델 기반 개체명 인식 기술을 활용한 119 신고 접수 도움 서비스로, 신고자의 음성에서  
(피해 장소, 피해 유형, 피해 인원) 등의 주요 개체들을 인식하여 빠르고 정확한 상황 요약을 제공합니다.**


<br>

### **주요 파일 설명**
+ **"KoBERT_NER_KMOU_for_119NER.ipynb"**
  + KoBERT 기반의 개체명 인식 모델을 구현하는 과정 및 코드가 작성된 파일입니다.
  + 보다 고성능 환경에서 구현하기 위해 Google Colab에서 작성하였습니다.
+ **"119ner.py"**
  + main code가 작성된 파일입니다.
  + ETRI 음성 인식 API, 구현된 개체명 인식 모델, 규칙 기반 모델을 사용하여 실제 기능을 하도록 이루어져 있습니다.
+ **"rule_based_model.py"**
  + 소규모의 규칙 기반 모델이 작성된 파일입니다.
  + 임의의 문장이 구현된 개체명 인식 모델을 통과한 결과를 일정한 출력 형태로 맞추는 역할을 합니다.
+ **"tokenization_kobert.py"**
  + KoBERT의 SentencePiece tokenization 기능이 작성된 파일입니다.

### **사용법**
+ **"KoBERT_NER_KMOU_for_119NER.ipynb"를 제외하고, 모두 같은 폴더에 설치한 뒤, "119ner.py"를 실행하면 됩니다.** 
  + "requirements.txt"를 통해 필요한 패키지 및 라이브러리를 설치를 한 번에 진행하면 더욱 수월합니다.
  + 실행 환경에 따라 부가적인 GPU 설정이 필요할 수 있으므로, CPU만으로 실행하는 방법을 47 line에 주석으로 기술해놓았습니다.
+ **실행을 위해서는 음성 인식을 위해 ETRI API 개인키를 신청하여 발급받아야 합니다. (개발자의 개인 키를 공유할 수 없습니다.)**
  + 발급 후  "119NER.py" 파일의 287 line에서 accessKey 변수에 키 값을 대입하면 됩니다.
+ **구현한 개체명 인식 모델은 용량 문제로 구글 드라이브에서 공유했습니다. 해당 url을 통해 "119ner.py"와 같은 폴더에 내려받으면 됩니다.**
  + https://drive.google.com/file/d/16Vjpc1WlhL7jov-RvtvF95zWD4CWSb41/view?usp=sharing


### **참고**
+ ETRI 음성인식 open API : https://aiopen.etri.re.kr/guide_recognition.php
+ 한국해양대학교 개체명 코퍼스 : https://github.com/kmounlp/NER
+ Google BERT : https://github.com/google-research/bert
+ SK T-Brain KoBERT : https://github.com/SKTBrain/KoBERT
+ huggingface.co : https://huggingface.co/transformers/model_doc/bert.html

<br>

## 119NER 구현 과정

<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101279555-57448680-3806-11eb-9f6f-17de4d10402a.png" /></p>
<br>

### 음성 -> 텍스트 구현
1. 사용자의 발화를 녹음하고, 오디오 파일로 생성하도록 합니다.
2. 해당 오디오 파일을 ETRI의 음성인식 API 서버로 전달하여 인식 결과를 텍스트로 받습니다.

### KoBERT 기반 개체명 인식 모델 구현
_세부 과정은 "KoBERT_NER_KMOU_for_119NER.ipynb"에 작성되어있습니다._  


**1. Data 수집 및 구축**
+ 한국해양대학교 개체명 코퍼스에서 input data와 target data 각각 약 21000 문장을 파싱한 뒤 Training, Validation 데이터로 분리합니다.
  + Training set : 약 17000 문장
  + Validation set : 약 4000 문장
+ 추가적으로 119 신고 도메인에 맞추기 위해 '피해 유형'을 나타내는 EMR 태그 생성했고, 이에 대한 문장 데이터를 각각 약 1000개씩 구축합니다

**2. Input data, Target data 전처리**
+ BERT 구조 형식에 맞게 데이터 전처리를 진행합니다. Input data와 Target data의 전처리는 차이가 있지만, 다음의 공통 과정을 거칩니다.
  + [CLS], [SEP] 토큰 부착
  + SentencePiece tokenizing
  + Embedding
  + Padding
  
**3. Modeling**
+ 사전학습된 KoBERT 모델에 Token Classification Layer를 쌓은 형태로 모델링을 진행합니다.
  + Huggingface의 transformers 라이브러리에서 BertForTokenClassification 클래스를 활용했습니다.

**4. Training**
+ Optimizer와 Hyper parameters를 다음과 같이 설정한 뒤 학습을 진행합니다.
  + Optimizer : AdamW optimizer
  + Learning rate : 1e-5
  + Epsilon : 1e-8
  + Epochs : 50
  + Batch size : 8
+ 학습을 마친 뒤 Validation을 수행합니다. 토큰 간 정확도(Accuracy)를 측정했습니다.
  + 정확도 결과 : 약 92%

**5. Testing**
+ 임의의 문장을 학습이 완료된 모델에 통과시켜 그 결과를 확인합니다.

### 구현 모델의 인식 결과를 가공 및 정리하는 코드 구현
**1. 인식 결과 가공**
+ 토큰 형태로 반환되는 결과를 다시 단어들로 결합합니다.
  + SentencePiece tokenization의 구분자인 '▁'를 활용했습니다.
**2. 결과 정리**
+ 출력될 결과를 일정한 형태로 맞추기 위해 구현한 규칙 기반 모델에 통과시킵니다.
+ 통과한 결과를 기반으로 요약문을 생성합니다.

#### 음성 입력부터 요약문 생성까지의 기능을 Python GUI 프로그램으로 사용할 수 있도록 구현했습니다.
