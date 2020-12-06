# 119NER
KoBERT 기반 개체명 인식 기술을 활용한 119 신고 접수 도움 서비스
<p align="center"><img src="https://user-images.githubusercontent.com/46772883/101274524-12a6f400-37e2-11eb-9b05-0ddce34585c2.png" /></p>


# 119NER 모델 구현 과정
- 세부 내용은 KoBERT_NER_KMOU_for_119NER.ipynb 파일에서 확인할 수 있음 **
---  
### 데이터셋
한국해양대학교 개체명 코퍼스에서 input data와 target data 각각 약 21000 문장을 파싱.
- Training set : 약 17000 문장
- Validation set : 약 4000 문장


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
