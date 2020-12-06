import tensorflow as tf
import torch
import datetime
from tokenization_kobert import KoBertTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import urllib3
import base64
from sys import byteorder
from array import array
from struct import pack
import json
import pyaudio
import wave
import rule_based_model as rbm
import os

form_class = uic.loadUiType("ui/119mic.ui")[0]
form_class2 = uic.loadUiType("ui/119dialog.ui")[0]
form_class3 = uic.loadUiType("ui/119list.ui")[0]


class MyWindow(QMainWindow, form_class):
    THRESHOLD = 500
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RATE = 16000

    tag_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'O': 3,
                'B-PER': 4, 'I-PER': 5,
                'B-ORG': 6, 'I-ORG': 7,
                'B-LOC': 8, 'I-LOC': 9,
                'B-POH': 10, 'I-POH': 11,
                'B-DAT': 12, 'I-DAT': 13,
                'B-TIM': 14, 'I-TIM': 15,
                'B-DUR': 16, 'I-DUR': 17,
                'B-MNY': 18, 'I-MNY': 19,
                'B-PNT': 20, 'I-PNT': 21,
                'B-NOH': 22, 'I-NOH': 23,
                'B-EMR': 24, 'I-EMR': 25}

    tag_dict_decode = inv_map = {v: k for k, v in tag_dict.items()}
    model = torch.load('KoBERTmodel_for_AI_201130.pt')
    # CUDA 설치 등의 GPU 설정이 까다로운 상태라면 model = torch.load('KoBERTmodel_for_AI_201130.pt', map_location='cpu')로 변경하여 CPU로 실행하기

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
    device_name = tf.test.gpu_device_name()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.mic.clicked.connect(self.btn_clicked)

    def is_silent(self, snd_data):
        "Returns 'True' if below the 'silent' threshold"
        return max(snd_data) < self.THRESHOLD

    def normalize(self, snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i * times))
        return r

    def trim(self, snd_data):
        "Trim the blank spots at the start and end"

        def _trim(snd_data):
            snd_started = False
            r = array('h')

            for i in snd_data:
                if not snd_started and abs(i) > self.THRESHOLD:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    def add_silence(self, snd_data, seconds):
        "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
        silence = [0] * int(seconds * self.RATE)
        r = array('h', silence)
        r.extend(snd_data)
        r.extend(silence)
        return r

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE,
                        input=True, output=True,
                        frames_per_buffer=self.CHUNK_SIZE)

        num_silent = 0
        snd_started = False

        r = array('h')

        while 1:
            # little endian, signed short
            snd_data = array('h', stream.read(self.CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

            silent = self.is_silent(snd_data)

            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > 30:
                break

        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = self.normalize(r)
        r = self.trim(r)
        r = self.add_silence(r, 0.5)
        return sample_width, r

    def record_to_file(self, path):
        "Records from the microphone and outputs the resulting data to 'path'"
        sample_width, data = self.record()
        data = pack('<' + ('h' * len(data)), *data)

        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(self.RATE)
        wf.writeframes(data)
        wf.close()

    def test_sentences(self, sentences):
        stopFlag = False
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        # 평가모드로 변경
        self.model.eval()
        # 문장을 입력 데이터로 변환
        inputs, masks = self.convert_input_data(sentences)

        device = torch.device("cpu")
        # 데이터를 GPU에 넣음
        b_input_ids = inputs.to(device).long()
        b_input_mask = masks.to(device).long()
        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = self.model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask
                                 )
        logits = np.argmax(outputs[0].to('cpu').numpy(), axis=2)
        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(b_input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, logits[0]):
            if stopFlag == True:
                break
            if token == '[SEP]':  # 패딩 전까지의 출력만을 보기 위해
                stopFlag = True
            new_labels.append(self.tag_dict_decode[label_idx])
            new_tokens.append(token)

        # return logits
        return new_labels, new_tokens

    def inferencing(self, input_text, new_label, new_token):
        all_inf_list = []  # inf_list들로 이루어진 리스트
        inf_list = []  # [태그, 태그에 대응하는 단어]로 이루어진 리스트
        isB = False  # B- 태그가 나왔을 경우 true로 설정
        tagword = ""  # 토큰을 하나씩 연결하여 태그에 대응하는 하나의 단어로 만들기 위한 문자열

        for label, token in zip(new_label, new_token):
            notBFlag = False  # B- 없이 I-만 나온 애들 처리하기 위함
            # B- 태그가 나왔을 경우
            if "B-" in label:
                # 만일 B- 태그가 나온상태에서 또 B- 태그가 나왔을 경우 (개체명이 연속으로 등장했을 경우)
                if isB:
                    inf_list.append(tagword)
                    all_inf_list.append(inf_list)
                    inf_list = []
                    tagword = ""
                # 태그명을 inf_list에 추가 ex) ORG, PER
                inf_list.append(label[2:5])

                if "▁" in token:  # 만일 B- 태그에 대응하는 토큰이 ▁으로 시작할 경우
                    tagword += token.replace("▁", "")  # '▁' 지움
                else:  # 아닐 경우
                    tagword += token  # 그냥 tagword에 추가

                isB = True  # B- 태그가 나왔으니 True로 설정

            # B- 태그가 나온 상태고, I- 태그가 나왔을 때 (B- 태그와 연결되지 않은 I- 태그는 비정상으로 판단)
            elif isB and "I-" in label:

                if label[2:5] not in inf_list:
                    notBFlag = True
                if notBFlag == False:
                    if "▁" in token:  # 만일 I- 태그에 대응하는 토큰이 ▁으로 시작할 경우
                        tagword_before = ""
                        tagword_before += token.replace("▁", "")  # '▁' 지우고 tagword에 추가
                        tagword += tagword_before.replace(tagword_before, " " + tagword_before)
                    else:  # 아닐 경우
                        """
                        그대로 추가
                        <왜?>
                        토크나이징 이후엔 띄어쓰기가 제거된 상태.
                        KoBert의 Sentencepiece tokenize는 띄어쓰기 이후의 단어에(첫 단어이거나) ▁이 붙는 특성을 활용하여
                        띄어쓰기가 포함된 개체명을 제대로 출력하기 위함 ex) 11시 30분, 2020년 8월 15일
                        """
                        tagword += token

            # B- 태그가 나온 상태고, O 태그, 혹은 끝났을 때 (개체명이 끝났을 때)
            elif isB and ("O" in label or "[SEP]" in label):
                inf_list.append(tagword)  # 지금까지의 tagword를 inf_list에 추가
                all_inf_list.append(inf_list)  # inf_list를 all_inf_list에 추가
                inf_list = []  # 싹 다 초기화
                tagword = ""
                isB = False

        # 예쁘게 출력하기
        print(all_inf_list)
        return all_inf_list

    # 입력 데이터 전처리
    def convert_input_data(self, sentences):
        text_CLS = ["[CLS] " + str(txt) + " [SEP]" for txt in sentences]
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        # 토크나이징
        tokenized_texts = [tokenizer.tokenize(sent) for sent in text_CLS]
        MAX_LEN = 128  # MAX_LEN 설정
        # 임베딩 및 패딩 진행
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        # 어텐션 마스크 설정
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        # 데이터를 파이토치의 텐서로 변환
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)

        return inputs, masks

    def see_ok_dialog(self, org, loc, noh, emr, ser):
        '''
        새로운 윈도우 창을 여는 함수로 TaskList를 불러와서 show 새로운 창을 염
        '''
        see_okDialog = OkDialog(org, loc, noh, emr, ser)
        see_okDialog.showModal()

    def btn_clicked(self):
        name = 'speak.wav'  # 저장할 파일 이름
        print("please speak a word into the microphone")
        self.record_to_file(name)
        print("done - result written to" + name)

        openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
        accessKey = "7e73ebba-e4ca-47aa-97fa-364b66aee636" # 개인 키를 발급받아 입력해야 함
        audioFilePath = name
        languageCode = "korean"

        file1 = open(audioFilePath, "rb")
        audioContents = base64.b64encode(file1.read()).decode("utf8")
        file1.close()

        requestJson = {
            "access_key": accessKey,
            "argument": {
                "language_code": languageCode,
                "audio": audioContents
            }
        }

        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )

        print("[responseCode] " + str(response.status))
        print("[responBody]")
        data = json.loads(response.data.decode("utf-8", errors='ignore'))
        print("음성 인식 결과 : ", data['return_object']['recognized'])

        input_text = data['return_object']['recognized']

        new_label, new_token = self.test_sentences([input_text])
        inf_list = self.inferencing(input_text, new_label, new_token)
        loc, org, noh, emr = rbm.rule_based(inf_list)
        ser = rbm.ser_rule(input_text)
        self.see_ok_dialog(org, loc, noh, emr, ser)


class OkDialog(QDialog, form_class2):
    def __init__(self, org, loc, noh, emr, ser):
        super().__init__()
        self.setupUi(self)
        self.ok.clicked.connect(self.ok_clicked)
        self.org_text.setText(org)
        self.loc_text.setText(loc)
        self.noh_text.setText(noh)
        self.emr_text.setText(emr)
        self.ser_text.setText(ser)

    def showModal(self):
        return super().exec_()

    def ok_clicked(self):
        # global org_name, loc_name, noh_name, emr_name, ser_name
        org_name = self.org_text.toPlainText()
        loc_name = self.loc_text.toPlainText()
        noh_name = self.noh_text.toPlainText()
        emr_name = self.emr_text.toPlainText()
        ser_name = self.ser_text.toPlainText()
        list_Dialog = ListDialog(org_name, loc_name, noh_name, emr_name, ser_name)
        list_Dialog.showModal()
        self.reject()


class ListDialog(QDialog, form_class3):
    def __init__(self, org, loc, noh, emr, ser):
        super().__init__()
        self.setupUi(self)
        self.save.clicked.connect(self.save_clicked)
        self.cancel.clicked.connect(self.cancel_clicked)
        self.now = datetime.datetime.now()
        self.final_text_input = ""
        self.final_text_input += str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + " " + str(self.now.hour) + ":" + str(
            self.now.minute) + ":" + str(self.now.second) + "\n" + "\n"
        if org:
            self.final_text_input += "기관 : " + org + "\n"
        if loc:
            self.final_text_input += "장소 : " + loc + "\n"
        if noh:
            self.final_text_input += "인원 : " + noh + "\n"
        if emr:
            self.final_text_input += "상황 : " + emr + "\n"
        if ser:
            self.final_text_input += "정도 : " + ser + "\n"
        self.final_text.setText(self.final_text_input)


    def showModal(self):
        return super().exec_()

    def save_clicked(self):
        title_name = str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + " Recode.txt"
        save_text = self.final_text_input + "---------- \n"
        if not os.path.isfile("Record/" + title_name):
            with open("Record/" + title_name, "w") as f:
                f.write(save_text)
                print("새로운 파일을 생성하여 내용을 추가합니다.")
        else:
            with open("Record/" + title_name, "a") as f:
                f.write(save_text)
                print("기존 파일에 내용을 추가합니다.")

        self.reject()

    def cancel_clicked(self):
        self.reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
