# 법률 도메인 번역기

본 프로젝트는 한국법제연구원 대한민국 영문법령에서 한영 법령 문장 데이터쌍을 크롤링 및 전처리하여 수집하여, Transformer Network로 학습함으로써 법률 도메인에 특화된 번역기를 구현하였습니다.

<br/>

---

### 데이터의 준비

한영 문장쌍으로 구성된 데이터를 확보하기 위하여 [한국법제연구원 사이트](https://elaw.klri.re.kr/kor_service/main.do)의 문장쌍을 크롤링 및 전처리 했습니다.

관련 코드는 하기와 같습니다.

| file            |description                                                  |
| ------------------- |------------------------------------------------------------|
| `01_크롤링.ipynb`       |Selenium과 BeautifulSoup 모듈을 이용하여 한국법제연구원의 한영 법령 문장 쌍을 엑셀 파일로 스크래핑합니다. 웹사이트 구조 개편이 있을 시, html parsing 과정에서 코드의 수정이 필요합니다.|
| `02_파일 골라내기.ipynb`       |한영 문단의 쌍이 맞지 않는 문서를 골라냅니다. (다수의 문서들이 코드 구현의 결과를 검토한 후 수작업으로 처리되었기 때문에 쥬피터노트북 상에서는 코드들이 엉켜있을 수 있습니다.)|
| `03_line by line.ipynb`       |html 상에서 단락 단위로 스크랩된 문서들을 문장(또는 문단) 단위로 쪼개는 코드의 예제입니다. 부칙 부분의 생략(omitted)으로 인해 문장쌍이 중간부터 안 맞게 된 경우, 해당 단락은 쌍으로 제거됩니다. (다수의 문서들이 코드 구현의 결과를 검토한 후 수작업으로 처리되었기 때문에 쥬피터노트북 상에서는 코드들이 엉켜있을 수 있습니다.)|
| `04_중복문장 다수 제거.ipynb`       |영어를 기준으로 중복된 문장을 제거합니다. 예를 들어, '제1장 총칙' 같은 문장이 수천번 나오는데, 이러한 경우들을 제거해줍니다.|
| `05_문장쪼개기.ipynb`       |max sequence length 내에 문장들이 처리될 수 있도록 일부 문단으로 이루어진 라인들을 분리하는 작업을 수행합니다. (참고로, 모델 내 max sequence length를 120 정도로 설정하면, 기본적인 문장 쪼개기 작업만을 수행하여도 대부분의 문장들을 처리할 수 있었기에 추가적인 작업은 보류하였습니다.)|

<br/>