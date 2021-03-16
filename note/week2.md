# 2. 전처리와 최적화
---

## 1. 데이터 전처리
###      1. 정의
데이터의 품질을 올리는 과정
          
###      2. 데이터 전처리 과정
1. 데이터 실수화: 컴퓨터가 이해할 수 있는 값으로 변환
2. 불완전한 데이터 제거: NULL, NA, NAN 값 제거
3. 잡음 섞인 데이터 제거: 가격 데이터에 있는 - 값 제거, 연령 데이터 중 과도하게 큰 값 제거
4. 모순된 데이터 제거: 남성 데이터 중 주민번호가 '2'로 시작하는 경우
5. 불균형 데이터 해결: 특정 클래스의 데이터가 많음
   * 샘플링기법으로 전처리→ 과소표집, 과대표집
                
###      3. 데이터 전처리의 주요 기법
####        1. 데이터 실수화(Data Vectorization): 데이터(범주형, 텍스트, 이미지)를 실수로 전환
* 2차원 자료(행렬, 2차원 텐서) = [n_sample, n_feature]
  * 자료의 유형: 연속형 자료, 범주형 자료, 텍스트 자료

#####            1. 범주형 자료의 실수화
1. One-hot encoding: 1, 0으로 mapping
2. Scikit-learn의 DictVectorizer 함수

```python
x = [{'city':'seoul', 'temp':10.0},{'city':'Dubiai', 'temp':33.5},{'city':'LA','temp':20.0}]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False) #Sparse = False로 해야 행렬을 볼 수 있음
vec.fit_transform(x)
```
* 희소행렬(Sparse Matrix): 헹렬의 값이 대부분 0인 경우
   프로그램 시 불필요한 0값으로 인해 메모리 낭비가 심함 → COO 표현식, CSR표현식

#####            2. 텍스트 자료의 실수화
1. 단어의 출현 횟수를 이용한 데이터 실수화
* 문제점: 출현 횟수가 정보의 양과 비례 X (ex. The, a 등의 관사)
* 해결: __TF-IDF 기법__ 
          자주 등장하여 분석에 의미를 갖지 못하는 단어의 중요도를 낮추는 기법
```python
text = {'떴다 떴다 비행기 날아라 날아라',
       '높이 높이 날아라 우리 비행기',
       '내가 만든 비행기 날아라 날아라',
       '멀리 멀리 날아라 우리 비행기'}

from sklearn.feature_extraction.text import CountVectorizer
vec2 = CountVectorizer() #defult는 sparse = True
t = vec2.fit_transform(text).toarray() #toarray로 sparse 압축 해제 또는 sparse = False로 하면 행렬을 볼 수 있음

import pandas as pd
t1 = pd.DataFrame(t, columns = vec2.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
x2 = tfid.fit_transform(text).toarray()
x3 = pd.DataFrame(x2, columns = tfid.get_feature_names())
```
---
####        2. 데이터 정제(Data Cleaning): 2-2, 2-3, 2-4
#####            결측데이터(Empty Values) 채우기
* 결측 데이터: np.nan, npNaN, None
평균, 중위수, 최빈수로 대처하는 기법
-> sklearn의 Imputer()

---
####        3. 데이터 통합(Data Integration): 여러개의 데이터 파일을 하나로 합치는 과정
pandas의 merge

---
####        4. 데이터 축소(Data Reduction): 데이터의 수를 줄이거나(Sampling), 데이터 차원을 축소
---
####        5. 데이터 변환(Data Transformation): 데이터 정규화, 로그, 평균값, 구간화
* 데이터 변환의 필요성: 머신러닝 알고리즘은 데이터가 가진 특성들을 비교하여 데이터패턴을 찾음
          -> 데이터가 가진 특성 간 스케일 차가 크면 패턴을 찾는데 문제 발생
#####            1. 표준화
#####            2. 정규화
* 정규화가 표준화보다 유용
* bell-shape, 이상치(노이즈 많거나, empty data가 많은 경우) -> 표준화가 유용

---
####        6. 데이터 균형(Data Balancing): 2-5
* 불균형 자료: (머신러닝 목적이 분류일 때) 특정 클래스의 관측치가 다른 클래스에 비해 매우 낮게 나타남
#####            1. 과소표집(undersampling): 다수 클래스 제거
#####            2. 과대표집(oversampling): 소수 클래스 복제 -> 학습데이터 추가
                    ex. SMOTE, ADASYN
* 과소표집보다 과대표집이 통계적으로 유용
* 의사결정나무, 앙상블: 불균형 자료에 강인
