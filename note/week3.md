# 3. KNN(K-Nearest Neighbors): 최근접 이웃
___
## ⇒ **분류**와 **회귀** 해결 가능
지도학습
1. 분류
    - 미리 정의된, 가능성 있는 여러 클래스 레이블 중 하나를 예측
    - 이진분류(두 개로만 나눔), 다중 분류(셋 이상의 클래스로 분류)
    - ex. 얼굴 인식, 숫자 판별(MNIST: 사람이 손으로 쓴 digit data)
2. 회귀
    - 연속적인 숫사, 부동소수점수를 예측하는 것
    - ex. 주식 가격을 예측하여 수익을 내는 알고리즘, 공부 시간에 따른 학점 예측
___

## 1. KNN의 개념
###    1. 정의: 주변 k개의 자료의 클래스 중 가장 많은 클래스로 특정 자료를 분류하는 방식
- 새로운 자료를 가장 가까운(거리를 계산: minkowski 거리를 이용) 자료 5개의 자료(k=5)를 이용하여 투표하여 가장 많은 클래스로 할당

![minkowski 거리: p=1 맨하탄거리, p=2 유클리디언 거리](https://user-images.githubusercontent.com/80238096/111924221-b775e480-8ae6-11eb-8fcf-848f53ad7865.png "minkowski 거리: p=1 맨하탄거리, p=2 유클리디언 거리")
- **학습 데이터 자체가 모형** = 추정 방법x, 모형 x
> ↔ linear classfication: wx + b의 판별식(discriminative function)을 갖고 데이터를 분류 (w,b라는 파라미터를 추정)
    
-  **게으른 학습(lazy learner), 사례중심학습(instance-based learning)** :훈련 데이터 셋을 메모리에 저장하는 방법 
> ↔ model based learning: 수학 모델의 파라미터를 추정

___
###    2. 문제점: 차원의 저주(curse of dimension)
차원이 증가할수록 공간의 크기가 기하급수적으로 증가 => **데이터의 밀도가 희박(sparse)** 해짐
___
###    3. KNN의 하이퍼파라미터

####        1. 탐색할 이웃 수(k): 근접치 K의 개수에 따라 Group이 달리 분류

#####        (1) 다수결 방식(Majority Voting): 이웃 범주 가운데 빈도가 높은 범주로 새로운 데이터의 범주를 예측 
= 거리에 관계없이 무조건 많이 voting된 쪽으로 판단
→ __K에 따라 속하는 class가 변경__

#####        (2) 가중합 방식(Weighted voting): 가까운(거리 = 유사도) 이웃의 정보에 좀 더 가중치를 부여
→ __K에 따라 속하는 class 불변__
            
####        2. 거리 측정 방법(맨하탄 거리, 유클리디언 거리)
* k가 너무 작으면 ⇒ **overfitting**
>데이터의 경향성을 파악하지 못해 새로운 데이터를 오판할 수 있음
* k가 너무 크면 ⇒ **underfitting**
>학습 데이터에 대한 표현력도 적고 테스트 데이터에 대한 표현력도 적음
___

###        5. 장단점
####        1. 장점
1. 학습데이터의 노이즈에 큰 영향 x
2. 학습데이터 수가 많으면 효과적인 알고리즘: 전체 공간에서 데이터의 표현력을 보일 수 있음
3. 간단하지만 성능이 떨어지지 않음
4. 마할로비스 거리와 같이 사용할 때 매우 __강건(robust)__ 한 방법론이 됨
####        2. 단점
1. 최적 이웃의 수(k), 어떤 거리 척도(distance mertric)이 적합한지 알 수 없어 을 __경험적 선택__ = 데이터 각각의 특성에 맞게 연구자가 임의로 설정
> → best K = 탐욕적인 방식으로 탐색

2. 새로운 관측치와의 거리를 전부 측정해야해서 __계산 시간이 오래걸림__
> → 해결 방식; Locality Sensitive Hashing, Network based Indexer, Optimized product quantization
___
## 2. KNN 분류
### 기계학습의 일반적인 실습 순서
#### 1. 데이터 셋 불러오기
Iris 데이터셋
- seabon 라이브러리: 데이터 시각화를 담당하는 모듈
- 통계 그래픽을 그리기 위한 고급 인터페이스 제공
- 약간의 변수, 파라미터 조정으로 쉽게 그래프 표현 가능

#### 2. 데이터셋 카테고리의 실수화
DictVectorize(One-hot encoding) vs LabelEncoder(범주형 라벨)
* LabelEncoder.fit_transform: 문자열을 cateogrial 값으로 전환

#### 3. 데이터 분할
* data = train + validation + test
* 데이터 분할 = 학습 데이터(train)와 시험 데이터(test)를 서로 __겹치지 않도록__ 나누는 것
- 목적: 학습과 평가를 나누기위해
> 학습데이터로 학습시키고 학습에 사용되지 않은 시험데이터에 적용하여 generalization이 가능한지
- validation
 정답이 없을 때⇒ overfiiting의 여부를 판단하기 위해 test 데이터의 일부를 validation으로 나눠놓고 학습 중간에 평가로 사용
→ 학습하는 과정에서는 사용 x
> 정답이 있을 때 ⇒ 학습 데이터가 많을수록 좋으므로 학습데이터로 사용

* train_set_split(train,test,test_size,random_state,stratify)
    * __straitify__: default = None → 쏠려서 분배될 수 있음
        __-> 고정해주면 일정한 비율로 분배__

#### 4. 입력데이터의 표준화
경우에 따라 성능이 좋아지기도 하고 나빠지기도 함(통계적 특성이 깨지면 학습 저하) 
- 표준화
    - 측정 단위에 의해 영향 받지 않도록 하는 과정
    - StandardScaler 클래스 사용

- test data의 표준화 = train data 에서 구한 특성 변수의 평균, 표준편차 이용
    - __항상 학습과 시험 데이터를 분리해서 사용__

#### 5. 학습: 모형 추정 또는 사례 중심학습
#### 6. 모델에 데이터 입력
#### 7. 결과 분석
- 성능평가
__1. confusion_matrix(y_true, y_pred)__
> 혼합행렬: 원래 클래스와 예측 클래스가 일치하는 갯수를 표로 나타냄
__2. accuracy_score(y_true, y_pred)__
> 전체 샘플 중 맞게 예측한 샘플의 비율
3. precision_score(y_true, y_pred)
> 양성 클래스에 속한다고 예측한 샘플 중 실제 양성 클래스에 속하는 샘플의 비율
4. recall_score(y_true, y_pred) 
> 실제 양성 클래스에 속한 표본 중 양성 클래스에 속한다고 예측한 표본의 수의 비율
5. fbeta_score(y_true, y_pred, beta)
6. f1_score(y_true, y_pred)
7. roc_curve
8. auc

___
## 3. KNN 회귀
KNN 분류와 동일
* 차이점: y의 예측치 계산 
    * k개의 정답의 __평균__ 을 구함
### 1. 단순 회귀: 단순 평균
### 2. 가중회귀 (weighted regression)
가중평균을 구해 가중치 부여
> 거리가 가까울수록 유사도 증가
* KNN 회귀를 이용한 영화 평점 예측
    * 등급을 예측
> Binary Classification: 평이 좋다 vs 나쁘다로 분류하는 것이 아님
