# 3.KNN(K-Nearest Neighbors): 최근접 이웃
___
⇒ 분류와 회귀 해결 가능

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

            ![image](https://user-images.githubusercontent.com/80238096/111924221-b775e480-8ae6-11eb-8fcf-848f53ad7865.png)

        p=1, 맨하탄 거리/ p=2, 유클리디언 거리

        - 학습 데이터 자체가 모형일 뿐 어떠한 추정 방법도 모형도 없음
            <-> linear regression(classfication): wx + b의 판별식(discriminative function)을 갖고 데이터를 분류 (w,b라는 파라미터를 추정)
        -  게으른 학습(lazy learner), 사례중심학습(instance-based learning) ↔ model based learning: 수학 모델의 파라미터를 추정
            훈련 데이터 셋을 메모리에 저장하기 방법
            ![image](https://user-images.githubusercontent.com/80238096/111924221-b775e480-8ae6-11eb-8fcf-848f53ad7865.png)


___
###    2. 문제점: 데이터의 차원이 증가하면 차원의 저주(curse of dimension)
          차원이 증가할수록 공간의 크기가 기하급수적으로 증가 => 데이터의 밀도가 희박(sparse)해짐
___
###    3. KNN의 하이퍼파라미터

####        1. 탐색할 이웃 수(k): 근접치 K의 개수에 따라 Group이 달리 분류

#####        * 다수결 방식(Majority Voting): 이웃 범주 가운데 빈도 기준 제일 많은 범주로 새로운 데이터의 범주를 예측 
                = 거리에 관계없이 무조건 많이 voting된 쪽으로 판단
                → K에 따라 속하는 class가 변경

#####        * 가중합 방식(Weighted voting): 가까운(거리 = 유샤도) 이웃의 정보에 좀 더 가중치를 부여
               → K에 따라 속하는 class 불변
            
####        2. 거리 측정 방법(맨하탄 거리, 유클리디언 거리)
            * k가 너무 작으면 ⇒ overfitting: 데이터의 경향성을 파악하지 못해 새로운 데이터를 오판할 수 있음
            * k가 너무 크면 ⇒ underfitting: 학습 데이터에 대한 표현력도 적고 테스트 데이터에 대한 표현력도 적음
___

###        5. 장단점
####        1. 장점
            1. 학습데이터의 노이즈에 큰 영향 x
            2. 학습데이터 수가 많으면 효과적인 알고리즘: 전체 공간에서 데이터의 표현력을 보일 수 있음
            3. 간단하지만 성능이 떨어지지 않음
            4. 마할로비스 거리와 같이 사용할 때 매우 강건(robust)한 방법론이 됨
####        2. 단점
            1. 최적 이웃의 수(k), 어떤 거리 척도(distance mertric)이 적합한지 알 수 없어 을 경험적으로 선택 = 데이터 각각의 특성에 맞게 연구자가 임의로 설정

                → best K = 탐욕적인 방식으로 탐색

            2. 새로운 관측치와의 거리를 전부 측정해야해서 계산 시간이 오래걸림

                → 해결 방식; Locality Sensitive Hashing, Network based Indexer, Optimized product quantization
