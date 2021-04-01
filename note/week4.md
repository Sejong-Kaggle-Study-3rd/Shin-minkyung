# Logistic regression
## 다중선형회귀
  * 수치형 설명변수 X -> 연속형 숫자 종속변수 Y
  * X와 Y의 관계를 선형으로 가정
  * 이를 잘 표현할 수 있는 회귀계수를 데이터로부터 추정

### 다중선형회귀 모델 방정식
  이미지 첨부

### 회귀 계수 결정법
1. Direct Solution
회귀계수 = 오차제곱합을 최소로 하는 값
-> 명시적 해: error식의 미분값이 0이 되는 곳


2. Numerical Search
회귀계수 = 경사하강법(gradient descent)
> 경사하강법(gradient descent): 어떤 함수(목적함수, 비용함수, 에러 값)값을 최소화하기 위해 임의의 시작점을 잡은 후 해당 지점에서의 경사 기울기를 구하고 경사의 반대방향으로 조금씩 이동하는 과정(learning rate)를 여러번 반복하는 과정

1. Batch Gradient Descent(GD) = Vanila Gradient Descent
파라미터를 업데이트 할 때: __모든 학습데이터__ 사용하여 gradient 계산
* 매우 낮은 학습 효율


2. Stochastic Gradient Descent(SGD)
파라미터를 업데이트 할 때: __무작위로 샘플링__ 된 학습데이터를 하나씩만 이용하여 gradient 계산
* 모델을 자주 업데이트 가능
* local minima에 빠질 가능성 ↓
* 최소 cost에 수렴했는지 판단이 어려움


3. Mini Batch Gradient Descent
파라미터를 업데이트 할 때: __일정량의 일부 데이터를 무작위__ 로 뽑아 gradient계산
* Batch GD + Stochastic GD
-> SGD의 노이즈를 줄임 + GD의 전체 배치보다 효율적


### 정규화
variance를 감소시켜 일반화 성능을 높이는 기법
> 미래 데이터에 대한 오차의 기대값 = bias + variance
> 회귀계수가 가질 수 있는 값에 제약조건을 부여하여 미래 데이터에 대한 오차 기대

* bias 증가할 수 있음
* 정규화 정도는 사람이 핸드튜닝 해줘야함

#### bias-variance decomposition
**일반화** 성능을 높이는 정규화, 앙상블 기법의 이론적 배경
> 일반화: 설계한 모델이 보편적으로 잘 동작하는 지

* 미래데이터에 대한 오차의 기대값 = bias + variance로 decompose하자는 내용
* bias ↓, variance ↑ = NN, SVM, KNN(small K)
> 튜닝만 잘하면 과녁을 제대로 맞출 수 있음

* bias ↑, variance ↓ = 로지스틱 회귀, LDA, KNN(large K)
> 데이터 노이즈에 강인한 모델

* boosting: bias를 줄여 성능을 높임
* lasso regression: variance를 줄여 성능을 높임


## Logistic regression
>다중선형회귀의 문제점: 범주형 데이터에 적용 불가

#### logstic function
* S-커브 함수 = sigmoid 함수
이미지  첨부

### 이항 로지스틱 회귀
Y가 범주형일 경우 다중선형회귀 모델을 적용할 수 없음
> Y = 연속형 숫자, 범주형 = 0,1인 의미없는 비연속형 숫자

범주형 데이터를 연속형 숫자로 change
~수식~

### 다항 로지스틱 회귀
