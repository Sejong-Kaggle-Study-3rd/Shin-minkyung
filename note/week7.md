# SVM

# 1. 개념

- 지도 학습 모델

- 분류, 회귀 (주로 분류)

- 이진 분류기 (선형 / 비선형: 커널 트릭)

- 학습 방향: margin의 최대화

결정 경계는 주변 데이터와의 거리가 최대화

→ 새로운 데이터가 들어와도 강인한 분류

- 용어

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled.png)

# 2. 선형 SVM

- quadratic program을 풀어 alpha를 구함 (alpha → W,b → 모델)

    *Quadratic Program*(QP)는 목적함수(objective function)가 이차식(convex quadratic)이고, 제약함수(constraint functions)가 모두 affine인 convex optimization problem이다. General quadratic program은 다음과 같은 형태로 표현될 수 있다.

    ### **Quadratic Program**

    > minimizexsubject to where P∈Sn+,G∈Rm x n, and A∈Rp x n.(1/2)xTPx+qTx+rGx⪯hAx=b,

## 1. Hard Margin SVM

- 선형 분리 가능한 문제

- 용어

- 결정 경계 (hyperplane): 서로 다른 클래스를 완벽하게 분류하는 기준

    데이터 임베딩 공간보다 1차원 낮은 부분공간

- 서포트 벡터(support vector): 결정 경계선 가장 가까이에 있는 클래스의 데이터
- 마진 (margin): 데이터 포함 x, 서포트 벡터와 직교하는 직선거리

- 목적 함수: margin의 최대화

1.  original problem

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%201.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%201.png)

- 2. 라그랑지 함수로 변환

    *라그랑지 승수법: 제약이 있는 최적화 문제에서 목적함수로 제약을 옮김으로써 제약이 없는 문제로 변환*

    ![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%202.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%202.png)

    라그랑지 함수

3. primal problem

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%203.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%203.png)

4. dual problem

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%204.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%204.png)

최적해 조건 (KKT Conditions)

1. stationarity
2. primal feasibility
3. dual feasibility
4. complementary slackness

    ![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%205.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%205.png)

## 2. Soft margin SVM

- 선형 분리 불가능 문제

- margin: 데이터 포함
- 결정 경계: 완벽하게 나누는 것 불가능

⇒ 에러 허용 

- 목적 함수: 에러 허용 + margin 최대화

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%206.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%206.png)

C: 에러의 허용 정도 조절

C ↑ → 에러↓ → margin ↓

C ↓ → 에러↑ → margin ↑

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%207.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%207.png)

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%208.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%208.png)

---

# 2. 비선형 SVM

## 1. 커널 SVM

데이터를 선형으로 분류하기 위해 feature map을 통해 데이터의 차원을 높임 = kernel trick

커널: feature map의 내적

- Feature space에서 학습

- Original space: Nonlinear → Feature space: linear

- 목적 함수

![SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%209.png](SVM%204e37cd151fe84880935518de6c01cb4a/Untitled%209.png)

- K(X,Y): kernel function

    암묵적으로 바로 계산 → 연산 효율 

    어떤 kernel이 좋은지는 실험적으로 데이터 특성에 맞게 선택

    1. Linear Kernel
    2. Polynomial Kernel
    3. Sigmoid Kernel
    4. Gaussian Kernel (RBF Kernel)
    5. 

---

# 3. Multiple Classification

SVM: 이진분류기 → 다중분류기로 확장

## 1. One-vs-Rest (OvR)

one = 1 , rest(나머지 class 전체) = -1

class 1 vs 나머지 전체 class / class 2 vs 나머지 전체 class ⇒ N개

결과: 가장 큰 값을 class로 할당

## 2.One-vs-One (OvO)

class 하나씩 비교 

class 1 vs class 2 / class 2 vs class 3 / class 3 vs class 1 ⇒ nC2개

결과: 가장 많이 할당된 class로 할당 (voting)