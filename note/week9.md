# 차원축소

# 1. Feature Selection

- 유의미한 변수만 선택
- 장점: 선택한 변수에 대해 해석할 수 있음
- 단점: 변수 간 상관관계를 고려하기 어려움

## 1. 비지도학습

피드백 X

### Filter Method

![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled.png)

통계적 측정 방법 사용 → feature 간 상관관계 파악 → 높은 상관계수를 가지는 feature 사용

상관계수가 높은 feature가 반드시 model에 적합하지는 않음

- Information Gain
- Odds Ratiod

## 2. 지도학습

피드백 O → Feature Selection 반복

반복학습(정답, 예측값 비교를 통해 에러보정)을 통해 성능을 극대화

### Wrapper Method

![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%201.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%201.png)

예측 정확도 측면에서 가장 좋은 성능을 보이는 Feature subset을 뽑아내는 방법

기존 데이터에서 테스트를 진행할 hold-out set을 따로 두어야함

여러번 Machine Learning을 진행하기 때문에 시간과 비용이 매우 높게 발생하지만 최종적으로 Best Feature Subset을 찾기 때문에, 모델의 성능을 위해서는 매우 바람직한 방법

- Forward: 변수가 없는 상태로 시작, 반복할 때마다 가장 중요한 변수를 추가하여 더 이상 성능의 향상이 없을 때까지 변수를 추가
- Backward: 모든 변수를 가지고 시작, 가장 덜 중요한 변수를 하나씩 제거하면서 모델의 성능을 향상
- Stepwise = Foward Selection+ Backward Elimination: 모든 변수를 가지고 시작하여 가장 도움이 되지 않는 변수를 삭제 or 모델에서 빠져있는 변수 중에서 가장 중요한 변수를 추가하는 방법

# 2. Feature Extraction

- 예측 변수의 **변환**을 통해 새로운 변수 추출

    데이터가 가지고 있는 본질적인 정보를 살려서 변수 추출

- 장점: 변수 간 상관관계 고려 (선형결합), 변수의 개수를 많이 줄일 수 있음
- 단점: 추출된 변수의 해석이 어려움

## 1. Max.Variance(선형): PCA

- 비지도 학습
- 데이터(X) **사영** 후(Z) **분산**을 **최대**한 보존할 수 있는 **기저**(a)를 찾아 차원 축소

### Principal Components

주축 = bias와 data의 선형 결합

- 학습 데이터셋에서 분산이 최대인 축(axis)을 찾는다.
- 이렇게 찾은 첫번째 축과 직교(orthogonal)하면서 분산이 최대인 두 번째 축을 찾는다.
- 첫 번째 축과 두 번째 축에 직교하고 분산을 최대한 보존하는 세 번째 축을 찾는다.
- `1~3`과 같은 방법으로 데이터셋의 차원(특성 수)만큼의 축을 찾는다.

### 공분산

데이터 벡터를 어떤 벡터에 사영하는 것이 최적의 결과를 내주는가?

→ 공분산 행렬의 eigen vector와 eigen value

![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%202.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%202.png)

공분산행렬을 구하기 위해 각 데이터들이 얼마나 닮았는지 계산하는 과정

![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%203.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%203.png)

covariance matrix

### 사영

![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%204.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%204.png)

### 선형변환

(벡터) x (행렬)

eigen vector (주축): 방향변화 X, eigen value만큼 크기변화

### PCA 알고리즘

1. 데이터 센터링: 데이터 평균을 0으로 변경
2. 최적화 문제 정의
    1. 데이터 X → W(=S(=X의 covariance matrix)의 eigen vector)에 사영

        ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%205.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%205.png)

        S: X의 covariance matrix

    2. 분산 V 최대화

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%206.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%206.png)

3. 최적화 문제 솔루션

    라그랑지 멀티플라이어 적용 → eigen vector, eigen value

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%207.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%207.png)

4. 주축 정렬

    Eigen value(Eigen vector에 사영된 데이터 분산) 순서대로 정렬: 가장 큰 값을 주축으로 선택

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%208.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%208.png)

5. PCA로 변환된 데이터

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%209.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%209.png)

    Z: 데이터 사영 후, W: X의 covariance matrix의 eigen vector, X: 데이터

6. 원데이터로 복원

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%2010.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%2010.png)

    완벽하게 복원되지 않고 loss 발생

### 주성분 개수 선정법

1. Elbow Point에 해당하는 주성분
2. 일정 수준 이상의 분산 비(비율은 경험적 선택 or Elbow point)를 보존하는 최소의 주성분 선택

### 주성분 분석 한계

1. 데이터가 **단일 가우시안**이여야함
2. **분류** 성능 향상을 보장하지 못함

    ![%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%2011.png](%E1%84%8E%E1%85%A1%E1%84%8B%E1%85%AF%E1%86%AB%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9%203efcce77504a49508391187d106277ad/Untitled%2011.png)

### Randomized PCA

- **자료의 크기, 특성변수의 크기가 매우 클 때** 사용
- QR분해 이용 → 행렬의 SVD

### Kernelized PCA

- **비선형 변환**일 때 사용
- 커널트릭 사용

### scikit-learn PCA

```python
class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
```

Scikit-Learn에서는 PCA를 계산할 때, 데이터셋에 대한 공분산의 **고유값 분해(eigenvalue-decomposition)** 이 아닌 **특이값 분해(SVD, Singular Value Decomposition)** 를 이용해 계산

[차원 축소 - PCA, 주성분분석 (1)](https://excelsior-cjh.tistory.com/167)

→ 이유: eigenvalue-decomposition에서는 공분산 행렬을 메모리상에 가지고 있어야하는 반면 SVD는 공분산 행렬을 따로 메모리에 저장할 필요가 없으므로 효율적이기 때문이다

[Why in doing PCA, scikit-learn use svd to get eigenvalue while the traditional method is to use eig?](https://stackoverflow.com/questions/42291068/why-in-doing-pca-scikit-learn-use-svd-to-get-eigenvalue-while-the-traditional-m?rq=1)

## 2. Max.Dist.Info(선형): MDS

## 3. Reveal Non-linear Structure(비선형)

# 3. 최근 동향: 인공신경망

인공신경망 방법을 이용하여 차원 축소

Representation learning: Deep Auto-Encoder, CNN

feedback

- 선형대수가 중요하다 .. PCA에 대해서 선형대수 강의에서 공부했었던 것도 다시 복습할 수 있는 기회였다.
- Scikit-Learn에서는 PCA를 계산할 때, 데이터셋에 대한 공분산의 **고유값 분해(eigenvalue-decomposition)** 이 아닌 **특이값 분해(SVD, Singular Value Decomposition)** 를 이용해 계산하는데 eigenvalue-decomposition에서는 공분산 행렬을 메모리상에 가지고 있어야하는 반면 SVD는 공분산 행렬을 따로 메모리에 저장할 필요가 없으므로 효율적이기 때문이라고 한다. [https://stackoverflow.com/questions/42291068/why-in-doing-pca-scikit-learn-use-svd-to-get-eigenvalue-while-the-traditional-m?rq=1](https://stackoverflow.com/questions/42291068/why-in-doing-pca-scikit-learn-use-svd-to-get-eigenvalue-while-the-traditional-m?rq=1)