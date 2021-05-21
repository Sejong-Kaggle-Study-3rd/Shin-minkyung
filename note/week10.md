# 교차검증

## Overview

전처리→ 머신러닝 알고리즘 → 성능평가를 반복하여 최적의 모델을 찾는다. 

- 전처리: 차원축소(PCA,LDA), 정규화
- 머신러닝 알고리즘: KNN, Logistic Regression, LDA,  SVM, Decision Tree, K-means, 앙상블
- 성능평가: 교차검증(Fold out cross validation, K-fold cross validation)

    성능평가도구: 파이프라인(연속된 변환을 묶어서 처리할 수 있는 wrapper 도구)

# 1. 알고리즘 선택 :: 앙상블

**여러 분류기를 하나로 연결**하여 개별 분류기보다 더 좋은 일반화 성능을 달성하는 것

## 1. 여러 분류 알고리즘 사용: Voting

**동일한 학습 데이터**로 **여러 알고리즘**을 사용한뒤 voting을 통해 최종 예측

```python
class sklearn.ensemble.VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
```

```python
from sklearn.ensemble import VotingClassifier
voting_estimators = [('logistic', logistic), ('tree',tree), ('knn', knn)]
voting = VotingClassifier(estimators = voting_estimators, voting = 'soft') #soft: 확률기반
```

## 2-1. 하나의 분류 알고리즘을 여러번 사용: Bagging

Bootstrap을 사용하여 알고리즘 수행마다 **서로 다른 학습 데이터**를 복원 추출(중복 O) 하여 사용

```python
class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
```

### Random Forest

Decision Tree를 사용하여 무작위로 예측변수를 선택하여 모델 구축

## 2-2. 하나의 분류 알고리즘을 여러번 사용: Boosting

샘플을 뽑을 때 **잘못 분류된 데이터의 50%를 재학습**에 사용하거나 **가중치**를 사용

# 2. 모델 성능 평가 :: 교차 검증

## 1. Fold out cross validation

전체 데이터 = 학습 데이터 + 검증 데이터 + 테스트 데이터로 나누어서 모델 검증

학습 데이터: 모델 학습

검증 데이터: 하이퍼파라미터 튜닝

테스트 데이터: 성능 추정

- 학습 데이터와 검증 데이터를 어떻게 나누는지에 대해 민감

## 2. K-fold cross validation

훈련 데이터 = **(K-1 fold: 모델 훈련) +  (1 fold: 성능 평가) (중복X)** 을 K번 반복하여 K개의 서로 다른 모델을 얻어 각 fold에서 얻은 성능을 기반으로 평균 성능을 계산

- K 값

    추천하는 K 값 = 10

    K ↑ → 실행시간 ↑

- fold out cross validation보다 데이터 분할에 대해 덜 예민

    중복없기 때문에 모든 샘플이 검증에 딱 한 번 사용됨

## 3. 모델 성능 최적화

Overfitting, Underfitting이 없어야 모델 성능이 최적화

### Overfitting

학습 데이터에만 잘 맞아 일반화 ↓

해결방법

- 학습 데이터 추가 수집
- 모델 제약 늘리기: regularization ↑ (logistic regression의 C값)
- 학습 데이터 노이즈 줄이기

### Underffiting

모델이 너무 단순하여 학습 기능 ↓

해결방법

- 복잡한 모델 선택: 파라미터가 더 많은 모델 선택
- 모델 제약 줄이기: regularization ↓
- overfitting이전까지 학습

### Overfitting/Underfitting 판단

1. 학습 곡선의 편향과 분산 분석

```python
class sklearn.model_selection.learning_curve(estimator, X, y, *, groups=None, train_sizes=array([0.1, 0.33, 0.55, 0.78, 1.0]), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', verbose=0, shuffle=False, random_state=None, error_score=nan, return_times=False, fit_params=None)
```

2. 검증곡선

```python
class sklearn.model_selection.validation_curve(estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0, error_score=nan, fit_params=None)
```

### GridSearch방식을 이용해 모델 최적화 :: 최적의 파라미터

```python
class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
```