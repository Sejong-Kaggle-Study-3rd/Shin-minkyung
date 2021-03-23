## Categorical Encoding
categorical data를 numerical data로 인코딩
### 1. One-Hot Encoding
```python
class sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse=True, dtype=<class 'numpy.float64'>, handle_unknown='error')
```
- parameter 참고
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

- Encode categorical features as a one-hot numeric array
    - 기계에 학습시키기 위해 문자열을 숫자로 바꾸기 위한 전처리 작업
    - categorical feature -> one-hot(one-of-K or dummy) 숫자 배열로 인코딩 = 하나의 값만 True고 나머지는 False
> 각각의 범주를 속성으로 만들어서 해당 범주면 1 아니면 0

    - 2차원 데이터

- parameter
    1. categories(default = auto): auto 또는 array-like
      * auto:training datad에서 자동으로 category를 결정
    2. drop(default=None):{‘first’, ‘if_binary’} 또는 a array-like of shape (n_features,)
      1. None
      2. first: 각 feature에서 첫번째 category drop
      3. if_binary: 두 개의 category에서 각 feature에서 첫번째 category drop
      4. array: drop
    3. sparse (default = True)
    4. dtype(default = float): 자료형
    6. handle_unknown(default = error): 새로운 data가 들어왔을 때 무시할지 에러를 발생시킬지

사용: 순서가 없을 때(연속적인 특성x), 고유값의 개수가 적을 때
### 2. Lable Encoding
categorical feature -> 알파벳 순서로 숫자 할당(0,1,2,3..)
문제점: 숫자로 변환되어 예측 성능 저하-> 숫자의 크기에 대한 특성 (ex.선형회귀)
>데이터의 연속성: 1에 할당된 데이터와 3의 할당된 데이터의 중간 데이터를 2라고 할 수 없음

사용: 순서의 의미가 있을 때(연속적인 특성 o), 고유값의 개수가 많을 때
