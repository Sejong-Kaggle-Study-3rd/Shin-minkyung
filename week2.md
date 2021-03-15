#2. 전처리와 최적화
---

1. 데이터 전처리
    1. 정의: 데이터의 품질을 올리는 과정
    2. 데이터 전처리 과정
        1. 데이터 실수화: 컴퓨터가 이해할 수 있는 값으로 변환
        2. 불완전한 데이터 제거: NULL, NA, NAN 값 제거
        3. 잡음 섞인 데이터 제거

            가격 데이터에 있는 - 값 제거

            연령 데이터 중 과도하게 큰 값 제거

        4. 모순된 데이터 제거: 남성 데이터 중 주민번호가 '2'로 시작하는 경우
        5. 불균형 데이터 해결: 특정 클래스의 데이터가 많음

            샘플링기법으로 전처리→ 과소표집, 과대표집

    3. 데이터 전처리의 주요 기법
        1. 데이터 실수화(Data Vectorization): 데이터(범주형, 텍스트, 이미지)를 실수로 전환

            2차원 자료(행렬, 2차원 텐서) = [n_sample, n_feature]

            자료의 유형: 연속형 자료, 범주형 자료, 텍스트 자료

            1. 범주형 자료의 실수화
                1. One-hot encoding: 1, 0으로 mapping

                    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7285afc4-5aa9-444a-b2fd-312b9fa68180/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7285afc4-5aa9-444a-b2fd-312b9fa68180/Untitled.png)

                2. Scikit-learn의 DictVectorizer 함수

                    ~~코드~~

                    Sparse = False로 해야 행렬을 볼 수 있음

                    희소행렬(Sparse Matrix): 헹렬의 값이 대부분 0인 경우

                    프로그램 시 불필요한 0값으로 인해 메모리 낭비가 심함 → COO 표현식, CSR표현식

                2. 텍스트 자료의 실수화

                단어의 출현 횟수를 이용한 데이터 실수화

                ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/50cf02db-5dcb-4f5f-8586-277d16b941da/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/50cf02db-5dcb-4f5f-8586-277d16b941da/Untitled.png)

                문제점→ 출현 횟수가 정보의 양과 비례 X (ex. The, a 등의 관사)

                해결→ TF-IDF 기법: 자주 등장하여 분석에 의미를 갖지 못하는 단어의 중요도를 낮추는 기법

        2. 데이터 정제(Data Cleaning): 2-2, 2-3, 2-4
        3. 데이터 통합(Data Integration): 여러개의 데이터 파일을 하나로 합치는 과정
        4. 데이터 축소(Data Reduction): 데이터의 수를 줄이거나(Sampling), 데이터 차원을 축소
        5. 데이터 변환(Data Transformation): 데이터 정규화, 로그, 평균값, 구간화
        6. 데이터 균형(Data Balancing): 2-5
