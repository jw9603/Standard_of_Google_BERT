
## Chapter 1. 트랜스포머 입문

### Contents
- 트랜스포머 소개
- 트랜스포머의 인코더 이해하기
- 트랜스포머의 디코더 이해하기
- 인코더와 디코더 결합
- 트랜스포머 학습

### 1.1 트랜스포머 소개

RNN과 LSTM 모델은 NSP, MT, Text generation등의 순차적 태스크에서 널리 사용된다. 하지만 이 네트워크들은 Long-Term Dependency 문제가 있다.
이런 한계점을 극복하기 위해 "Transformer"라는 아키텍처를 제안하였다.
트랜스포머가 출현함으로써 자연어 처리 분야는 획기적으로 발전했으며 BERT, GPT-3, T5등과 같은 혁명적인 아키텍처가 발전하는 기반이 마련되었다.

트랜스포머는 셀프 어텐션이라는 특수한 형태의 어텐션만을 사용한다.
![image](https://github.com/jw9603/Standard_of_Google_BERT/assets/70795645/d99db27b-fd2b-4612-93f7-a0d88679de63)


### 1.2 트랜스포머의 인코더 이해하기
- 트랜스포머는 N(=6)개의 인코더가 쌓인 형태
- 인코더는 Multi-Head Attention 과 Feed Forward

#### 1.2.1 셀프 어텐션의 작동 원리

Ex) A dog ate the food because it was hungry.
이 문장에서 'it'는 'dog'나 'food'를 의미할 수 있다. 하지만 사람들 입장에서는 'it'은 'food'가 아닌 'dog'를 의미한다는 것을 쉽게 알 수 있다.
위와 같은 문장이 주어질 경우 모델은 어떻게 알 수 있을까? 이때 셀프 어텐션이 필요하다.
이 문장이 입력되었을 때, 모델은 가장 먼저 단어 'A'의 representation을, 그다음으로 단어 'dog'의 표현을 계산한 다음 'ate'라는 단어의 표현을 계산한다. 각각의 단어를 계산하는 동안 각 단어의 표현들은 문장 안에 있는 다른 모든 단어의 표현과 연결해 단어가 문장 내에서 갖는 의미를 이해한다.

![image](https://github.com/jw9603/Standard_of_Google_BERT/assets/70795645/0b2ee908-dd87-4037-8012-cc43e93baf55)

출처 : https://www.analyticsvidhya.com/blog/2023/07/transformers-encoder-the-crux-of-the-nlp-issues/

셀프 어텐션을 요약하자면 다음과 같다.
- 단어 그대로 attention을 자기 자신에게 수행한다는 의미
- 입력 문장 내의 단어들끼리 유사도를 구함
- 특정 단어와 문장 내에 있는 모든 단어가 어떤 연관이 있는지를 이해하면 좀 더 좋은 표현을 학습하는데 도움이 됨
- Query, Key, value는 입력 행렬(문장 길이 x 임베딩 차원)로 부터 쿼리 행렬, 키 행렬, 밸류 행렬을 생성한다.
    - 무작위 초기화 된 가중치 행렬 W^Q, W^K, W^V을 만들고 입력 행렬 X에 곱해서 Q, K, V를 생성. 가중치 행렬들은 학습 과정에서 업데이트 됨.
- Q와 K의 내적을 구하고 sqrt(d_k)로 나누기 때문에 Scale-dot product attention이라고도 함.

1. 쿼리(Q) 행렬과 키(K^T) 행렬의 내적 연산을 수행
- Q와 K^T간의 내적을 계산하면 유사도를 구할 수 있음
- 문장의 각 단어가 다른 모든 단어와 얼마나 유사한지 파악하는데 도움을 줌
2. QK^T 행렬을 키 벡터 차원(sqrt(d_k))의 제곱근값으로 나누기
- 안정적인 경사값을 얻을 수 있음
3. Softmax 함수로 Normalizing
- 2번까지의 값은 unnormalized form
- 행 별로 softmax
0 softmax 함수로 normalizing하면 전체 합이 1, 각각의 값은 0~1 사이. 확률값으로 이해가능
- score matrix: 각각의 단어가 문장 전체의 단어와 얼마나 연관이 있는지 확률로 알 수 있음
4. Attention(Z) 행렬 구하기
- Normalized Similarity * V
- Similarity를 Value를 Weight sum!

