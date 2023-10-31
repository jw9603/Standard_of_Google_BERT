
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

#### 1.2.2 멀티 헤드 어텐션 원리

문장 내에서 모호한 의미를 가진 단어가 있을 경우에 앞의 예와 같이 적절한 의미를 가진 단어의 벡터값이 잘 할당되었을 경우에는 문장의 의미를 이해하는 데 좋은 영향을 줄 수 있다. 하지만 그 반대의 경우, 즉 의미가 맞지 않은 단어의 벡터값이 높을 경우에는 문장의 의미가 잘못 해석될 수 있다.
그래서 어텐션 결과의 정확돌르 높이기 위해서 단일 헤드 어텐션 행렬이 아닌 멀티 헤드 어텐션을 사용한 후 그 결괏값을 더하는 형태로 진행한다.
이와 같은 방법으 사용하는 데는 단일 헤드 어텐션을 사용하는 것보다 멀티 헤드 어텐션을 사용하면 좀 더 정확하게 문장의 의미를 이해할 수 있다는 가정이 깔려 있다.
구체적인 방법으로는 MHA의 결과로 나온 각각의 Attention vector를 concat하고 새로운 가중치 행렬 W^0를 곱하는 방법을 사용한다. (Attention head의 최종 결과는 Attention head의 원래 크기이므로 크기를 줄이기 위해 W_0을 곱하는 것)


#### 1.2.3 Positional Encoding
RNN은 순차적으로 문장이 들어가지만 Transformer는 모든 단어를 병렬 형태로 입력한다. 병렬로 단어를 입력하는 것은 학습 시간을 줄이고 RNN의 long-term Dependency 문제를 해결하는 데 도움이 된다.
하지만 트랜스포머에서 단어를 병렬로 입력하면 한 가지 문제가 발생한다. 그것은 바로 단어의 순서 정보가 유지되지 않은 상태에서 문장의 의미를 어떻게 이해할 수 있느냐는 점이다.
![image](https://github.com/jw9603/Standard_of_Google_BERT/assets/70795645/ccec3a0f-6121-47fb-9bbd-06d90ac7d0d1)
- Pos : 문장에서 단어의 위치
- i : 해당 위치의 임베딩

위치 인코딩 P를 계산한 후 임베딩 행렬 X에 대해서 element-wise addition을 수행한 후 인코더의 입력 행렬로 입력한다.

#### 1.2.4 Feed Forward Network(FFN)

2개의 dense layer + ReLU activation func. FFN의 파라미터(weight param)는 문장의 다른 위치에서는 동일하고 인코더 블록에서는 다르게 나타난다.

#### 1.2.5 ADD & NORM
- sublayer에서 MHA의 input과 output을 서로 연결
- sublayer에서 FFN의 input과 output을 서로 연결
- Layer Normalization + Residual Connection
- Layer Normalization: 각 layer의 값이 크게 변화하는 것을 방지해 모델을 더 빠르게 학습할 수 있게 함
- Residual Connection: Vision의 ResNet에서 보던 구조. Skip Connection이라고도 하는데 기존 학습 정보를 보존하고 추가적인 정보만 학습하기 때문에 학습이 쉬워지고, 수렴이 빨라진다.

### 1.3 트랜스포머의 디코더 이해하기
- Encoder의 결과값을 가져와서 Decoder의 입력값으로 사용(Decoder의 K,V)
- Decoder는 이전 Decoder의 출력과 Encoder의 출력을 입력으로 사용함
    - <SOS> + 문장 embedding -> 첫 단어 생성
    - <SOS>첫단어 + 문장 embedding -> 첫 단어, 두 번째 단어 생성
    - <EOS>가 나올 때까지 반복
- Decoder의 입력도 Positional Encoding을 더해서 사용

**디코더의 구성요소**
- Masked multi-head attention
- Encoder-Decoder Multi-head attention
- FFN

#### 1.3.1 Masked Multi-Head Attention
- Decoder에서는 앞에 <sos>를 붙여서 input으로 넣으면, output으로 맨 뒤에 <eos>를 붙여서 결과가 나옴
- 한 단계 shifted된 문장을 출력
- Self-attention은 각 단어의 의미를 이해하기 위해 각 단어와 전체 단어를 연결함.
- Decoder에서는 이전 단계에서 생성한 단어까지만 들어감
- 따라서 각 단어에서 아직 예측되지 않은 오른쪽은 볼 수 없다고 생각 → mask!
- 똑같이 계산하되, softmax 함수에 넣기 전에 -∞으로 mask

![image](https://github.com/jw9603/Standard_of_Google_BERT/assets/70795645/22476f54-56dc-4e27-a216-c55fba21842e)


#### 1.3.2 Encoder-Decoder Multi-Head Attention
- Decoder의 MHA는 이전 sublayer의 출력(M)과, Encoder의 출력(R)을 입력으로 받음.
- Encoder-Decoder attention layer: Encoder의 결과와 Decoder의 결과 사이의 상호작용이 일어나는 곳
- 행렬 M으로 Q를 만들고, R로 K, V를 생성
- 일반적으로 Q는 target 문장의 표현을 포함하기 때문에 M에서 가져옴
- K, V는 입력 문장의 표현을 가져서 R을 참조
- Q, K, V가 어디서 왔는지를 제외하면 다른 과정들은 Encoder의 MHA와 동일함.

#### 1.3.3 FFN, add, norm

Encoder와 동일

#### 1.3.4 Linear, Softmax Layer

- REMIND: Decoder는 입력 단어들(<sos> + 지금까지 예측한 단어들)을 입력받아서 다음 단어를 예측
- Linear layer는 마지막 decoder의 최상위 output을 input으로 받아서 vocab 크기와 같은 logit을 출력
- 이 logit을 softmax에 넣어주면 확률값으로 변환 가능
- vocab에서 가장 큰 확률을 가지는 단어를 argmax로 찾아내면 → 그 단어가 예측된 다음 단어


### 1.4 Transformer 학습

- Decoder가 vocab에 대한 확률 분포를 예측하고 확률이 가장 큰 단어를 선택함
- 실제 확률 분포와 모델이 예측하는 확률 분포의 차이를 줄여나가는 방법으로 학습
- loss function으로 cross-entropy를 사용
- optimizer는 Adam을 사용
- overfitting을 막기 위해 각 sublayer의 output, embedding, positional encoding의 합을 구할 때 Dropout을 적용

### Summary

이번 장에서는 트랜스포머 모델이 무엇인지, 인코더-디코더 아키텍처가 어떤 원리로 작동하는지를 다뤘다. 트랜스포머의 인코더 부분을 살펴보면서 멀티 헤드 어텐션과 피드포워드 네트워크 같은 인코더에서 사용하는 다양한 sublayer를 확인했다.

셀프 어텐션은 단어를 좀 더 잘 이해하기 위해 주어진 문장의 모든 단어와 해당 단어를 연결하는 형태다. 셀프 어텐션을 계산하기 위해 Query, Key, Value 행렬이라는 세 가지 행렬을 사용했다.

그다음으로 위치 인코딩을 계산하는 방법과 위치 인코딩을 사용해 문장 내 단어의 순서를 입력하는 방법을 살펴보았다. 인코더에서 피드포워드 네트워크가 작동하는 방법과 add & norm 에 대해서도 알아보았다.

디코더에서는 masked multi-head attention, encoder-decoder attention, FFN 등 디코더에서 사용하는 서브레이어를 알아봤다.

### 연습문제

1. 셀프 어텐션의 전체 단계를 설명하라.
   
sol)

- 쿼리 행렬과 키 행렬간의 내적을 계산하고 유사도를 구하고 root(차원 크기)값으로 나눈다.
- 스코어 행렬에 대해 softmax를 적용해 정규화를 진행한다
- 마지막으로 score matrix에 value matrix를 곱해 attention matrix를 구한다.
  
2. 스케일 닷 프로덕트 어텐션을 정의하라.
   
sol)

- self attention의 메커니즘을 scaled-dot product attention이라고도 한다. 그 이유는 쿼리와 키 벡터의 내적을 먼저 계산하고, 값에 대한 스케일링을 적용하기 때문이다.
   
3. 쿼리, 키, 밸류 행렬은 어떻게 생성하는가?
   
sol)

-  쿼리, 키, 밸류 행렬을 생성하기 위해 Wq,Wk,Wv의 새로운 matrix를 사용한다. 입력 행렬의 W matrix를 각각 곱해 생성한다.

4. 위치 인코딩이 필요한 이유는 무엇인가?
   
sol)

- 트랜스포머는 병렬 형태로 한 번에 문장이 들어가므로 위치 정보를 파악할 수 없다. 따라서 필요하다.

5. 디코더의 서브레이어에는 무엇이 있는가?
   
sol)

- masked multi-head attention
- Encoder-Decoder multi-head attention
- FFN
  
6. 디코더의 인코더-디코더 어텐션 레이어의 입력은 무엇인가?
   
sol)

- 인코더의 마지막 레이어에서 오는 representation vector(K,V matrix)와 이전 masked_multi-head attention layer의 출력값(Q matrix)


### References
1. VASWANI, Ashish, et al. Attention is all you need. Advances in neural information processing systems, 2017, 30.
2. https://jalammar.github.io/illustrated-transformer/
3. https://www.analyticsvidhya.com/blog/2023/07/transformers-encoder-the-crux-of-the-nlp-issues/
    






