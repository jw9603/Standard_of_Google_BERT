
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


