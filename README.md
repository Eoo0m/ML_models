# lgbm.py
 ### 주요 특징
	•	1:1 언더샘플링으로 불균형 완화
	•	hour cyclic 변환(sin/cos) 적용
	•	max_bin=511, depth↑로 미세 구간의 차이를 세밀하게 포착
	•	seq 정보를 반영하기 위해 BiGRU seq32 벡터 추가(seq있는 parquet 사용해야함)


# bigru.py

원본 데이터의 seq 컬럼(숫자 토큰 시퀀스)을
32차원 의미 벡터(seq32) 로 변환하는 전용 모델

### 주요 특징
	•	BiGRU 기반 Masked LM 방식 사전학습
	•	PAD/UNK/MASK 토큰 포함한 자체 vocab 구성
	•	시퀀스 길이를 tail-truncate 후 left-padding
	•	마지막 hidden state를 평균 풀링 → 32차원 프로젝트 벡터


# DeepCrossCTR.py

복잡한 고차 교호작용과 시퀀스 정보의 활용을 목표로 설계한 딥러닝 모델.

### 내부 구성
	1.	Numerical Feature + BatchNorm
	•	history 계열의 극단적 분포를 완화하기 위해
일부 피처는 네제곱근 변환으로 안정화
	2.	Categorical Embedding
	•	각 카테고리 ID를 임베딩으로 변환해 dense feature로 통합
	3.	Bi-LSTM 기반 시퀀스 인코더
	•	seq token을 embedding 후 LSTM으로 양방향 요약
	4.	DCNv2
	•	복잡한 피처 교호작용을 구조적·명시적으로 생성하는 핵심 모듈
	•	Wide & Deep보다 더 안정적으로 고차 관계 포착
	5.	MLP (BN + GELU + Dropout)
	•	분포 폭등을 억제하기 위해 cross 이후 BN 적용
	•	ReLU 대신 GELU로 미세 신호까지 부드럽게 반영
