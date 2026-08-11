# 03. 고급 적용

## 성능

attention 대상 수에 비례해 key 정규화와 logit 계산 비용이 증가한다. block 수, block size와 cache layout을 profiling하고 kernel fusion 가능성을 검토한다. 긴 sequence에서는 depth cache와 KV cache가 동시에 memory를 점유한다.

## 수치 안정성

RMSNorm epsilon, softmax 전 max subtraction, FP32 accumulation을 점검한다. pseudo-query 초기화가 너무 크면 초기에 한 depth로 붕괴할 수 있으므로 attention entropy와 layer별 gradient norm을 기록한다.

## 통합

기존 checkpoint와 구조가 달라 strict load가 불가능할 수 있다. pseudo-query 초기화, 기존 residual에서의 warm start, tensor-parallel shard 규칙과 inference cache serialization을 명시한다.

## 평가

training loss만 보지 말고 downstream reasoning·code, output magnitude, gradient distribution, throughput, peak memory를 함께 측정한다. Full과 Block 사이에는 품질뿐 아니라 block 수별 latency-memory Pareto curve가 필요하다.
