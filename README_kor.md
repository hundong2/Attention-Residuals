# Attention Residuals (AttnRes)

[English](README.md) | **한국어** | [한국어 학습 가이드](guide/README.md) | [논문 PDF](Attention_Residuals.pdf) | [arXiv](https://arxiv.org/abs/2603.15031)

이 저장소는 Transformer의 표준 residual connection을 대체하는 **Attention Residuals(AttnRes)** 공식 저장소다. 각 layer는 이전 표현을 고정 가중치로 모두 더하는 대신, 입력에 따라 학습된 depth 방향 attention으로 필요한 표현을 선택한다.

## 개요

표준 residual은 모든 layer 출력을 단위 가중치로 누적한다. 깊어질수록 개별 layer의 기여가 희석되고 PreNorm hidden-state 크기가 제한 없이 증가할 수 있다. AttnRes는 이전 layer 출력에 softmax attention을 적용한다.

$$\mathbf{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\mathbf{v}_i$$

가중치 $\alpha_{i \to l}$는 layer마다 하나인 학습 가능한 pseudo-query $\mathbf{w}_l\in\mathbb{R}^d$로 계산된다. 이에 따라 각 layer는 앞선 모든 표현에 content-aware하게 접근한다.

## Block AttnRes

Full AttnRes는 직관적이지만 깊이 L에서 O(Ld) memory가 필요하다. Block AttnRes는 layer를 N개 block으로 묶고 block 내부는 표준 residual로 누적하며 block 대표 표현 사이에서만 attention을 적용한다. 논문은 약 8개 block이 Full 방식 이득의 대부분을 유지하면서 memory를 O(Nd)로 낮춘다고 보고한다.

## 결과

- scaling law에서 모든 compute budget의 baseline보다 낮은 loss를 보였으며, Block AttnRes는 1.25배 compute로 학습한 baseline과 같은 loss를 기록했다.
- Kimi Linear 48B(활성 3B), 1.4T token 실험에서 MMLU 73.5→74.6, GPQA-Diamond 36.9→44.4, HumanEval 59.1→62.2 등 모든 공개 표 항목이 개선됐다.
- PreNorm dilution을 완화해 depth에 따른 출력 크기를 제한하고 gradient norm을 layer 전반에 더 균일하게 분배했다.

이 수치는 저장소가 포함한 논문의 특정 학습 조건에 대한 결과이며 다른 architecture나 규모에서 자동으로 보장되지 않는다.

## 코드와 학습

저장소 README는 PyTorch 스타일 의사코드를 제공하지만 완전한 학습 구현이나 package 설정은 포함하지 않는다. 개념부터 구현·검증까지는 [한국어 가이드](guide/README.md)와 [표준 라이브러리 toy example](guide/examples/attnres_toy.py)을 사용한다.

## 인용

연구에 사용할 경우 원본 README의 [Citation](README.md#citation)에 있는 BibTeX를 사용한다.
