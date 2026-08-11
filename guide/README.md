# Attention Residuals 한국어 학습 가이드

작성일: 2026-08-12

## 학습 순서

1. [기초와 수식](01_foundations.md): residual, PreNorm dilution, depth attention
2. [구현과 실험](02_implementation.md): Full/Block AttnRes, shape와 검증
3. [고급 적용](03_advanced.md): 효율·안정성·통합·평가
4. [실행 예제](examples/README.md): 외부 의존성 없는 Python toy implementation

## 프로젝트가 해결하는 문제

표준 residual stream은 깊이가 증가할수록 과거 모든 출력을 동일하게 누적한다. AttnRes는 layer마다 학습하는 pseudo-query와 정규화된 이전 표현의 유사도로 depth attention을 계산해 현재 layer에 필요한 과거 표현을 선택한다. Full 방식의 cache 비용은 Block 방식으로 제한한다.

## 저장소 범위

현재 저장소는 논문 PDF, 설명 README와 그림을 제공한다. production-ready PyTorch module·학습 script·test는 없으므로 가이드 예제는 알고리즘 의미를 확인하는 toy implementation이다. 원 논문 결과를 재현했다고 주장하지 않는다.

## 빠른 시작

```powershell
python guide/examples/attnres_toy.py
```

Python 3.10+ 표준 라이브러리만 필요하다. 성공하면 standard sum, Full attention 가중치·출력, Block attention 결과와 assertion 통과 메시지가 출력된다.

## 핵심 점검표

- softmax 축은 layer/block 축인가?
- value는 정규화 전 표현이고 key 계산에만 norm을 쓰는가?
- completed block과 현재 partial block의 경계가 정확한가?
- 혼합정밀도에서 logits와 softmax가 안정적인가?
- baseline과 parameter·token·compute budget을 공정하게 맞췄는가?
