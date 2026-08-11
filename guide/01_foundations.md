# 01. 기초와 수식

표준 residual은 $h_l=h_0+\sum_{i<l}F_i(h_i)$처럼 과거 출력을 균일하게 더한다. PreNorm에서는 이 합의 크기가 depth와 함께 커져 후반 layer의 상대적 변화가 작아지는 dilution이 생길 수 있다.

AttnRes는 이전 value $v_i$를 정규화해 key $k_i$를 만들고 layer별 pseudo-query $w_l$로 logit을 계산한다.

$$s_{i,l}=w_l^Tk_i,\quad \alpha_{i\to l}=\operatorname{softmax}_i(s_{i,l}),\quad h_l=\sum_i\alpha_{i\to l}v_i$$

softmax 가중치 합은 1이므로 단순 합과 달리 출력 크기를 제어한다. pseudo-query는 token마다 새 query를 만드는 것이 아니라 layer별 학습 parameter이며, key가 입력 표현에 의존하므로 최종 가중치는 token·입력별로 달라진다.

실습에서는 [toy example](examples/attnres_toy.py)의 `attn_res`를 읽고 query를 바꾸었을 때 선택되는 depth가 어떻게 달라지는지 확인한다.
