# 02. 구현과 실험

원본 의사코드의 `V` shape은 `[N+1,B,T,D]`이며 norm 뒤 `einsum`으로 `[N+1,B,T]` logit을 만든다. softmax는 첫 축, 즉 depth/block 후보 축에 적용한다. weighted sum 결과는 `[B,T,D]`다.

Full AttnRes는 모든 이전 layer 출력을 보관해 O(Ld) cache를 요구한다. Block AttnRes는 block 내부 출력을 표준 residual로 합치고 완료된 block 대표와 현재 partial sum만 후보로 사용해 O(Nd)로 낮춘다. block boundary에서 partial sum을 먼저 완료 목록에 옮기고 새 partial을 시작하는 순서가 중요하다.

검증 항목:

1. weight가 음수가 아니고 후보 축 합이 1인지 검사한다.
2. 후보가 하나면 출력이 그 value와 같은지 검사한다.
3. 동일 logit이면 산술 평균이 되는지 검사한다.
4. batch·sequence·hidden shape이 보존되는지 검사한다.
5. standard, Full, Block을 동일 seed·parameter budget에서 비교한다.
