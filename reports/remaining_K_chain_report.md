# remaining_K chain report

Scope: previous state/boundary-layer analysesと同じく、long-word かつ final state が定義できる語を対象にした。

## Mass placement

ALL で最大の `|delta|` は `32-63`。

- `32-63`: actual `1.959532`, iid `2.139743`, delta `-0.180211`, ratio `0.916`, L1 share `38.57%`
- `64-95`: actual `0.266435`, iid `0.341662`, delta `-0.075227`, ratio `0.780`, L1 share `16.10%`
- `96-127`: actual `0.018644`, iid `0.031704`, delta `-0.013059`, ratio `0.588`, L1 share `2.80%`

上流側にも薄さは続くが、質量差の中心は `32-63` から逃げ切っていない。`96-127` は ratio は最も低いが、mass と L1 share は小さい。

この母集団では `128-191` と `192+` の bin は観測されなかった。したがって今回の chain は `96-127` で上端に達する。

## Downstream transitions

主要な下流遷移:

- `96-127 -> 64-95`: delta `-0.002017`, ratio `0.620`, conditional actual `0.176505`, conditional iid `0.167431`, conditional_delta `+0.009074`
- `64-95 -> 32-63`: delta `-0.006059`, ratio `0.836`, conditional actual `0.116249`, conditional iid `0.108388`, conditional_delta `+0.007861`
- `32-63 -> 16-31`: delta `-0.009262`, ratio `0.957`, conditional actual `0.106222`, conditional iid `0.101604`, conditional_delta `+0.004618`
- `16-31 -> 8-15`: delta `-0.008947`, ratio `0.973`, conditional actual `0.104883`, conditional iid `0.104666`, conditional_delta `+0.000217`
- `8-15 -> 4-7`: delta `-0.008947`, ratio `0.973`, conditional actual `0.177130`, conditional iid `0.177386`, conditional_delta `-0.000257`
- `4-7 -> 2-3`: delta `-0.008947`, ratio `0.973`, conditional actual `0.323621`, conditional iid `0.320795`, conditional_delta `+0.002826`
- `2-3 -> 0-1`: delta `-0.008947`, ratio `0.973`, conditional actual `0.568177`, conditional iid `0.570519`, conditional_delta `-0.002342`

`64-95` と同じ現象が `96-127` と `32-63` でも出ている。bin mass は actual が薄いが、下流へ落ちる条件付き確率は actual の方が高い。

## Mass delta vs conditional transition delta

mass delta と最大 conditional transition delta が逆符号になる bin:

- `96-127`: mass delta `-0.013059`, largest conditional `96-127 -> 64-95`, conditional_delta `+0.009074`
- `64-95`: mass delta `-0.075227`, largest conditional `64-95 -> 32-63`, conditional_delta `+0.007861`
- `32-63`: mass delta `-0.180211`, largest conditional `32-63 -> 16-31`, conditional_delta `+0.004618`
- `16-31`: mass delta `-0.091890`, largest conditional `16-31 -> 8-15`, conditional_delta `+0.000217`
- `2-3`: mass delta `-0.013338`, largest conditional `2-3 -> 2-3`, conditional_delta `+0.002342`

これは、actual がその bin にいる質量は薄い一方で、そこにいる条件では下流側へ進む比率が弱いとは限らない、という分離を示す。

## Focus states

- `late_growth|deep_32_63|even`: 最大 `|delta|` は `32-63`, delta `-0.056670`, ratio `0.820`, L1 share `41.40%`
- `late_growth|deep_32_63|odd`: 最大 `|delta|` は `16-31`, delta `+0.004322`, ratio `1.035`, L1 share `25.75%`
- `late_growth|exhaustion_0_31|odd`: 最大 `|delta|` は `16-31`, delta `+0.017864`, ratio `1.047`, L1 share `44.13%`

focus state では全体と止まり方が違う。`late_growth|deep_32_63|even` は全体と同じく `32-63` の deficit が濃い。一方で odd 側と exhaustion 側は `16-31` の positive delta が最大になり、全体の negative chain とは符号がずれる。

## Short reading

1. actual の薄さは `64-95` から `96-127` へは続く。ただし最大の質量差は `32-63` に残る。
2. 最大 `|delta|` は `32-63`。最低 ratio は `96-127`。
3. `32-63`, `64-95`, `96-127` では、mass delta は negative だが、主要な下流 conditional_delta は positive。
4. これは単一の発生点というより、remaining_K chain 上の質量配置差として読むのが自然。
5. ただし focus state では `late_growth|deep_32_63|even` だけが全体の deficit と強く対応し、odd/exhaustion 側は `16-31` で positive に寄る。
6. ここで boundary/upstream 掘削は一区切りでよい。次は論旨接続に戻り、state-level delta、boundary remaining_K、conditional transition の三者を同じ記述にまとめるのがよい。

## Outputs

- `remaining_K_chain_mass.csv`
- `remaining_K_chain_transition.csv`
- `remaining_K_chain_conditional_transition.csv`
- `remaining_K_chain_mass_vs_conditional.csv`
- `remaining_K_chain_by_focus_state.csv`
- `remaining_K_chain_focus_state_max_bin.csv`
- `remaining_K_chain_delta.svg`
- `remaining_K_chain_ratio.svg`
- `remaining_K_chain_l1_share.svg`
- `remaining_K_chain_transition_heatmap.svg`
- `remaining_K_chain_focus_state_heatmap.svg`
