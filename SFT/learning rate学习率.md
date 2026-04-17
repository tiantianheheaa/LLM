

- 一个常见的设置：warmup+cosine
  - warmup_ratio表示总steps的热身占比。会线性从0增加到lr的设定值/最大值。 【有利于开始模型训练的稳定性】
  - 到达lr最大值后`cosine衰减`到lr最小值。【有利于模型逐步收敛】
  - 总体趋势：先线性增加到最大值。然后cos衰减。
<img width="1448" height="1532" alt="image" src="https://github.com/user-attachments/assets/9b79c638-5019-4c80-a52e-c405ef7017da" />

<img width="1514" height="816" alt="image" src="https://github.com/user-attachments/assets/d3fd338b-ee63-4a1e-82c7-14802d5e5e5d" />

