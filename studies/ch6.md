# Ch6. 深層強化学習の発展版を実装しよう

## 深層強化学習のアルゴリズムマップ

- DDQN
    - Double-Q学習+DQN
        - (おさらい)Fixed Target Networkを用いた学習
            - $Q_m(s_t, a_t) = Q_m(s_t, a_t) + \eta*(R_{t+1}+\gamma\max_aQ_t(s_{t+1}, a)-Q_m(s_t, a_t))$
            - ($Q_m$ : main network, $Q_t$ : target network)
        - DDQNでは、TD誤差算出の行動選択にて、target netoworkではなくmain networkを使うことで、学習より安定させる
            - $a_m=\argmax_a{Q_m(s_{t+1}, a)}$
            - $Q_m(s_t, a_t)=Q_m(s_t,a_t)+\eta*(R_{t+1}+\gamma Q_t(s_{t+1}, a_m)-Q_m(s_t, a_t))$
    - ポイント：TD誤差算出に、main networkとtaget networkの二種類を使うことで、学習をより安定化させられる。
- Dueling Network
    - 行動価値関数の出力層の前に、行動に依存しない値を出力する層を設ける
        - 状態価値 $V(s)$ (最大の行動価値ではなく、平均の行動価値)
        - アドバンテージ関数 $A(s,a) =Q(s,a)-V(s)$
    - ポイント：報酬を、状態に依存する値と行動に依存する値の組み合わせととらえ、状態に依存する値を行動に依存せず直接学習することで、少ない試行回数で学習できる
        - 例：倒立振子にて、振子が倒れる寸前で行動選択するとき
            - 行動によらず、倒れてしまう
- Prioritized Experience Replay
    - Experience Replayで、学習対象の複数ステップを選択する際、ランダムに選択するのではなく、優先順位付けして選択する。
        - TD誤差の絶対値が大きいものを選択
        - 理由：TD誤差の絶対値が大きいほど、行動価値関数の学習が進んでいないと考えられるため
    - ポイント：学習性能が上がる
- A3C(Asynchronous Advantage Actor-Critic)
    - 3つの工夫により、高性能な深層強化学習を実現
        - Asynchronous : 複数のエージェントを用意して学習させる
        - Advantage : 2ステップ以上後の状態を利用して、行動価値関数を更新する
            - 例：2ステップを考慮した行動価値関数の更新式
                - $Q_(s_t, a_t)=R_{t+1}+\gamma R_{t+2} + \gamma^2 \max_a Q(s_{t+2}, a)$
            - ステップ数が大きすぎると、誤った行動を選択し、誤った学習をする危険性があるため、ステップ数は適切な値とする。
        - Actor-Critic : 方策選択と価値算出に一つの行動価値関数を使うのではなく、アドバンテージ関数と状態価値関数という別の関数を使う
            - ネットワーク構成(アドバンテージ関数と状態価値関数でネットワークを共有する場合)
                - 入力：$s_t$
                - 出力：$A(s_t, a)$, $V(s)$
            - 誤差関数の定義
                - アドバンテージ関数の学習：評価関数とエントロピーの加算値を最大化
                    - 評価関数：$J(\theta, s_t) = E[\log{\pi_\theta(a|s)(Q^{\pi}(s,a)-V(s))}]$
                    - エントロピー：$Actor_{entoropy}=\Sigma^a[\pi_{\theta}(a|s)\log{\pi_{\theta}(a|s)}]$
                - 状態価値関数の学習：誤差の最小化
                    - $loss_{critic}=(Q^{\pi}(s,a)-V(s))$
                    - 行動価値と状態価値との差を誤差としている理由：行動が一意に定まる場合、行動価値と状態価値が同じ値になるから？
    - ポイント
        - 複数エージェントの利用を前提としているため、現実のロボットなどに適用しやすい
        - 複数エージェントが独自のステップを生み出すため、Experience Replayをしなくても、安定して学習できる
        - Experience Replayしなくてよいため、LSTMなどを使える

## DDQNの実装(省略)

## Dueling Networkの実装(省略)

## Prioritized Experience Replayの実装

## A2Cの実装