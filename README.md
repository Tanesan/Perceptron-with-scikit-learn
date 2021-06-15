# Perceptron with scikit learn
 - StandardScaler(preprocessing) 特徴量標準化  
# 分類ステップ
 1. 特徴量を選択し、ラベル付けされた訓練データを収集
 2. 性能指標を選択する
 3. 分類器と最適化アルゴリズムを選択する
 4. モデルの性能を評価する
 5. アルゴリズムを調整する

# main.py
pythonファイルをそのまま貼り付ける。コメントアウトが結構良さそう。
### iris データセットロード
### 花びらの長さと幅
``` python
iris = datasets.load_iris()
```
### 3,4列目を抽出
``` python
X = iris.data[:, [2, 3]]
```
### クラスラベルを取得
``` python
y = iris.target
```
一意なクラスラベルを出力　np.unique(y)
### scikit learnのtestとtrainを分ける関数。　test_sizeで30%をテストデータとして扱う。random_stateでランダムに分割。
内部ではシャッフルして分割している
staratify = yは層化サンプリング。層化抽出。クラスラベル(0, 1, 2)の出現数がすべて一緒になるように設定される。 np.bitcountで確認可能
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
```
### 値を標準化。ex) がく　5枚　重さ2000mgだとすると、重さの方が数値が大きいので、そっちが重要と勘違いする。
``` python
sc = StandardScaler()
```
### trainデータを元に標準化
``` python
sc.fit(X_train)
```
### 標準化
``` python
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```
# one vs rest (一対他手法） 
直接多クラスの分類モデルを構築するのではなく、2値分類を行う分類器をいくつか組み合わせることで多クラス分類を行う
3つの品種のクラスをパーセプトロンに同時に対応することができる。
epoc 40 学習率　0.1のパーセプトロンのインスタンス
``` python
ppn = Perceptron(eta0=0.1, random_state=1)
```
### トレーニングデータを用いて、学習
``` python
ppn.fit(X_train_std, y_train)
```
### テストデータで予測
``` python
y_pred = ppn.predict(X_test_std)
```
### 語分類の数
``` python
# print('誤分類', (y_test != y_pred).sum())
```
### 分類の正解率を表示
``` python
print('Accuracy:', accuracy_score(y_test, y_pred))
```

ppn.score(X_test_std, y_test)
でも可能

### ここはperceptronの時と同じ
``` python
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    # ここまでは前と同じ
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        # テストデータをわかりやすく
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
```
### ここから
``` python
X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('patel length [std]')
plt.ylabel('patel width [std]')
# 左上に詳細、マークの説明
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```
