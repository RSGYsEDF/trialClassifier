# -*- coding: utf_8 -*-

#このモジュールには、パス名を操作する便利な関数が実装されています。
import os
import sys
#glob モジュールは Unix シェルで使われているルールに従って指定されたパターンにマッチするすべてのパス名を見つけ出します。
import glob
import numpy as np
from skimage import io
from sklearn import datasets

IMAGE_SIZE = 50
COLOR_BYTE = 4
CATEGORY_NUM = 2

## ラベル名(0～)を付けたディレクトリに分類されたイメージファイルを読み込む
## 入力パスはラベル名の上位のディレクトリ
def load_handImages(path):
    
    #ファイル一覧を取得 引数にマッチする空の可能性のあるパス名のリストを返します。
    #os.path.join("c:", "foo") によって、 c:\foo ではなく、ドライブ C: 上のカレントディレクトリからの相対パス(c:foo) が返されることに注意してください。
    files = glob.glob(os.path.join(path, '*/*.png'))

    
    #イメージとラベル領域確保
    #np.ndarray は， N-d Array すなわち，N次元配列を扱うためのクラス dtype = data-type
    #ndarray最初の引数、shapeは次元数と各次元の要素数を表す？　次元数は、4つ。
    images = np.ndarray((len(files),IMAGE_SIZE,IMAGE_SIZE,COLOR_BYTE), dtype=np.uint8)
    labels = np.ndarray(len(files), dtype=np.int)
    fileNames = list()
    #イメージとラベルを読み込み
    #enumerate()は引数の要素とインデックス値を返してくれる
    for idx, file in enumerate(files):
        #イメージ読み込み
        image = io.imread(file)
        images[idx] = image
        fileNames.append(os.path.basename(file))

        #ディレクトリ名よりラベルを取得

        #パス名 path を (head, tail) のペアに分割します。 tail はパス名の構成要素の末尾で、 head はそれより前の部分です。
        #配列[-1]は、最後の要素を取り出している。
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)
        
    #scikit-learnのほかのデータセットの形式にあわせる
    #配列の総要素数が不明の場合は，大きさが不明な次元で -1 を指定すると適切な値が自動的に設定されます．
    flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE))
    images = flat_data.view()
    return datasets.base.Bunch(data = flat_data,
                               target = labels.astype(np.int),
                               target_name = np.arange(CATEGORY_NUM),
                               images=images,
                               fileName = fileNames,
                               DESCR=None)

######################################################

from sklearn import svm, metrics
from sklearn import tree

##学習データのディレクトリ

#__name__という変数はPythonプリンタでPythonスクリプトを読み込むと自動的に作成されます。
#さらに、この中には実行しているスクリプトのモジュール名が自動的に代入されるようになっています。
#Pythonスクリプトを直接実行した時には、そのスクリプトファイルは「__main__」という名前のモジュールとして認識されます。
#そのため、スクリプトファイルを直接実行すると__name__変数の中に自動で__main__という値が代入されるのです。
#つまり、「if __name__ == ‘__main__’:」の意味は、「直接実行された場合のみ実行し、それ以外の場合は実行しない」という意味だったのです。
if __name__ == '__main__':

    #sys.argvは、コマンドラインで渡される引数が入ったリスト。sys.argv[0]にはスクリプト自身が入る。
    argvs = sys.argv
    train_path = argvs[1]
    test_path = argvs[2]
    
    #学習データ読み込み
    train = load_handImages(train_path)
    
    #手法・線形SVM
    classifier = svm.LinearSVC()
    
    #学習
    classifier.fit(train.data, train.target)
    
    #テストデータ読み込み
    test = load_handImages(test_path)

    #テスト
    #predictで予測ラベルを取得。
    predicted = classifier.predict(test.data)

    for idx, file in enumerate(predicted):
        if file == test.target[idx]:
            
            # Pythonは、文字列と文字列を連結して新しい文字列を作成することは出来ますが、文字列と数値を連結することは出来ません。
            print("正解/ " + test.fileName[idx] + "予測" + str(file) + "答え：" + str(test.target[idx]))
        else:
            print("不正解/ " + test.fileName[idx] + "予測：" + str(file) + "答え：" + str(test.target[idx]))

    #結果表示
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
    print(metrics.precision_score(test.target, predicted))
    print(metrics.recall_score(test.target, predicted))

#実行は、 ./data/learn_chara ./data/test_chara
