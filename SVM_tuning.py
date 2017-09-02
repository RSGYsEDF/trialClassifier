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

from sklearn import svm, metrics, preprocessing, model_selection
import matplotlib.pyplot as plt

##学習データのディレクトリ

#__name__という変数はPythonプリンタでPythonスクリプトを読み込むと自動的に作成されます。
#さらに、この中には実行しているスクリプトのモジュール名が自動的に代入されるようになっています。
#Pythonスクリプトを直接実行した時には、そのスクリプトファイルは「__main__」という名前のモジュールとして認識されます。
#そのため、スクリプトファイルを直接実行すると__name__変数の中に自動で__main__という値が代入されるのです。
#つまり、「if __name__ == ‘__main__’:」の意味は、「直接実行された場合のみ実行し、それ以外の場合は実行しない」という意味だったのです。
if __name__ == '__main__':

    #sys.argvは、コマンドラインで渡される引数が入ったリスト。sys.argv[0]にはスクリプト自身が入る。
    argvs = sys.argv
    data_path = argvs[1]
    
    #データ読み込み
    dataSet = load_handImages(data_path)

    #スケーリング
    dataSet.data = preprocessing.scale(dataSet.data)

    #手法・線形SVM
    svc = svm.LinearSVC()

    result = model_selection.cross_val_score(estimator = svc, Ｘ = dataSet.data, ｙ = dataSet.ｔａｒｇｅｔ)
    print(result.mean())
#
#    for train_index, test_index in kf.split(dataSet):
#        print(train_index.target)
#        print(test_index.target)
##        svc.fit(train_index, target[train])
#        #        X_train, X_test = X[train_index], X[test_index]
##        y_train, y_test = y[train_index], y[test_index]
##        svc.fit(data[train], target[train])



#
#    accuracy = (sum(scores) / len(scores)) * 100
#    print(accuracy)
#    msg = '正答率: {accuracy:.2f}%'.format(accuracy=accuracy)
#    print(msg)


#    i = 0
#    for C in C_list:
#        for train, test in k_fold:
#            svc.fit(data[train], target[train])
#            tmp_train.append(svc.score(data[train],target[train]))
#            tmp_test.append(svc.score(data[test],target[test]))
#            score[i,0] = C
#            score[i,1] = sum(tmp_train) / len(tmp_train)
#            score[i,2] = sum(tmp_test) / len(tmp_test)
#            del tmp_train[:]
#            del tmp_test[:]
#        i = i + 1
#
#    xmin, xmax = score[:,0].min(), score[:,0].max()
#    ymin, ymax = score[:,1:2].min()-0.1, score[:,1:2].max()+0.1
#    plt.semilogx(score[:,0], score[:,1], c = "r", label = "train")
#    plt.semilogx(score[:,0], score[:,2], c = "b", label = "test")
#    plt.axis([xmin,xmax,ymin,ymax])
#    plt.legend(loc='upper left')
#    plt.xlabel('C')
#    plt.ylabel('score')
#    plt.show



#    for idx, file in enumerate(predicted):
#        if file == test.target[idx]:
#            
#            # Pythonは、文字列と文字列を連結して新しい文字列を作成することは出来ますが、文字列と数値を連結することは出来ません。
#            print("正解/ " + test.fileName[idx] + "予測" + str(file) + "答え：" + str(test.target[idx]))
#        else:
#            print("不正解/ " + test.fileName[idx] + "予測：" + str(file) + "答え：" + str(test.target[idx]))
#
#    #結果表示
#    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
#    print(metrics.precision_score(test.target, predicted))
#    print(metrics.recall_score(test.target, predicted))

    # 最終的な正答率を出す
#    accuracy = (sum(scores) / len(scores)) * 100
#    msg = '正答率: {accuracy:.2f}%'.format(accuracy=accuracy)
#    print(msg)

#実行は、 ./data/dataset