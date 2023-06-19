from __future__ import division, absolute_import, print_function
import argparse
import pandas as pd
from common.util import *
from setup_paths import *
os.environ["NPY_NUM_ARRAY_FUNCTION_ARGUMENTS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fs.datasets.datasets_utils import *
from fs.utils.squeeze import *
from fs.utils.output import write_to_csv
from fs.robustness import evaluate_robustness
from fs.detections.base import DetectionEvaluator, evalulate_detection_test, get_tpr_fpr

act_set=[]
# from tensorflow.python.platform import flags
# FLAGS = flags.FLAGS
# flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')

def get_distance(model, dataset, X1):
    X1_pred = model.predict(X1)
    vals_squeezed = []

    if dataset == 'mnist':
        X1_seqeezed_bit = bit_depth_py(X1, 1)  # 位深度压缩~四舍五入
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)  # 中值滤波
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
    else:
        X1_seqeezed_bit = bit_depth_py(X1, 5)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
        X1_seqeezed_filter_local = non_local_means_color_py(X1, 13, 3, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_local))
    #除了第一个维度（即X1_pred中样本数量的维度）以外的所有维度对差进行求和
    dist_array = []
    for val_squeezed in vals_squeezed:
        dist = np.sum(np.abs(X1_pred - val_squeezed), axis=tuple(range(len(X1_pred.shape))[1:]))
        dist_array.append(dist)
    dist_array = np.array(dist_array)
    return np.max(dist_array, axis=0)

#train_fpr该参数表示要保留多少样本（以假阳性率为单位），以便用于特征选择
#如果train_fpr=0.1，则函数将保留前10%的样本，即具有最小距离的10%的样本将被用于特征选择
def train_fs(model, dataset, X1, train_fpr):
    distances = get_distance(model, dataset, X1)
    selected_distance_idx = int(np.ceil(len(X1) * (1 - train_fpr)))
    threshold = sorted(distances)[selected_distance_idx - 1]
    threshold = threshold
    #选择与预测值之间距离阈值
    print("Threshold value: %f" % threshold)
    return threshold

#将距离大于阈值的样本标记为正类（True），将距离小于或等于阈值的样本标记为负类（False）
def test(model, dataset, X, threshold):
    distances = get_distance(model, dataset, X)
    Y_pred = distances > threshold
    return Y_pred, distances


def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.ecnn.ecnn_mnist import MNISTECNN as myModel
        model_class = myModel(mode='load', filename='ecnn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as myCNNModel
        model_class_cnn = myCNNModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model_cnn = model_class_cnn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_cnn.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        # model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), 
        #       loss='CategoricalCrossentropy',
        #       metrics=['accuracy'])

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model = model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    # Load the dataset
    X_train_all, Y_train_all, X_test_all, Y_test_all = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test
    # --------------
    # Evaluate the trained model.
    # Refine the normal and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    #用cnn抽取正确数据集
    # print("Evaluating the pre-trained model...")
    Y_pred_all = model.predict(X_test_all)
    # accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    # print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    #代码从预测结果中筛选出原始版本被正确分类的样本
    # 并将这些样本存储在X_test和Y_test数组中
    # 而且也存储它们的预测结果在Y_pred数组中
    inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
    X_test = X_test_all[inds_correct]
    Y_test = Y_test_all[inds_correct]
    Y_pred = Y_pred_all[inds_correct]
    #random.sample函数从X_test和Y_test中随机选择一些样本作为特征选择器的训练集
    indx_train = random.sample(range(len(X_test)), int(len(X_test) / 2))
    #list(set())函数将这些样本从测试集中移除
    indx_test = list(set(range(0, len(X_test))) - set(indx_train))
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    x_train = X_test[indx_train]
    y_train = Y_test[indx_train]
    x_test = X_test[indx_test]
    y_test = Y_test[indx_test]
    #print(x_train)
    # compute thresold - use test data to compute that
    threshold = train_fs(model, args.dataset, x_train, 0.05)
    Y_test_copy = Y_test
    X_test_copy = X_test
    y_test_copy = y_test
    x_test_copy = x_test
    ## Evaluate detector
    # on adversarial attack
    for attack in ATTACKS:
        df_all = pd.DataFrame()
        df_success = pd.DataFrame()
        df_fail = pd.DataFrame()
        Y_test = Y_test_copy
        X_test = X_test_copy
        y_test = y_test_copy
        x_test = x_test_copy
        results_all = []

        # Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))
        #reduce_precision_py将图像的精度降低
        X_test_adv = reduce_precision_py(X_test_adv, 256)
        #根据不同的攻击类型和数据集来选择测试数据
        if attack == 'df' and args.dataset == 'tiny':
            Y_test = model_class.y_test[0:2700]
            X_test = model_class.x_test[0:2700]
            cwi_inds = inds_correct[inds_correct < 2700]
            Y_test = Y_test[cwi_inds]
            X_test = X_test[cwi_inds]
            X_test_adv = X_test_adv[cwi_inds]
            xtest_inds = np.asarray(indx_test)[np.asarray(indx_test) < 2700]
            xtest_inds = np.in1d(cwi_inds, xtest_inds)
            x_test = X_test[xtest_inds]
            y_test = Y_test[xtest_inds]
            X_test_adv = X_test_adv[xtest_inds]
        else:
            X_test_adv = X_test_adv[inds_correct]
            X_test_adv = X_test_adv[indx_test]
        #评估模型在对抗样本上的性能，并将成功和失败的样本分别存储在不同的数组中
        #loss, acc_suc = model.evaluate(X_test_adv, y_test, verbose=0)
        X_test_adv_pred = model.predict(X_test_adv)
        print(X_test_adv_pred)
        #print(acc_suc)
        inds_success = np.where(X_test_adv_pred.argmax(axis=1) != y_test.argmax(axis=1))[0]
        inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == y_test.argmax(axis=1))[0]
        # inds_all_not_fail = list(set(range(0, len(inds_correct)))-set(inds_fail))
        X_test_adv_success = X_test_adv[inds_success]
        Y_test_success = y_test[inds_success]
        X_test_adv_fail = X_test_adv[inds_fail]
        Y_test_fail = y_test[inds_fail]
        #准备用于检测器的数据集，将原始测试集和对抗样本合并，并将它们标记为成功分类或失败分类
        # prepare X and Y for detectors
        X_all = np.concatenate([x_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(x_test), dtype=bool), np.ones(len(x_test), dtype=bool)])
        X_success = np.concatenate([x_test[inds_success], X_test_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([x_test[inds_fail], X_test_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        # for Y_all
        # if attack == ATTACKS[0]:
        #测试并评估检测器在整个数据集上的性能
        Y_all_pred, Y_all_pred_score = test(model, args.dataset, X_all, threshold)
        print(Y_all_pred)
        print(Y_all_pred_score)
        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        print(acc_all)
        
        # for Y_success
        #测试并评估检测器在对抗样本成功分类的样本上的性能
        if len(inds_success) == 0:
            accuracy_success=0.0
        else:
            Y_success_pred, Y_success_pred_score = test(model, args.dataset, X_success, threshold)
            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(
                Y_success, Y_success_pred)
            print(accuracy_success)
        # for Y_fail
         #测试并评估检测器在对抗样本失败分类的样本上的性能
        if len(inds_fail) == 0:
            accuracy_fail=0.0
        else:
            Y_fail_pred, Y_fail_pred_score = test(model, args.dataset, X_fail, threshold)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail,
                                                                                                            Y_fail_pred)
        print(accuracy_fail)
        df_all = pd.DataFrame({'Attack': attack, 'Threshold': threshold , 'Accuracy': acc_all}, index=[0])
        df_success = pd.DataFrame({'Attack': attack, 'Threshold': threshold , 'Accuracy': accuracy_success}, index=[0])
        df_fail = pd.DataFrame({'Attack': attack, 'Threshold': threshold,  'Accuracy': accuracy_fail}, index=[0])
         # Concatenate new dataframe with existing dataframe
        df_all.to_csv('fs_enn_mnist_results_all.csv', index=False, mode='a', header=False)
        df_success.to_csv('fs_enn_mnist_results_success.csv', index=False, mode='a', header=False)
        df_fail.to_csv('fs_enn_mnist_results_fail.csv', index=False, mode='a', header=False)

        
    print('Done!')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)
