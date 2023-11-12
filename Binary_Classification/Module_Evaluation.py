from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score


def Evaluation(test_Y, prediction):

    report = classification_report(test_Y, prediction, output_dict = True)

    # report['0']['precision']*100 : 음성이라고 판단한 것 중 실제로 음성
    # report['0']['recall']*100 : 음성인것 중 음성이라고 판단한 것
    # 0-> 1 이면 음성 -> 양성
    result = {}
    result['정상'] = report['0']['f1-score']
    result['고장'] = report['1']['f1-score']
    result['macro avg'] = report['macro avg']['f1-score']
    result['weighted avg'] = report['weighted avg']['f1-score']
    result['accuracy'] = report['accuracy']
    result['AUC'] = roc_auc_score(test_Y, prediction)
    result['confusion matrix'] = confusion_matrix(test_Y, prediction).ravel()
    result['F2 score'] = fbeta_score(test_Y, prediction, beta = 2)
    return result

# score = {'0': {'precision': 0.8901098901098901, 'recall': 0.9204545454545454, 'f1-score': 0.9050279329608938, 'support': 176}, \
#     '1': {'precision': 0.2222222222222222, 'recall': 0.16666666666666666, 'f1-score': 0.1904761904761905, 'support': 24},\
#          'accuracy': 0.83, 'macro avg': {'precision': 0.5561660561660562, 'recall': 0.5435606060606061, 'f1-score': 0.5477520617185422, 'support': 200}, \
#              'weighted avg': {'precision': 0.8099633699633699, 'recall': 0.83, 'f1-score': 0.8192817238627295, 'support': 200}}

def Plot_ROC(test_Y, proba, method):
    # ROC 커브를 그려줌
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_Y, proba)
    from matplotlib import pyplot as plt
    print(method)

    plt.plot(fpr, tpr, label = 'ROC')
    plt.plot([0,1], [0,1], 'k--', label = '50%')
    # plt.show()
    plt.savefig("C:/Users/user/Desktop/ROC/" + str(method))
    plt.close()

def Plot_Recall_Precision(test_Y, pred_proba, method):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precisions, recalls, thresholds = precision_recall_curve(test_Y, pred_proba)
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle = '--', label = 'precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.savefig("C:/Users/user/Desktop/Precision_Recall/" + str(method))
    plt.close()
    

def Save(result):
    # 결과값을 엑셀에 저장
    import pandas as pd

    df = pd.DataFrame(result)
    df = df.round(3)
    df.to_excel("BC_Result.xlsx")