from collections import Counter
import numpy as np

def predict_class_audio(MFCCs, model):
    '''
    MFCC 샘플을 기반으로 클래스 예측
    * param MFCCs: MFCC의 Numpy 배열
    * param model: 훈련된 모델
    * return: MFCC 세그먼트 그룹의 예측 된 클래스
    '''
    #MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    #y_predicted = model.predict_classes(MFCCs,verbose=0)
    try:
        MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
        y_predicted = model.predict_classes(MFCCs, verbose=0)
    except:
        return None
    return(Counter(list(y_predicted)).most_common(1)[0][0])


def predict_prob_class_audio(MFCCs, model):
    '''
    MFCC 샘플의 확률을 기반으로 클래스 예측
    * param MFCCs: MFCC의 Numpy 배열
    * param model: 훈련된 모델
    * return: MFCC 세그먼트 그룹의 예측 된 클래스
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_proba(MFCCs,verbose=0)
    return(np.argmax(np.sum(y_predicted,axis=0)))

def predict_class_all(X_train, model):
    '''
     * param X_train: 세그먼트 화 된 mfcc 목록
    * param model: 훈련된 모델
    * return: 예측 목록
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    '''
    confusion matrix 생성
    * param y_predicted: 예측 목록
    * param y_test: shape의 numpy 배열 (len (y_test), 클래스 수).
        -  실제 인덱스에 있으면1,  그렇지 않으면 0
    * return: numpy 배열. confusion matrix
    '''
    confusion_matrix = np.zeros((len(y_test[0]),len(y_test[0])),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def get_accuracy(y_predicted,y_test):
    '''
    accuracy 얻기
    * param y_predicted: numpy 예측 배열
    * param y_test: 실제 숫자 배열
    * return: accuracy
    '''
    c_matrix = confusion_matrix(y_predicted,y_test)
    return( np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))

if __name__ == '__main__':
    pass


