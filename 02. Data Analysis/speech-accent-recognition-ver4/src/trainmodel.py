# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
import sys
sys.path.append('../speech-accent-recognition/src>')


import getsplit
from keras import utils
import accuracy
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard


DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10 #35#250


def to_categorical(y):
    '''
    원핫인코딩
    * param y (list): 언어 목록
    * return (numpy array): 이진 클래스 배열
        - set(y) : 집합형 자료형 만들기
        - enumerate : 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate객체로 돌려준다.
        -   filter 함수는 첫 번째 인수로 함수 이름을, 두 번째 인수로 그 함수에 차례로 들어갈 반복 가능한 자료형을 받는다.
            그리고 두 번째 인수인 반복 가능한 자료형 요소가 첫 번째 인수인 함수에 입력되었을 때 반환 값이 참인 것만 묶어서(걸러 내서) 돌려준다.
        - list(map(lambda x: lang_dict[x],y))
             lasmbda(쓰고 버리는 일시적 함수) : leang_dict[x],y에 대한 map(2개의 인수를 갖음)을 생성하고 리스트형태로 y에 값을 덮어 씌우기
                a = [1,2,3,4]
                b = [17,12,11,10]
                list(map(lambda x, y:x+y, a,b))
                    [18, 14, 14, 14]
        - utils.to_categorical :utils.to_categorical(입력 클래스 벡터, 총 클래스 수)
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))


def get_wav(language_num):
    '''
    CSV파일에 불러온 wav을 정의해놓은 RATE로 리샘플링
    * param language_num (list): 로딩한 CSV파일의 파일 리스트를 정의한 컬럼
    * return (numpy array): 리샘플링된 WAV파일
        - y : 오디오 시계열 데이터(모노/스테레오)
        - orig_sr : y의 원래 샘플링 속도
        - target_sr : 목포 샘플링 속도
        - scale : 스케일 여부
    '''
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))


def to_mfcc(wav):
    '''
    wav 파일을 MFCC(Mel Frequency Ceptral Coefficients)로 변환
    * param wav (numpy array): Wav 형태
    * return (2D numpy array): 리샘플링된 WAV파일
        - y : 오디오 시계열 데이터(모노/스테레오)
        - sr : 샘플링 속도
        - n_mfcc : 반환할 MFCC 수
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))


# 사용 안함 ###########################################################################################################
# def remove_silence(wav, thresh=0.04, chunk=5000):
#     '''
#     wav의 무음 부분 제거
#     * return : 무음이 제거된 wav 반환
#         - 만약에 [5000*1 : 5000(1+1)]가 >= 0.04이상이거나, [5000 * 1 : 5000(1+1)]가 <= -0.04이하이면
#             tf_list의 True 컬럼에 5000을 곱해라.
#             tf_list.extend([True] * chunk)는 tf_list += [True] * chunk와 동일한 결과
#             tf_list.extend([False] * chunk)는 tf_list += [False] * chunk와 동일한 결과
#         - (wav의 수 - tf_list의 수) * [False]를 추가
#     '''
#
#     tf_list = []
#     for x in range(len(wav) / chunk):
#         if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
#             tf_list.extend([True] * chunk)
#         else:
#             tf_list.extend([False] * chunk)
#
#     tf_list.extend((len(wav) - len(tf_list)) * [False])
#     return(wav[tf_list])
# 사용 안함 ###########################################################################################################


# 선택적 사용 ##########################################################################################################
def normalize_mfcc(mfcc):
    '''
    mfcc 정규화
    * param mfcc
    * return : 정규화된 mfcc
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))
# 선택적 사용 ##########################################################################################################


def make_segments(mfccs,labels):
    '''
    mfcc 세그먼트를 만들어 레이블을 붙임
    * param mfccs: mfcc 목록
    * param labels: labels 목록
    * return (tuple):  labels가 붙어진 Segments
        - zip(mfccs, leables)
            zip : 동일한 개수로 이루어진 자료형을 묶어 주는 역할을 하는 함수
                list(zip([1, 2, 3], [4, 5, 6]))
                >> [(1, 4), (2, 5), (3, 6)]
        -  range(0, int(mfcc.shape[1] / COL_SIZE))
             mfcc의 열의 개수 / 13
        - mfcc[:, start * COL_SIZE : (start + 1) * COL_SIZE])을 segments 리스트에 추가
            mfcc[:,     :]
            mfcc[:,     start~ * 13 : start+1 * 13]
        - seg_labels에 불러온 요소인 leabel을 붙임.
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)


def segment_one(mfcc):
    '''
    만약 마지막 세그먼트가 열길이가 COL_SIZE로 나눌 만큼 길지 않으면, mfcc 이미지에서 세그먼트를 만듬.
    * param mfcc (numpy array): MFCC 배열
    * return (numpy array): Segmented MFCC 배열
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    X_train에서 세그먼트 화 된 MFCC를 작성합니다
    * param X_train: MFCC 목록
    * return: mfcc 세그먼트
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def train_model(X_train,y_train,X_validation,y_validation, batch_size=128): #64
    '''
    2D 컨볼 루션 신경망 훈련
    * param X_train: mfccs의 numpy 배열
    * param y_train: 레이블 기반 이진 행렬
    * return: 훈련된 모델
    '''

    # 행, 열 및 클래스 크기 가져 오기
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    # 2D ConvNet 입력 레이어에 공급할 입력 이미지 크기
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], '훈련 샘플')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes, activation='sigmoid'))
    #model.compile(loss='categorical_crossentropy',
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary()

    # 정확도가 10 에포크에서 0.005 이상으로 변경되지 않으면 훈련을 중단
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # TensorBoard를 사용하여 그래픽 해석을위한 로그 파일 생성
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # ImageDataGenerator의 이미지 이동 범위
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # ImageDataGenerator를 사용하여 모델 맞추기
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_validation,y_validation))

    return (model)

def save_model(model, model_filename):
    '''
    모델 저장
    * param model: 저장할 모델
    * param model_filename: 파일 이름
    '''
    model.save('../models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'



############################################################




#######################################

if __name__ == '__main__':
    '''
    콘솔 커맨드 예제 :  python trainmodel.py bio_metadata.csv model50
    '''
    file_name = "binary_bio_data.csv"
    model_filename = "model_3000_1610_14"

    # 메타데이터 로딩
    df = pd.read_csv(file_name)

    # 메타 데이터를 필터링하여 원하는 파일만 검색
    filtered_df = getsplit.filter_df(df)

    # filtered_df = filter_df(df)
    # print(filtered_df)
    # print("filterd df is empty {}".format(filtered_df))

    # Train test 나누기
    X_train, X_test, y_train, y_test = getsplit.split_people(filtered_df)


    # 자료수 얻기
    train_count = Counter(y_train)  # 출력 예시 : Counter({'Jeollado': 1815, 'gyeonggido': 692})
    test_count = Counter(y_test)    # 출력 예시 : Counter({'Jeollado': 769, 'gyeonggido': 306})
    print("y_train count : {}".format(train_count))
    print("y_test count : {}".format(test_count))

    print("main 구동!!!")
    # import ipdb;
    # ipdb.set_trace()

    # 출력 예시 : test_count.values()은 dict_values([769, 306])
    # 출력 예시 :  float(np.sum(list(test_count.values())))
    # 출력 예시 :  test_count.most_common()는 [('Jeollado', 769), ('gyeonggido', 306)]
    # acc_to beat : 0.7153488372093023
    acc_to_beat = test_count.most_common(1)[0][1] / float(np.sum(list(test_count.values())))

    # keras.np_utils.categorical()을 사용하여 원핫인코딩(One-Hot-Encoding)로 변환
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # 멀티 프로세싱을 사용하여 리샘플링 된 WAV 파일 가져 오기
    if DEBUG:
        print('wav 파일 읽는 중.')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)
    print('WAV 파일 로딩 완료!')

    # # 무음 제거
    # if DEBUG:
    #     print('일정 임계치에 대한 무음 제거하는 중 ...')
    # X_train = pool.map(remove_silence, X_train)
    # X_test = pool.map(remove_silence, X_test)
    # print('무음 제거 완료!')


    # MFCC로 변환
    if DEBUG:
        print('MFCC로 변환 중 ...')
    X_train = pool.map(to_mfcc, X_train)
    X_test = pool.map(to_mfcc, X_test)
    print('MFCC로 변환 완료!')


    # MFCC 정규화 - 기능 새로 추가
    if DEBUG:
        print('MFCC 정규화 중 ...')
    x_train = pool.map(normalize_mfcc, X_train)
    X_test = pool.map(normalize_mfcc, X_test)
    print('MFCC로 정규화 완료!')


    # MFCC에서 세그먼트 생성
    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)
    print('MFCC 세그먼트 생성 종료!')


    # 훈련 세그먼트 randomize화
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)
    print('훈련 세그먼트 randomize 완료!')

    # 학습 모델
    model = train_model(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation))
    print('학습 완료!')

    # 모든 X_test MFCC에 대한 예측
    y_predicted = accuracy.predict_class_all(create_segmented_mfccs(X_test), model)
    print('모든 X_test MFCC에 대한 예측 완료!')

    # Print statistics
    print('\n=====================================================================================')
    print('==================================== 최종 보고서 ====================================')
    print('=====================================================================================\n')
    print('Training samples:', train_count)
    print('Testing samples:', test_count)
    print('Accuracy to beat:', acc_to_beat)
    print('Confusion matrix of total samples:\n', np.sum(accuracy.confusion_matrix(y_predicted, y_test), axis=1))
    print('Confusion matrix:\n', accuracy.confusion_matrix(y_predicted, y_test))
    print('Accuracy:', accuracy.get_accuracy(y_predicted, y_test))
    print('\n=====================================================================================')
    print('=====================================================================================')
    print('=====================================================================================\n')


    # 모델 저장
    save_model(model, model_filename)
    print('\n{}에 모델 저장 완료'.format(model_filename))
