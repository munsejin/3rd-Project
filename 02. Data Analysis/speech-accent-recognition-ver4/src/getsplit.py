import pandas as pd
import sys
from sklearn.model_selection import train_test_split




# def filter_df(df):
#     '''
#     df 열을 기준으로 오디오 파일을 필터링하는 기능
#     df 컬럼 옵션
#       * age 나이
#       * age_of_english_onset 한국어 사용 기간
#       * age_sex, 나이_성별
#       * birth_place 태어난 장소
#       * english_learning_method 한국어 교육 여부 (교육기관, 네거티브)
#       * english_residence 한국어 사용지역 여부
#       * length_of_english_residence 현재지역 거주 기간
#       * native_language  출신 지역 사투리 사용
#       * other_languages 타지역 사투리 사용
#       * sex 성별
#     '''

def filter_df(df):
    '''
        지역별 DataFrame  필터링
          * param df (DataFrame): 필터링되지 않은 전체 DataFrame
          * return (DataFrame): 필터링된 DataFrame
    '''

    gyeonggido = df[df.native_language == 'gyeonggido']
    #gyeongsangdo = df[df.native_language == 'gyeongsangdo']
    jeollado = df[df.native_language == 'jeollado']

    df = df.append(gyeonggido)
    #df = df.append(gyeongsangdo)
    df = df.append(jeollado)

    return df





def split_people(df,test_size=0.3):
    '''
    DataFrame을 Train, Test 데이터로 분할
    * param df (DataFrame): 분할할 오디오 파일의 DataFrame
    * param test_size (float): 테스트로 분할할 총파일의 백분율
    * return X_train, X_test, y_train, y_test (tuple): X는 df['사투리의 갯수'] 그리고 Y는 df['네거티브_언어']
    '''


    return train_test_split(df['language_num'],df['native_language'],test_size=test_size,random_state=1234)


if __name__ == '__main__':
    '''
    콘솔 커맨드 예제 :  python binary_bio_data.csv
    '''

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    filtered_df = filter_df(df)
    print(split_people(filtered_df))
