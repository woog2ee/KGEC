import pandas as pd
import time

import ast
import json
import pandas as pd
from tqdm import tqdm
import torch
from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool

from multiprocessing import Manager



# 조화 평균
def harmonic_mean(a, b):
    inv_a = 1 / a
    inv_b = 1 / b
    mean_inv = (inv_a + inv_b) / 2
    harmonic_mean = 1 / mean_inv
    return harmonic_mean



def rank_augmented_sents(model, origin, augmented):
    '''
        model: origin sent <-> noised sent similarity 구할 pretrained model
        origin / augmented: 원래 문장 1개 / 노이즈 문장 50개
    '''
    # levenshtein dist
    dists = [distance(origin, a) for a in augmented]

    # cos sim
    sents = [origin] + augmented
    vectors = model.encode(sents)
    sims = list(util.cos_sim(vectors, vectors)[0])
    sims = [float(s) for s in sims]
    del sims[0]
    
    # difficulty: levenshtein dist & cos sim 조화 평균
    diffs = [harmonic_mean(1/d, s) if d != 0
             else 1
             for d, s in zip(dists, sims)]
    
    # difficulty로 sorting해서 'info' 리스트에 저장
    # 하는 김에 구해놓은 difficulty까지 저장
    info = [[diffs[i], i] for i in range(len(diffs))]
    info.sort(key=lambda x: x[0], reverse=True)
    info = [info[i] + [augmented[info[i][1]]] for i in range(len(info))]
    return info



def make_doc(idx, ns):
    '''
        idx: 인덱싱할 데이터프레임 위치
        ns: 원래는 여기에 df, model을 멀티프로세싱 namespace로 활용하려 했으나 이슈가 생겼음
    '''
    # 멀티프로세싱할 때 이슈됐던 부분 (df, model의 값 변경 없이 인덱싱만 하거나, inference만 할거임)
    df, model = ns.df, ns.model

    # 인덱싱
    origin    = df.iloc[idx]['sents']
    augmented = df.iloc[idx]['augmented']

    # 한 쌍 (문장 1개 - 노이즈 문장 50개) 마다 하나의 'document' dictionary 생성
    try:
        augmented = ast.literal_eval(str(augmented))  # 가끔 더러운 문장 걸러주는 용도

        info = rank_augmented_sents(model, origin, augmented)
        document = {'origin'       : origin,
                    'origin length': len(origin),
                    'augmented'    : [info[i][2] for i in range(len(info))],
                    'difficulties' : [info[i][0] for i in range(len(info))]}
        return document
    except:
        return None  # 더러운 문장 들어오면 아무것도 안 내보냄



if __name__ == '__main__':
    # 데이터프레임 4개 불러서 4번 멀티프로세싱 필요
    # 다른 컬럼 다 필요없고 'sents', 'augmented' 컬럼 2개만 볼 예정
    # 데이터프레임 불러오는 자체도 시간이 좀 걸릴거에요... 하다가 다른 방법으로 저장하시는 것도 방법
    # kowiki_splited_cutoff_augmented.csv, modu_splited_cutoff_augmented1, 2, 3.csv
    s_time = time.time()
    filename = '/home/seunguk/KGEC/ranking/kowiki_splited_cutoff_augmented.csv'
    df = pd.read_csv(f'{filename}', encoding='utf-8-sig')
    print('dataset loaded! length ', len(df), time.time() - s_time)

    
    # cos sim 측정할 pretrained model
    # KR-SBERT: https://github.com/snunlp/KR-SBERT
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 
    print('model loaded!')


    # 멀티프로세싱하려고 namespace 지정한건데, 잘 된건지 확인 필요
    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = df
    ns.model = model


    # 데이터프레임 인덱스(i)랑 위에서 지정한 namespace(ns)의 리스트 만들어서, 멀티프로세싱 인풋 준비
    infos  = [(i, ns) for i in tqdm(range(len(df)))]
    inputs = zip(list(zip(*infos))[0], list(zip(*infos))[1])
    print('inputs are ready!')

    
    # 멀티프로세싱(이슈)
    ctx = mp.get_context('spawn')
    # with ctx.Pool(processes=20) as pool:
    #     results = pool.starmap(make_doc, tqdm(inputs, total=len(infos)))
    pool = ctx.Pool(processes=10)
    result = pool.starmap(make_doc, tqdm(inputs, total=len(infos)))
    print('tqdm ends')


    # make_doc의 결과 'result' 리스트에 
    # 한 쌍 (문장 1개 - 노이즈 문장 50개) 마다 하나의 'document' dictionary가 담김,
    # 그러나 make_doc의 try~except에서 더러운 문장도 바로 내보냈기에 빼줘야 함
    # 현재  result = [document1, document2, None, document3, None, document4 ...]
    # 원하는 result = [document1, document2, document3, document4 ...]
    while None in result: result.remove(None)  # 이거보다 효율적인 방법이 있을까요?
    

    # 결과 json으로 저장
    # 용량이 엄청 클거라 예상되는데, 혹시 다른 효율적인 저장 방법을 아신다면... 그거로 하셔도됩니다.
    dataset_dict = {'documents': result}
    with open('/home/seunguk/KGEC/ranking/kowiki_ranked.json', 'w') as f:
        json.dump(dataset_dict, f, ensure_ascii=False)  # ensure_ascii=False: 한국어로 저장
    print('saved!')
