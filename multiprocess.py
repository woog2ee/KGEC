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



def harmonic_mean(a, b):
    inv_a = 1 / a
    inv_b = 1 / b
    mean_inv = (inv_a + inv_b) / 2
    harmonic_mean = 1 / mean_inv
    return harmonic_mean



def rank_augmented_sents(model, origin, augmented):
    # Get distances between origin and augmented sentences
    dists = [distance(origin, a) for a in augmented]
       
    # Get similarities between origin and augmented sentences
    sents = [origin] + augmented
    vectors = model.encode(sents)
    sims = list(util.cos_sim(vectors, vectors)[0])
    sims = [float(s) for s in sims]
    del sims[0]
    
    # Calculate difficulties using harmonic means between distance & similarity
    difficulties = [harmonic_mean(1/d, s) if d != 0
                    else 1
                    for d, s in zip(dists, sims)]
    
    # Rank by each sentence's difficulty
    info = [[dists[i], sims[i], difficulties[i], i] for i in range(len(difficulties))]
    info.sort(key=lambda x: x[2], reverse=True)
    info = [info[i] + [augmented[info[i][3]]] for i in range(len(info))]
    return info



def make_doc(idx, ns):
    df, model = ns.df, ns.model

    origin    = df.iloc[idx]['sents']
    augmented = df.iloc[idx]['augmented']

    try:
        augmented = ast.literal_eval(str(augmented))

        info = rank_augmented_sents(model, origin, augmented)
        document = {'origin'       : origin,
                    'origin length': len(origin),
                    'augmented'    : [info[i][4] for i in range(len(info))],
                    'distances'    : [info[i][0] for i in range(len(info))],
                    'similarities' : [info[i][1] for i in range(len(info))],
                    'difficulties' : [info[i][2] for i in range(len(info))],
                    'resource'     : 'kowiki'}
        return document
    except:
        return f'# {idx}, {origin}, {augmented}'



if __name__ == '__main__':
    s_time = time.time()
    filename = '/home/seunguk/KGEC/ranking/kowiki_splited_cutoff_augmented.csv'
    df = pd.read_csv(f'{filename}', encoding='utf-8-sig')
    print('dataset loaded! length ', len(df), time.time() - s_time)

    # KR-SBERT: https://github.com/snunlp/KR-SBERT
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 
    print('model loaded!')

    

    # namespace
    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = df
    ns.model = model

    infos  = [(i, ns) for i in tqdm(range(len(df)))]
    inputs = zip(list(zip(*infos))[0], list(zip(*infos))[1])
    print('inputs are ready!')

    

    ctx = mp.get_context('spawn')
    # with ctx.Pool(processes=20) as pool:
    #     results = pool.starmap(make_doc, tqdm(inputs, total=len(infos)))
    pool = ctx.Pool(processes=10)
    result = pool.starmap(make_doc, tqdm(inputs, total=len(infos)))
    print('tqdm ends')


    # Add ensure_ascii=False
    dataset_dict = {'documents': result}
    with open('/home/seunguk/KGEC/ranking/kowiki_ranked.json', 'w') as f:
        json.dump(dataset_dict, f, ensure_ascii=False)
    print('saved!')
