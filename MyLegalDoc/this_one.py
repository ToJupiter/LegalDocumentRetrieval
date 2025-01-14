# # %%
# !pip install -U sentence-transformers
# !pip install - q cupy

# # %%
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import cupy as cp
# import numpy as np

# # %% [markdown]
# # ### Import data

# # %%
# df = pd.read_csv('/kaggle/input/zaloai2021-legal-text-retrieval/legal_corpus_original.csv') #using splited version for better encoding
# df.head()

# # %% [markdown]
# # ### Preprocessing data

# # %%
# df = df.dropna(subset=['content'])
# df.head()

# # %%
# df['id'] = [i for i in range(len(df))]
# df = df[:10]
# df.info()

# # %%
# df.loc[:, 'law_article'] = df.apply(lambda row: {'law_id': row['law_title'], 'article_id': str(row['article_id'])}, axis=1)
# df.head()

# # %% [markdown]
# # ### Document Encoding

# # %%
# # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
# sentences = ["Cô ấy là một người vui_tính .", "Cô ấy cười nói suốt cả ngày ."]

# model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
# embeddings = model.encode(sentences)
# print(embeddings)


# # %%
# def encode(lst = [], convert_to_tensor=True):
#     encoded_vectors = model.encode(lst, convert_to_tensor=True)
#     encoded_vectors_cpu = cp.asnumpy(encoded_vectors)  # Convert to NumPy array
#     vector_arr = [np.array(arr) for arr in encoded_vectors.cpu().detach().numpy()]
#     return vector_arr

# # %%
# df['vector'] = encode(lst = list(df['content']))
# df.head()

# # %%
# df.to_json('/kaggle/working/legal_copus_data.json')

# %%
df = pd.read_json('/kaggle/working/legal_copus_data.json')
df

# %% [markdown]
# ### Train & Test

# %%
# df_train_test = pd.read_csv('/kaggle/input/zaloai2021-legal-text-retrieval/train_qna.csv')
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Assume df is your DataFrame
# # Splitting the DataFrame into 80% train and 20% test
# train_df, test_df = train_test_split(df_train_test, test_size=0.2, random_state=42)
# test_df['id'] = [i for i in range(len(test_df))]
# # Printing the sizes of train and test subsets
# print("Train subset size:", len(train_df))
# print("Test subset size:", len(test_df))

# # %%
# test_df['vector'] = encode(lst = list(test_df['question']))

# # %%
# test_df.head()

# %% [markdown]
# ### Search by comparing similarity

# %%
import cupy as cp
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# # Sample query and document dataframes
# queries = pd.DataFrame({'id': [1, 2],
#                         'vector': [[1, 2, 3], [4, 5, 6]]})
# documents = pd.DataFrame({'id': [1, 2, 3, 4],
#                           'vector': [[[1, 2, 3]], [[5, 6, 7]],[[2, 3, 4]],[[8, 9, 10]]]})

def top_k_relevance(queries, documents, k):
    # Calculate similarity matrix using NumPy arrays
    similarity_matrix = cosine_similarity(np.vstack(queries['vector'].to_numpy()),
                                          np.vstack(documents['vector'].to_numpy()))
    
    top_k_matrix = {}
    for i in range(len(queries)):
        query_id = queries['id'].iloc[i]
        similarity_scores = similarity_matrix[i]
        top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
        top_k_doc_ids = documents['id'].iloc[top_k_indices].tolist()
        top_k_matrix[query_id] = top_k_doc_ids

    return top_k_matrix



# %%
top_k = 2
result = top_k_relevance(test_df, df, top_k)
# print(result)

# %% [markdown]
# ### Postprocess relevant document

# %%
test_df['proposed_relevance'] = [[df['law_article'].iloc[i] for i in result[q]] for q in result]


# %%
test_df.head()

# %%
test_df['proposed_relevance'].iloc[0] = [ {'law_id': '28/2020/nđ-cp', 'article_id': '21'}, {'law_id': '64/2020/qh14', 'article_id': '97'}]

# %% [markdown]
# ## TESTING

# %%
import numpy as np

def recall(test_df, top_k):
    assert top_k == len(test_df['proposed_relevance'].iloc[0]), "Wrong top k"
    
    def calculate_recall(row):
        relevant_documents = set([str(e) for e in eval(row['relevant_articles'])])

        retrieved_documents = set([str(i) for i in row['proposed_relevance']])
        return len(relevant_documents.intersection(retrieved_documents)) / len(relevant_documents) if len(relevant_documents) > 0 else 0

    test_df['recall'] = test_df.apply(calculate_recall, axis=1)
    
    return np.mean(test_df['recall'])

recall_score = recall(test_df, top_k)
print("Recall Score:", recall_score)
test_df.head()

# %%
def precision(test_df, top_k):
    assert top_k == len(test_df['proposed_relevance'].iloc[0]), "Wrong top k"
    
    def calculate_precision(row):
        relevant_documents = set([str(e) for e in eval(row['relevant_articles'])])
        retrieved_documents = set([str(i) for i in row['proposed_relevance']])
        return len(relevant_documents.intersection(retrieved_documents)) / len(retrieved_documents) if len(retrieved_documents) > 0 else 0

    test_df['precision'] = test_df.apply(calculate_precision, axis=1)
    
    return np.mean(test_df['precision'])

precision_score = precision(test_df, top_k)
print("Precision Score:", precision_score)
test_df.head()


# %%
def f1(test_df, top_k):
    assert top_k == len(test_df['proposed_relevance'].iloc[0]), "Wrong top k"
    
    def calculate_f1(row):
        precision = row['precision']
        recall = row['recall']
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    test_df['f1'] = test_df.apply(calculate_f1, axis=1)
    
    return np.mean(test_df['f1'])

def f2(test_df, top_k):
    assert top_k == len(test_df['proposed_relevance'].iloc[0]), "Wrong top k"
    
    def calculate_f2(row):
        precision = row['precision']
        recall = row['recall']
        if precision + recall == 0:
            return 0
        return (5 * precision * recall) / (4 * precision + recall)

    test_df['f2'] = test_df.apply(calculate_f2, axis=1)
    
    return np.mean(test_df['f2'])


f1_score = f1(test_df, top_k)
f2_score = f2(test_df, top_k)

print("F1 Score:", f1_score)
print("F2 Score:", f2_score)


# %% [markdown]
# ### mAP 

# %% [markdown]
# This metric can not judge which is relevant more or less

# %%
def mAP(test_df, top_k):
    def calculateAP(row):
        relevant_documents = set([str(e) for e in eval(row['relevant_articles'])])
        ap_num = 0
        for k in range(1,top_k+1):
            
            retrieved_documents = set([str(i) for i in row['proposed_relevance'][:k]])
            precision_k = len(relevant_documents.intersection(retrieved_documents)) / len(retrieved_documents) if len(retrieved_documents) > 0 else 0
            retrieved = [str(i) for i in row['proposed_relevance']]
            if retrieved[k-1] in relevant_documents:
                rel_k = 1
            else:
                rel_k = 0
            
            ap_num += precision_k * rel_k
        return ap_num/len(relevant_documents)
    
    test_df['AP'] = test_df.apply(calculateAP, axis=1)
    return np.mean(test_df['AP'])


print("mAP: ", mAP(test_df, top_k))
test_df.head()

# %% [markdown]
# ### Normalized Discounted Cumulative Gain

# %%
def get_relevance(relevant_articles:list, proposed_relevance:list):
    '''
    Input: list of full value (not need to be unique). In this case 2 paragraph in same lawid can be same relevance score, also consider relevance to each other while with precision, we dont care 
    Output: relevance = [0, 1, 2, 4, 6, 1, 4, 7]'''
    relevance = []
    for art in proposed_relevance:
        try:
            relevance.append(len(relevant_articles) - relevant_articles.index(art) )
        except ValueError:
            relevance.append(0)
    return relevance
relevant_articles = [3,4]
proposed_relevance = [2,1,3,5,4,5]
get_relevance(relevant_articles, proposed_relevance)


# %%
def NDCG_K(test_df, top_k):
    K = top_k
    
    def calculateNDCG(row):
        relevant_documents = [str(e) for e in eval(row['relevant_articles'])]
        retrieved_documents = [str(i) for i in row['proposed_relevance']]
        relevance = get_relevance(relevant_documents,retrieved_documents)
        ideal_relevance = sorted(relevance, reverse=True)
        dcg = 0
        idcg = 0
        for k in range(1, K+1):
            # calculate rel_k values
            rel_k = relevance[k-1]
            ideal_rel_k = ideal_relevance[k-1]
            # calculate dcg and idcg
            dcg += rel_k / log2(1 + k)
            idcg += ideal_rel_k / log2(1 + k)
            # calcualte ndcg
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg
    test_df['NDCG'] = test_df.apply(calculateNDCG, axis=1)
    return np.mean(test_df['NDCG'])
print("NDCG: ", NDCG_K(test_df, top_k))
test_df.head()


