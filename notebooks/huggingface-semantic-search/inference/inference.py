# https://github.com/aws/sagemaker-inference-toolkit
    
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import tqdm
import time
import json
import pickle

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors



def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model cannot be provided.
    Users should provide customized model_fn() in script.
    
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_dir)
    model.to(device)
    return model

def __embed_docs():
    docs = set()
#    with open('../inference/paragraph_emb.pkl', "rb") as fIn:
#    with open('../inference/embed_support_titles.pkl', "rb") as fIn:
#    with open('./inference/embed_support_titles.pkl', "rb") as fIn:
    with open('embed_support_titles.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        support_titles = stored_data['titles']
        support_titles_embed = stored_data['embeddings']
    
    return support_titles, support_titles_embed

        
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    return input_data


def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.

    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn

    Returns: a prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_emb = model.encode(data, convert_to_tensor=True)
    embeddings = __embed_docs()[1]
    
    return util.semantic_search(query_emb, embeddings, top_k=10)[0]


def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns: output data serialized
    """
    support_titles = __embed_docs()[0]
    output = []
    for hit in prediction:
#        doc = docs_title[hit['corpus_id']]
#        output.append([hit['score'], doc])
        doc = support_titles[hit['corpus_id']]
        output.append({"score": hit['score'], "title": doc})

    return json.dumps(output)

