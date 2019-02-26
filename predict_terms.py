import argparse
import joblib
from pyhocon import ConfigFactory
from numpy import float32
import torch
from Evaluator_terms import Evaluator
from utils import make_embedder
import numpy as np

doc = """ Given a model and some test queries, write predictions of
model on test queries. Added by johmy592"""

DEFAULT_TO_RANDOM_EMBEDDING = False

def _re_map_queries(q_embeds, candidate_embeds):
    """
    ADDED (MAY NOT BE NEEDED IT SEEMS)
    Re-map query ids to map to the new candidates, instead
    of full model vocabulary.
    """
    return [np.where(candidate_embeds==q)[0][0] for q in q_embeds]


if __name__ == "__main__":
    """
    TODO: How to make the model only predict within
    the domain vocabulary (i.e. the extracted terms)?
    Right now it uses the full vocabulary from training the
    word embeddings.
    """

    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("path_model", help="path of model")
    msg = ("path of pickle file containing test_data")
    parser.add_argument("path_data", help=msg)
    parser.add_argument("path_output", help="path where we write predictions on test set")
    parser.add_argument("-s", "--seed", type=int, default=91500)
    args = parser.parse_args()

    print("Loading model <-- {}".format(args.path_model))
    model = torch.load(args.path_model)
    model_vocab_size = model.get_nb_candidates()
    print("Size of model's vocab (nb_candidates): {}".format(model_vocab_size))

    print("Loading test data <-- {}".format(args.path_data))
    data = joblib.load(args.path_data)

    candidates = data["candidates"]

    test_q_cand_ids = data["test_query_cand_ids"]
    test_q_embed = make_embedder(data["test_query_embeds"], grad=False,
                                 cuda=model.use_cuda, sparse=False)

    #print(candidates[:10])
    #print(data["test_query_embeds"][:3])
    #print("CAND: ",candidate_embeds.weight.shape[0])

    # Make list of test query IDs
    print("Nb test queries: {}".format(test_q_embed.weight.shape[0]))

    # ADDED STUFF
    candidates_e = data["candidate_embeds"]
    candidate_embeds =  make_embedder(candidates_e, grad=False,
                                 cuda=model.use_cuda, sparse=False)
    #print("ids: ", test_q_cand_ids[:10])
    #print("new ids: ", _re_map_queries(data["test_query_embeds"], candidates_e))
    #***********************

    # Write predictions on test set
    print("Writing predictions on test set ---> {}".format(args.path_output))
    test_eval = Evaluator(model, test_q_embed, test_q_cand_ids, candidate_embeds)
    test_eval.write_predictions(args.path_output, candidates)

    print("Done.\n")
