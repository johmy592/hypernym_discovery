import argparse
import joblib
from pyhocon import ConfigFactory
from numpy import float32
import torch
from Evaluator_terms import Evaluator
from utils import make_embedder


doc = """ Given a model and some test queries, write predictions of
model on test queries. Added by johmy592"""

DEFAULT_TO_RANDOM_EMBEDDING = False

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
    #candidates_e = data["candidate_embeds"]
    candidates = data["candidates"]


    test_q_cand_ids = data["test_query_cand_ids"]
    test_q_embed = make_embedder(data["test_query_embeds"], grad=False,
                                 cuda=model.use_cuda, sparse=False)

    print(candidates[:10])
    print(test_q_cand_ids[:10])
    #candidate_embeds =  make_embedder(candidates_e, grad=False,
    #                             cuda=model.use_cuda, sparse=False)
    #print("CAND: ",candidate_embeds.weight.shape[0])

    # Make list of test query IDs
    print("Nb test queries: {}".format(test_q_embed.weight.shape[0]))

    # Write predictions on test set
    print("Writing predictions on test set ---> {}".format(args.path_output))
    test_eval = Evaluator(model, test_q_embed, test_q_cand_ids, candidate_embeds)
    test_eval.write_predictions(args.path_output, candidates)

    print("Done.\n")
